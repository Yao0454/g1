"""
core/vision.py — 视觉模块：人体跟随 + 手势识别（一份 YOLO-pose 推理两用）。

封装成 Vision 类，外面只需要：
    bridge = Bridge()
    vision = Vision(bridge, stream=MjpegStream(6769))
    vision.start()          # 起 MJPEG 流（如果有）
    vision.run()            # 阻塞主循环；Ctrl-C 退
    # vision.stop()         # 异步退出（外部 thread 调）

它内部：
    · RealSense 拿对齐后的 color+depth
    · 单一 YOLO-pose 推理：拿到 boxes + keypoints
    · 过滤机器人自己的手（小 bbox / 没下肢关键点） → 选最近真人当跟随目标
    · 手腕被检出 → 触发握手/挥手手势（带 cooldown）
    · 深度图中 ROI 算左中右最近距离 → blocked 标志
    · 把结果 → bridge.send_move(dist, err_norm, blocked) + bridge.send_arm(action_id)
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from .bridge import Bridge
from .stream import MjpegStream

# ── 关键点索引（COCO 17 点）─────────────────────────────────────────────────
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

# ── 阈值 ─────────────────────────────────────────────────────────────────────
TARGET_DIST = 1.0  # m，跟随的目标距离
OBSTACLE_DIST = 0.6  # m，中央深度小于这个就算被挡
ROI_TOP, ROI_BOTTOM = 0.35, 0.75  # 障碍 ROI 在画面纵向的位置
MIN_PERSON_AREA = 8000  # 像素²，真人 bbox 至少这么大（小于这个当机器人自己的手）
KP_CONF, ARM_CONF = 0.5, 0.6
MAX_FOLLOW_DIST = 3.0  # m，超过这个距离的人不锁
ARM_SEG_MIN_FRAC = 0.02  # 大/小臂在画面里最短要这个比例（≈20px @ 1080p）

ARM_ACTIONS = {
    "left": (26, "挥手"),
    "right": (27, "握手"),
    "both": (15, "双手举起"),
}


class Vision:
    def __init__(
        self,
        bridge: Bridge,
        *,
        model_dir: Path | None = None,
        stream: MjpegStream | None = None,
        follow: bool = True,
        gesture: bool = True,
        action_cooldown: float = 4.0,
        max_dist: float = MAX_FOLLOW_DIST,
        snapshot_dir: Path | None = None,
        snapshot_every: float = 2.0,
        verbose: bool = False,
    ):
        self.bridge = bridge
        self.stream = stream
        self.follow = follow
        self.gesture = gesture
        self.cooldown = action_cooldown
        self.max_dist = max_dist
        self.verbose = verbose
        # 模型从这个目录找（默认 ~/g1，即 g1.py / talk.py 同级）
        self.model_dir = model_dir or Path(__file__).resolve().parent.parent

        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.snapshot_every = max(0.1, float(snapshot_every))
        self._last_snapshot_time = 0.0
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            print(f"[VISION] 帧快照 → {self.snapshot_dir} (每 {self.snapshot_every:.1f}s)")

        self._stop = threading.Event()
        self._last_action_time = 0.0
        self._fps = 0.0
        self._last_frame_time: float | None = None
        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None
        self._model: YOLO | None = None

    # ── 启动/停止 ───────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        pose_pt = self.model_dir / "yolov8n-pose.pt"
        pose_engine = self.model_dir / "yolov8n-pose.engine"
        if not pose_engine.exists() and pose_pt.exists():
            print(
                f"[VISION] {pose_engine.name} 不存在，从 {pose_pt.name} export TensorRT …"
            )
            YOLO(str(pose_pt)).export(format="engine", half=True)
        path = str(pose_engine if pose_engine.exists() else pose_pt)
        print(f"[VISION] 加载 {Path(path).name} …", flush=True)
        self._model = YOLO(path, task="pose")
        print("[VISION] 热身 …", flush=True)
        self._model(np.zeros((1080, 1920, 3), dtype=np.uint8), verbose=False)
        print("[VISION] 模型就绪")

    def _open_camera(self) -> None:
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)
        print("[VISION] RealSense 已启动")

    def start(self) -> None:
        """打开模型、相机、MJPEG 流。"""
        if self._model is None:
            self._load_model()
        if self._pipeline is None:
            self._open_camera()
        if self.stream is not None:
            self.stream.start()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    # ── 内部工具 ────────────────────────────────────────────────────────────
    @staticmethod
    def _kp(keypoints, idx, min_conf=KP_CONF):
        try:
            data = keypoints.data
            pt = data[0][idx] if data.ndim == 3 else data[idx]
            x, y, c = float(pt[0]), float(pt[1]), float(pt[2])
            return (x, y, c) if c >= min_conf else None
        except (IndexError, TypeError, AttributeError):
            return None

    @classmethod
    def _has_kp(cls, keypoints, *idxs) -> bool:
        for idx in idxs:
            if cls._kp(keypoints, idx) is not None:
                return True
        return False

    @classmethod
    def _is_real_person(cls, box, keypoints, frame_h: int) -> bool:
        """真人 = bbox 够大 + 看得到肩 + 看得到髋。
        机器人自己伸进 FOV 的手只能拍到前臂/手，既没肩也没髋。
        坐着的人（腿被桌挡）仍有肩和髋 → 不会被误判。
        """
        x1, y1, x2, y2 = (float(box.xyxy[0][i]) for i in range(4))
        if (x2 - x1) * (y2 - y1) < MIN_PERSON_AREA:
            return False
        if keypoints is None:
            return True  # 没关键点信息就只信 bbox 大小
        if not cls._has_kp(keypoints, KP_L_SHOULDER, KP_R_SHOULDER):
            return False
        if not cls._has_kp(keypoints, KP_L_HIP, KP_R_HIP):
            return False
        return True

    @staticmethod
    def _arm_extended(shoulder, elbow, wrist, frame_h: int) -> bool:
        """伸手握手姿势：大臂自然垂下、小臂往外伸 ≈90°。

        几何条件（图像坐标，y 向下增加）：
          - 大臂朝下：肘 y > 肩 y，且 |elbow.x - shoulder.x| < (elbow.y - shoulder.y)
          - 小臂偏水平：|wrist.x - elbow.x| > |wrist.y - elbow.y|
          - 大/小臂都得有最小长度，过滤识别噪声
        手自然下垂（大臂小臂都竖）→ 小臂条件不满足；高举过头（肘在肩上）→ 大臂条件不满足。
        """
        if shoulder is None or elbow is None or wrist is None:
            return False
        sx, sy, _ = shoulder
        ex, ey, _ = elbow
        wx, wy, _ = wrist
        min_seg = max(15.0, frame_h * ARM_SEG_MIN_FRAC)

        ua_dx = abs(ex - sx)
        ua_dy = ey - sy
        if ua_dy < min_seg or ua_dy < ua_dx:  # 大臂没朝下
            return False

        fa_dx = abs(wx - ex)
        fa_dy = abs(wy - ey)
        if fa_dx < min_seg * 1.5:  # 小臂没向外伸够长
            return False
        if fa_dy > fa_dx:  # 小臂还是竖的，不是水平
            return False
        return True

    @classmethod
    def _detect_gesture(cls, keypoints, frame_h: int) -> str | None:
        ls = cls._kp(keypoints, KP_L_SHOULDER)
        rs_ = cls._kp(keypoints, KP_R_SHOULDER)
        le = cls._kp(keypoints, KP_L_ELBOW)
        re = cls._kp(keypoints, KP_R_ELBOW)
        lw = cls._kp(keypoints, KP_L_WRIST, ARM_CONF)
        rw = cls._kp(keypoints, KP_R_WRIST, ARM_CONF)

        left = cls._arm_extended(ls, le, lw, frame_h)
        right = cls._arm_extended(rs_, re, rw, frame_h)

        if left and right:
            return "both"
        if left:
            return "left"
        if right:
            return "right"
        return None

    @staticmethod
    def _depth_median(depth_frame, box, w: int, h: int, sample_frac: float = 0.15):
        """取 bbox 中心一小块的有效深度中位数，单点 get_distance 容易踩到 0 或穿透。"""
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        sx = max(2, int(bw * sample_frac))
        sy = max(2, int(bh * sample_frac))
        px1 = max(0, int(cx - sx))
        py1 = max(0, int(cy - sy))
        px2 = min(w, int(cx + sx))
        py2 = min(h, int(cy + sy))
        if px2 <= px1 or py2 <= py1:
            return None
        arr = np.asanyarray(depth_frame.get_data())[py1:py2, px1:px2].astype(float) / 1000.0
        v = arr[(arr > 0.2) & (arr < 8.0)]
        if v.size < 5:
            return None
        return float(np.median(v))

    @staticmethod
    def _check_obstacles(depth_frame, w: int, h: int):
        depth = np.asanyarray(depth_frame.get_data()).astype(float) / 1000.0
        roi = depth[int(h * ROI_TOP) : int(h * ROI_BOTTOM), :]

        def min_valid(z):
            v = z[(z > 0.1) & (z < 5.0)]
            return float(np.percentile(v, 5)) if v.size else 5.0

        return (
            min_valid(roi[:, : w // 3]),
            min_valid(roi[:, w // 3 : 2 * w // 3]),
            min_valid(roi[:, 2 * w // 3 :]),
        )

    # ── 主循环 ──────────────────────────────────────────────────────────────
    def run(self) -> None:
        """阻塞主循环：30 FPS 跑相机 → YOLO-pose → bridge。Ctrl-C 退。"""
        self.start()
        assert self._model is not None and self._pipeline is not None
        try:
            while not self._stop.is_set():
                frames = self._pipeline.wait_for_frames()
                aligned = self._align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth:
                    continue
                frame = np.asanyarray(color.get_data())
                h, w = frame.shape[:2]

                # 障碍
                dist_l, dist_c, dist_r = self._check_obstacles(depth, w, h)
                blocked = dist_c < OBSTACLE_DIST

                # 推理：拉高 conf，过掉远处零散误检
                pose_res = self._model(frame, conf=0.5, verbose=False)[0]
                annotated = pose_res.plot(boxes=False)  # 自己画框，避免每个人都"亮起来"

                gesture = None
                follow_dist = None
                follow_err = None
                boxes = pose_res.boxes
                kpts_all = pose_res.keypoints

                # 1) 先把所有真人收集起来，算 bbox 中心 patch 的深度中位数
                candidates = []  # (depth, cx, box, kpts)
                for i, box in enumerate(boxes):
                    if int(box.cls) != 0:  # 只看 person
                        continue
                    kpts = (
                        kpts_all[i]
                        if (kpts_all is not None and i < len(kpts_all))
                        else None
                    )

                    if not self._is_real_person(box, kpts, h):
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            annotated,
                            "robot arm",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
                        continue

                    d = self._depth_median(depth, box, w, h)
                    cx = float(box.xywh[0][0])
                    if d is None:
                        # 深度无效，灰框标一下就跳过，不作为目标
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (160, 160, 160), 1)
                        cv2.putText(
                            annotated,
                            "no-depth",
                            (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (160, 160, 160),
                            1,
                        )
                        continue
                    if d > self.max_dist:
                        # 太远不锁，紫色细框标一下当参考
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 80, 200), 1)
                        cv2.putText(
                            annotated,
                            f"far {d:.1f}m",
                            (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (200, 80, 200),
                            1,
                        )
                        continue
                    candidates.append((d, cx, box, kpts))

                # 2) 只锁最近的一个，跟随 + 手势都用它
                candidates.sort(key=lambda c: c[0])
                target = candidates[0] if candidates else None

                # 3) 画框：未选中灰细框，选中绿粗框 + TARGET 标签
                for idx, (d, cx, box, _) in enumerate(candidates):
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    if idx == 0:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(
                            annotated,
                            f"TARGET {d:.2f}m",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (160, 160, 160), 1)
                        cv2.putText(
                            annotated,
                            f"{d:.2f}m",
                            (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (160, 160, 160),
                            1,
                        )

                if target is not None:
                    d, cx, _, kpts = target
                    if self.gesture and kpts is not None:
                        gesture = self._detect_gesture(kpts, h)
                    if self.follow:
                        follow_dist = d
                        follow_err = (cx - w / 2) / (w / 2)

                # 下发跟随
                if self.follow and follow_dist is not None and follow_dist > 0.1:
                    self.bridge.send_move(follow_dist, follow_err, blocked)

                # 下发手势（有冷却）
                now = time.time()
                if (
                    self.gesture
                    and gesture
                    and (now - self._last_action_time) > self.cooldown
                ):
                    self._last_action_time = now
                    action_id, label = ARM_ACTIONS[gesture]
                    print(f"[VISION] 手势 {gesture} → {label} (id={action_id})")
                    self.bridge.send_arm(action_id)

                # FPS (EMA)
                t = time.time()
                if self._last_frame_time is not None:
                    dt = t - self._last_frame_time
                    if dt > 1e-3:
                        inst = 1.0 / dt
                        self._fps = inst if self._fps <= 0 else (self._fps * 0.9 + inst * 0.1)
                self._last_frame_time = t

                # HUD
                cv2.putText(
                    annotated,
                    f"FPS {self._fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                bar = (0, 0, 255) if blocked else (0, 255, 0)
                cv2.putText(
                    annotated,
                    f"L:{dist_l:.1f} C:{dist_c:.1f} R:{dist_r:.1f}m",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    bar,
                    2,
                )
                if follow_dist:
                    cv2.putText(
                        annotated,
                        f"follow={follow_dist:.2f}m",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                if gesture:
                    cv2.putText(
                        annotated,
                        f"ARM: {gesture.upper()}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )

                if self.stream is not None:
                    self.stream.update(annotated)

                # 帧快照：拿来事后人工/我分析为啥识别不准
                if (
                    self.snapshot_dir is not None
                    and (time.time() - self._last_snapshot_time) > self.snapshot_every
                ):
                    self._last_snapshot_time = time.time()
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    raw_path = self.snapshot_dir / f"{ts}_raw.jpg"
                    ann_path = self.snapshot_dir / f"{ts}_ann.jpg"
                    cv2.imwrite(str(raw_path), frame)
                    cv2.imwrite(str(ann_path), annotated)

                if self.verbose:
                    cd = max(0.0, self.cooldown - (now - self._last_action_time))
                    print(
                        f"[VISION] L{dist_l:.1f} C{dist_c:.1f} R{dist_r:.1f} "
                        f"blk={int(blocked)} follow={follow_dist} ges={gesture} cd={cd:.1f}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\n[VISION] Ctrl-C 退出")
        finally:
            self.close()
