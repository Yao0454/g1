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
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from .bridge import Bridge
from .stream import MjpegStream

# ── 关键点索引（COCO 17 点）─────────────────────────────────────────────────
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

# ── 阈值 ─────────────────────────────────────────────────────────────────────
TARGET_DIST = 1.0         # m，跟随的目标距离
OBSTACLE_DIST = 0.6       # m，中央深度小于这个就算被挡
ROI_TOP, ROI_BOTTOM = 0.35, 0.75  # 障碍 ROI 在画面纵向的位置
MIN_PERSON_AREA = 8000    # 像素²，真人 bbox 至少这么大（小于这个当机器人自己的手）
KP_CONF, ARM_CONF = 0.5, 0.6

ARM_ACTIONS = {
    "left":  (26, "挥手"),
    "right": (27, "握手"),
    "both":  (15, "双手举起"),
}


class Vision:
    def __init__(self, bridge: Bridge, *,
                 model_dir: Path | None = None,
                 stream: MjpegStream | None = None,
                 follow: bool = True,
                 gesture: bool = True,
                 action_cooldown: float = 4.0,
                 verbose: bool = False):
        self.bridge = bridge
        self.stream = stream
        self.follow = follow
        self.gesture = gesture
        self.cooldown = action_cooldown
        self.verbose = verbose
        # 模型从这个目录找（默认 ~/g1，即 g1.py / talk.py 同级）
        self.model_dir = model_dir or Path(__file__).resolve().parent.parent

        self._stop = threading.Event()
        self._last_action_time = 0.0
        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None
        self._model: YOLO | None = None

    # ── 启动/停止 ───────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        pose_pt = self.model_dir / "yolov8n-pose.pt"
        pose_engine = self.model_dir / "yolov8n-pose.engine"
        if not pose_engine.exists() and pose_pt.exists():
            print(f"[VISION] {pose_engine.name} 不存在，从 {pose_pt.name} export TensorRT …")
            YOLO(str(pose_pt)).export(format="engine", half=True)
        path = str(pose_engine if pose_engine.exists() else pose_pt)
        print(f"[VISION] 加载 {Path(path).name} …", flush=True)
        self._model = YOLO(path, task="pose")
        print("[VISION] 热身 …", flush=True)
        self._model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)
        print("[VISION] 模型就绪")

    def _open_camera(self) -> None:
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
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
    def _has_lower_body(cls, keypoints) -> bool:
        for idx in (KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE):
            if cls._kp(keypoints, idx) is not None:
                return True
        return False

    @classmethod
    def _is_real_person(cls, box, keypoints) -> bool:
        x1, y1, x2, y2 = (float(box.xyxy[0][i]) for i in range(4))
        if (x2 - x1) * (y2 - y1) < MIN_PERSON_AREA:
            return False
        if keypoints is not None and not cls._has_lower_body(keypoints):
            return False
        return True

    @classmethod
    def _detect_gesture(cls, keypoints) -> str | None:
        lw = cls._kp(keypoints, KP_L_WRIST, ARM_CONF)
        rw = cls._kp(keypoints, KP_R_WRIST, ARM_CONF)
        if lw and rw:
            return "both"
        if lw:
            return "left"
        if rw:
            return "right"
        return None

    @staticmethod
    def _check_obstacles(depth_frame, w: int, h: int):
        depth = np.asanyarray(depth_frame.get_data()).astype(float) / 1000.0
        roi = depth[int(h * ROI_TOP):int(h * ROI_BOTTOM), :]

        def min_valid(z):
            v = z[(z > 0.1) & (z < 5.0)]
            return float(np.percentile(v, 5)) if v.size else 5.0

        return (min_valid(roi[:, :w // 3]),
                min_valid(roi[:, w // 3:2 * w // 3]),
                min_valid(roi[:, 2 * w // 3:]))

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

                # 推理
                pose_res = self._model(frame, verbose=False)[0]
                annotated = pose_res.plot()

                gesture = None
                follow_dist = None
                follow_err = None
                boxes = pose_res.boxes
                kpts_all = pose_res.keypoints

                for i, box in enumerate(boxes):
                    if int(box.cls) != 0:  # 只看 person
                        continue
                    kpts = kpts_all[i] if (kpts_all is not None and i < len(kpts_all)) else None

                    if not self._is_real_person(box, kpts):
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, "robot arm", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        continue

                    if self.gesture and kpts is not None and gesture is None:
                        gesture = self._detect_gesture(kpts)

                    if self.follow and follow_dist is None:
                        cx = float(box.xywh[0][0])
                        cy = float(box.xywh[0][1])
                        d = depth.get_distance(int(cx), int(cy))
                        if d > 0:
                            follow_dist = d
                            follow_err = (cx - w / 2) / (w / 2)

                # 下发跟随
                if self.follow and follow_dist is not None and follow_dist > 0.1:
                    self.bridge.send_move(follow_dist, follow_err, blocked)

                # 下发手势（有冷却）
                now = time.time()
                if (self.gesture and gesture
                        and (now - self._last_action_time) > self.cooldown):
                    self._last_action_time = now
                    action_id, label = ARM_ACTIONS[gesture]
                    print(f"[VISION] 手势 {gesture} → {label} (id={action_id})")
                    self.bridge.send_arm(action_id)

                # HUD
                bar = (0, 0, 255) if blocked else (0, 255, 0)
                cv2.putText(annotated, f"L:{dist_l:.1f} C:{dist_c:.1f} R:{dist_r:.1f}m",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar, 2)
                if follow_dist:
                    cv2.putText(annotated, f"follow={follow_dist:.2f}m",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if gesture:
                    cv2.putText(annotated, f"ARM: {gesture.upper()}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                if self.stream is not None:
                    self.stream.update(annotated)

                if self.verbose:
                    cd = max(0.0, self.cooldown - (now - self._last_action_time))
                    print(f"[VISION] L{dist_l:.1f} C{dist_c:.1f} R{dist_r:.1f} "
                          f"blk={int(blocked)} follow={follow_dist} ges={gesture} cd={cd:.1f}",
                          flush=True)
        except KeyboardInterrupt:
            print("\n[VISION] Ctrl-C 退出")
        finally:
            self.close()
