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

import math
import os
import queue
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
KP_NOSE = 0
KP_L_EYE, KP_R_EYE = 1, 2
KP_L_EAR, KP_R_EAR = 3, 4
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
KP_CONF, ARM_CONF = 0.5, 0.5  # 推理用关键点置信度阈值（握手姿势小臂常斜着，conf 拉太高会丢）
KP_CONF_PERSON = 0.3  # "是真人"的判据宽容点，边缘肩膀 conf 容易低
MAX_FOLLOW_DIST = 3.0  # m，超过这个距离的人不锁
WRIST_FORWARD_M = 0.15  # 手腕比身体（bbox 深度）近相机 ≥ 此米数 → 伸手

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
        self._model(np.zeros((720, 1280, 3), dtype=np.uint8), verbose=False)
        print("[VISION] 模型就绪")

    def _open_camera(self) -> None:
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
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
    def _has_kp(cls, keypoints, *idxs, min_conf=KP_CONF) -> bool:
        for idx in idxs:
            if cls._kp(keypoints, idx, min_conf) is not None:
                return True
        return False

    @classmethod
    def _is_real_person(cls, box, keypoints, frame_h: int) -> bool:
        """真人 = bbox 够大 + 看到任一肩/肘/髋/膝关键点（低 conf 也行）。
        机器人自己伸进 FOV 的手只有腕，没肩没肘没髋没膝 → 仍过滤。
        近距离贴脸/拿东西挡腰也能进入候选。
        """
        x1, y1, x2, y2 = (float(box.xyxy[0][i]) for i in range(4))
        if (x2 - x1) * (y2 - y1) < MIN_PERSON_AREA:
            return False
        if keypoints is None:
            return True
        return cls._has_kp(
            keypoints,
            KP_L_SHOULDER, KP_R_SHOULDER,
            KP_L_ELBOW, KP_R_ELBOW,
            KP_L_HIP, KP_R_HIP,
            KP_L_KNEE, KP_R_KNEE,
            min_conf=KP_CONF_PERSON,
        )

    @classmethod
    def _is_facing_camera(cls, keypoints) -> bool:
        """脸部关键点（鼻/眼）至少有一个 → 朝向相机。
        背对时这些都看不到，YOLO 顶多估出两个耳朵；纯背影不做手势检测。
        """
        if keypoints is None:
            return False
        return cls._has_kp(keypoints, KP_NOSE, KP_L_EYE, KP_R_EYE,
                           min_conf=KP_CONF_PERSON)

    @staticmethod
    def _sample_depth(depth_arr, x: float, y: float, w: int, h: int) -> float:
        """3×3 邻域取最小有效深度（米），无效返回 0。numpy 切片，比 get_distance 快得多。"""
        xi, yi = int(x), int(y)
        if xi < 1 or yi < 1 or xi >= w - 1 or yi >= h - 1:
            return 0.0
        patch = depth_arr[yi - 1 : yi + 2, xi - 1 : xi + 2]
        valid = patch[patch > 100]  # uint16 mm, > 0.1m
        if valid.size == 0:
            return 0.0
        return float(valid.min()) * 0.001

    @classmethod
    def _wrist_state(cls, wrist, body_depth: float,
                     depth_arr, w: int, h: int) -> tuple[bool, str]:
        """纯深度判：手腕比身体近相机 ≥ WRIST_FORWARD_M → 伸手。
        不需要大臂可见 —— 近距离 YOLO 经常拿不到肩部关键点。
        """
        if wrist is None:
            return False, "no-W"
        wx, wy, _ = wrist
        d_w = cls._sample_depth(depth_arr, wx, wy, w, h)
        if d_w <= 0:
            return False, "no-d"
        fwd = body_depth - d_w  # 正值 = 手腕比身体更近相机
        if fwd < WRIST_FORWARD_M:
            return False, f"{fwd:+.2f}m"
        return True, f"OK {fwd:+.2f}m"

    @classmethod
    def _detect_gesture_full(cls, keypoints, frame_h: int, body_depth: float,
                             depth_arr, w: int, h: int) -> tuple[str | None, str]:
        """返回 (gesture, debug_label)。判据：左/右手腕是否明显比身体靠前。"""
        lw = cls._kp(keypoints, KP_L_WRIST, ARM_CONF)
        rw = cls._kp(keypoints, KP_R_WRIST, ARM_CONF)

        left_ok, l_lbl = cls._wrist_state(lw, body_depth, depth_arr, w, h)
        right_ok, r_lbl = cls._wrist_state(rw, body_depth, depth_arr, w, h)

        if left_ok and right_ok:
            g = "both"
        elif left_ok:
            g = "left"
        elif right_ok:
            g = "right"
        else:
            g = None
        return g, f"body:{body_depth:.2f}m | L:{l_lbl} | R:{r_lbl}"

    @staticmethod
    def _depth_median(depth_arr, box, w: int, h: int, sample_frac: float = 0.15):
        """取 bbox 中心一小块的有效深度中位数，单点容易踩到 0 或穿透。"""
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
        arr = depth_arr[py1:py2, px1:px2].astype(np.float32) * 0.001
        v = arr[(arr > 0.2) & (arr < 8.0)]
        if v.size < 5:
            return None
        return float(np.median(v))

    @staticmethod
    def _check_obstacles(depth_arr, w: int, h: int):
        # 只把 ROI 这一条带从 uint16 转成 float32 / 1000，省一大半时间
        roi = depth_arr[int(h * ROI_TOP) : int(h * ROI_BOTTOM), :].astype(np.float32) * 0.001

        def min_valid(z):
            v = z[(z > 0.1) & (z < 5.0)]
            return float(np.percentile(v, 5)) if v.size else 5.0

        return (
            min_valid(roi[:, : w // 3]),
            min_valid(roi[:, w // 3 : 2 * w // 3]),
            min_valid(roi[:, 2 * w // 3 :]),
        )


    # ── 四段流水线 ───────────────────────────────────────────────────────────
    # capture 线程：RealSense 抓帧 + align + 复制成 numpy
    # infer  线程：YOLO-pose 推理（GPU）
    # 主线程（decide）：候选筛选、目标决策、bridge 下发（控制环跑得快）
    # draw  线程：plot() + HUD + stream.update + 快照（视觉慢一点没关系）
    # 各 queue maxsize=1 + 丢旧策略：低延迟，控制环和视觉解耦。

    def _put_latest(self, q: queue.Queue, item) -> None:
        try:
            q.get_nowait()  # 丢掉上一帧
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    def _capture_loop(self, q_out: queue.Queue) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue
            aligned = self._align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue
            # 必须 copy —— rs 内部缓冲下一帧会覆盖
            frame = np.asanyarray(color.get_data()).copy()
            depth_arr = np.asanyarray(depth.get_data()).copy()  # uint16 mm
            self._put_latest(q_out, (frame, depth_arr))

    def _infer_loop(self, q_in: queue.Queue, q_out: queue.Queue) -> None:
        while not self._stop.is_set():
            try:
                frame, depth_arr = q_in.get(timeout=0.5)
            except queue.Empty:
                continue
            pose_res = self._model(frame, conf=0.5, verbose=False)[0]
            self._put_latest(q_out, (frame, depth_arr, pose_res))

    def _draw_loop(self, q_in: queue.Queue) -> None:
        while not self._stop.is_set():
            try:
                payload = q_in.get(timeout=0.5)
            except queue.Empty:
                continue
            self._do_draw(payload)

    def run(self) -> None:
        """四线程流水线：capture | infer | decide(主) | draw。Ctrl-C 退。"""
        self.start()
        assert self._model is not None and self._pipeline is not None

        q_cap = queue.Queue(maxsize=1)
        q_dec = queue.Queue(maxsize=1)  # infer → decide
        q_drw = queue.Queue(maxsize=1)  # decide → draw
        cap_t = threading.Thread(target=self._capture_loop, args=(q_cap,), daemon=True)
        inf_t = threading.Thread(target=self._infer_loop, args=(q_cap, q_dec), daemon=True)
        drw_t = threading.Thread(target=self._draw_loop, args=(q_drw,), daemon=True)
        cap_t.start()
        inf_t.start()
        drw_t.start()
        print("[VISION] 四段流水线启动：capture | infer | decide | draw")

        try:
            while not self._stop.is_set():
                try:
                    frame, depth_arr, pose_res = q_dec.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._decide_and_dispatch(frame, depth_arr, pose_res, q_drw)
        except KeyboardInterrupt:
            print("\n[VISION] Ctrl-C 退出")
        finally:
            self._stop.set()
            cap_t.join(timeout=2)
            inf_t.join(timeout=2)
            drw_t.join(timeout=2)
            self.close()

    def _decide_and_dispatch(self, frame, depth_arr, pose_res, q_drw: queue.Queue) -> None:
        """快速决策：障碍 + 目标 + 手势 + bridge 下发；画图丢给 draw 线程。"""
        h, w = frame.shape[:2]
        dist_l, dist_c, dist_r = self._check_obstacles(depth_arr, w, h)
        blocked = dist_c < OBSTACLE_DIST

        gesture = None
        follow_dist = None
        follow_err = None
        boxes = pose_res.boxes
        kpts_all = pose_res.keypoints

        box_marks = []  # (kind, x1, y1, x2, y2, label)，draw 线程根据 kind 上色
        candidates = []  # (d, cx, box_xy, kpts)

        for i, box in enumerate(boxes):
            if int(box.cls) != 0:
                continue
            kpts = (
                kpts_all[i] if (kpts_all is not None and i < len(kpts_all)) else None
            )
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

            if not self._is_real_person(box, kpts, h):
                box_marks.append(("robot_arm", x1, y1, x2, y2, "robot arm"))
                continue

            d = self._depth_median(depth_arr, box, w, h)
            cx = float(box.xywh[0][0])
            if d is None:
                box_marks.append(("no_depth", x1, y1, x2, y2, "no-depth"))
                continue
            if d > self.max_dist:
                box_marks.append(("far", x1, y1, x2, y2, f"far {d:.1f}m"))
                continue
            candidates.append((d, cx, (x1, y1, x2, y2), kpts))

        candidates.sort(key=lambda c: c[0])
        target = candidates[0] if candidates else None

        for idx, (d, _cx, xy, _kpts) in enumerate(candidates):
            x1, y1, x2, y2 = xy
            if idx == 0:
                box_marks.append(("target", x1, y1, x2, y2, f"TARGET {d:.2f}m"))
            else:
                box_marks.append(("other", x1, y1, x2, y2, f"{d:.2f}m"))

        gesture_dbg = ""
        target_xy_for_dbg = None
        if target is not None:
            d, cx, txy, kpts = target
            if self.gesture and kpts is not None:
                gesture, gesture_dbg = self._detect_gesture_full(
                    kpts, h, d, depth_arr, w, h
                )
            if self.follow:
                follow_dist = d
                follow_err = (cx - w / 2) / (w / 2)
            target_xy_for_dbg = (txy[0], txy[1])

        # 控制下发（快）
        if self.follow and follow_dist is not None and follow_dist > 0.1:
            self.bridge.send_move(follow_dist, follow_err, blocked)

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

        # 控制环 FPS（用户在 HUD 上看到的就是这个，反映 bridge 下发速率）
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 1e-3:
                inst = 1.0 / dt
                self._fps = inst if self._fps <= 0 else (self._fps * 0.9 + inst * 0.1)
        self._last_frame_time = now

        # 决定要不要让 draw 线程存快照
        snapshot_now = (
            self.snapshot_dir is not None
            and (now - self._last_snapshot_time) > self.snapshot_every
        )
        if snapshot_now:
            self._last_snapshot_time = now

        # 把所有渲染需要的状态打包扔给 draw 线程（丢旧）
        self._put_latest(
            q_drw,
            {
                "frame": frame,
                "pose_res": pose_res,
                "box_marks": box_marks,
                "target_xy_for_dbg": target_xy_for_dbg,
                "gesture": gesture,
                "gesture_dbg": gesture_dbg,
                "blocked": blocked,
                "dist_l": dist_l,
                "dist_c": dist_c,
                "dist_r": dist_r,
                "follow_dist": follow_dist,
                "fps": self._fps,
                "snapshot_now": snapshot_now,
                "h": h,
                "w": w,
            },
        )

        if self.verbose:
            cd = max(0.0, self.cooldown - (now - self._last_action_time))
            print(
                f"[VISION] L{dist_l:.1f} C{dist_c:.1f} R{dist_r:.1f} "
                f"blk={int(blocked)} follow={follow_dist} ges={gesture} cd={cd:.1f}",
                flush=True,
            )

    # 颜色/粗细/字号 表，给 draw 线程用
    _BOX_STYLE = {
        "robot_arm": ((0, 0, 255), 2, 0.5, 1),
        "no_depth":  ((160, 160, 160), 1, 0.5, 1),
        "far":       ((200, 80, 200), 1, 0.5, 1),
        "other":     ((160, 160, 160), 1, 0.5, 1),
        "target":    ((0, 255, 0), 3, 0.7, 2),
    }

    def _do_draw(self, p: dict) -> None:
        """画图 + 推流 + 快照。在 draw 线程里跑，慢了不影响控制环。"""
        annotated = p["pose_res"].plot(boxes=False)
        h, w = p["h"], p["w"]

        for kind, x1, y1, x2, y2, label in p["box_marks"]:
            col, thk, fs, ft = self._BOX_STYLE.get(kind, ((255, 255, 255), 1, 0.5, 1))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, thk)
            cv2.putText(annotated, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col, ft)

        if p["target_xy_for_dbg"] and p["gesture_dbg"]:
            tx1, ty1 = p["target_xy_for_dbg"]
            cv2.putText(annotated, p["gesture_dbg"], (tx1, max(0, ty1 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        cv2.putText(annotated, f"FPS {p['fps']:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        bar = (0, 0, 255) if p["blocked"] else (0, 255, 0)
        cv2.putText(annotated,
                    f"L:{p['dist_l']:.1f} C:{p['dist_c']:.1f} R:{p['dist_r']:.1f}m",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar, 2)
        if p["follow_dist"]:
            cv2.putText(annotated, f"follow={p['follow_dist']:.2f}m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if p["gesture"]:
            cv2.putText(annotated, f"ARM: {p['gesture'].upper()}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if self.stream is not None:
            self.stream.update(annotated)

        if p["snapshot_now"] and self.snapshot_dir is not None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(str(self.snapshot_dir / f"{ts}_raw.jpg"), p["frame"])
            cv2.imwrite(str(self.snapshot_dir / f"{ts}_ann.jpg"), annotated)
