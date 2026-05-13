"""
follow.py — 人体跟随 + 手势识别 合并版
用法: python follow.py

功能:
  - 检测最近的人 → UDP → move binary (port 9870) 跟随
  - 检测手腕手势 → UDP → arm binary  (port 9871) 执行动作
  - 过滤机器人自身手臂（黑色手 + 银色臂，bbox 小且无下肢关键点）

调试流: http://192.168.123.164:6769/
不修改 main.py / movement.py / run.sh
"""

import os
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ── 网络 ──────────────────────────────────────────────────────────────────────
MOVE_HOST  = "127.0.0.1"
MOVE_PORT  = 9870
ARM_HOST   = "127.0.0.1"
ARM_PORT   = 9871
STREAM_PORT = 6769

# ── 跟随参数 ──────────────────────────────────────────────────────────────────
TARGET_DIST   = 1.0    # 目标保持距离 (m)
OBSTACLE_DIST = 0.6
ROI_TOP       = 0.35
ROI_BOTTOM    = 0.75

# ── 手势参数 ──────────────────────────────────────────────────────────────────
ACTION_COOLDOWN    = 4.0
CONF_THRESHOLD     = 0.5
ARM_CONF_THRESHOLD = 0.6
KP_L_WRIST, KP_R_WRIST   = 9, 10
KP_L_KNEE,  KP_R_KNEE    = 13, 14
KP_L_ANKLE, KP_R_ANKLE   = 15, 16

# ── 机器人自身手臂过滤 ────────────────────────────────────────────────────────
# 真实人体检测框面积至少要有这么大（像素²），机器人手臂 bbox 远小于此
MIN_PERSON_AREA = 8000
# 真实人体需要至少一个下肢关键点（膝/踝）被检测到
# 机器人手臂不会有这些关键点

# ── UDP socket ────────────────────────────────────────────────────────────────
_move_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_arm_sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_move(dist, err_norm, blocked):
    data = struct.pack("fff", dist, err_norm, 1.0 if blocked else 0.0)
    _move_sock.sendto(data, (MOVE_HOST, MOVE_PORT))

def send_arm(action_id: int, name: str):
    _arm_sock.sendto(struct.pack("B", action_id), (ARM_HOST, ARM_PORT))
    print(f"[ARM] {name} (id={action_id})")

# ── 关键点工具 ────────────────────────────────────────────────────────────────
def get_kp(keypoints, idx):
    try:
        data = keypoints.data
        pt = data[0][idx] if data.ndim == 3 else data[idx]
        x, y, c = float(pt[0]), float(pt[1]), float(pt[2])
        return (x, y, c) if c >= CONF_THRESHOLD else None
    except (IndexError, TypeError, AttributeError):
        return None

def has_lower_body(keypoints):
    """至少有一个膝盖或脚踝被检测到 → 是真实人体而非机器人手臂"""
    for idx in (KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE):
        if get_kp(keypoints, idx) is not None:
            return True
    return False

def is_real_person(box, keypoints):
    """
    过滤机器人自身手臂：
    1. bbox 面积必须足够大
    2. 必须有下肢关键点
    """
    x1, y1, x2, y2 = float(box.xyxy[0][0]), float(box.xyxy[0][1]), \
                      float(box.xyxy[0][2]), float(box.xyxy[0][3])
    area = (x2 - x1) * (y2 - y1)
    if area < MIN_PERSON_AREA:
        return False
    if keypoints is not None and not has_lower_body(keypoints):
        return False
    return True

# ── 手势检测 ──────────────────────────────────────────────────────────────────
def detect_arm_extend(keypoints):
    lw = get_kp(keypoints, KP_L_WRIST)
    rw = get_kp(keypoints, KP_R_WRIST)
    left  = lw is not None and lw[2] >= ARM_CONF_THRESHOLD
    right = rw is not None and rw[2] >= ARM_CONF_THRESHOLD
    if left and right: return "both"
    if left:           return "left"
    if right:          return "right"
    return None

# ── 障碍检测 ──────────────────────────────────────────────────────────────────
def check_obstacles(depth_frame, w, h):
    depth = np.asanyarray(depth_frame.get_data()).astype(float) / 1000.0
    roi   = depth[int(h * ROI_TOP):int(h * ROI_BOTTOM), :]
    def min_valid(z):
        v = z[(z > 0.1) & (z < 5.0)]
        return float(np.percentile(v, 5)) if len(v) > 0 else 5.0
    return (min_valid(roi[:, :w//3]),
            min_valid(roi[:, w//3:2*w//3]),
            min_valid(roi[:, 2*w//3:]))

# ── MJPEG 流 ──────────────────────────────────────────────────────────────────
latest_frame = None
frame_lock   = threading.Lock()

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def do_GET(self):
        if self.path != "/":
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    frame = latest_frame
                if frame is None:
                    continue
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpg.tobytes() + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

# ── 主循环 ────────────────────────────────────────────────────────────────────
def main():
    global latest_frame

    # 检测模型（跟随用）
    detect_path = "yolov8n.engine" if os.path.exists("yolov8n.engine") else "yolov8n.pt"
    print(f"[MODEL] 加载检测模型 {detect_path} ...")
    model_det = YOLO(detect_path, task="detect")

    # 姿态模型（手势用）
    pose_path = "yolov8n-pose.engine" if os.path.exists("yolov8n-pose.engine") \
                else "yolov8n-pose.pt"
    print(f"[MODEL] 加载姿态模型 {pose_path} ...")
    model_pose = YOLO(pose_path, task="pose")

    print("[MODEL] 热身中...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model_det(dummy, verbose=False)
    model_pose(dummy, verbose=False)
    print("[MODEL] 热身完成")

    # 相机（颜色 + 深度）
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    print("[CAM] RealSense 已启动")

    threading.Thread(
        target=lambda: HTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler).serve_forever(),
        daemon=True).start()
    print(f"[STREAM] http://192.168.123.164:{STREAM_PORT}/")

    last_arm_time = 0.0
    ARM_ACTIONS   = {"left": (26, "挥手"), "right": (27, "握手"), "both": (15, "双手举起")}

    try:
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            h, w  = frame.shape[:2]

            # ── 障碍检测 ──────────────────────────────────────────────────
            dist_l, dist_c, dist_r = check_obstacles(depth_frame, w, h)
            blocked = dist_c < OBSTACLE_DIST

            # ── 姿态推理（手势 + 过滤机器人手臂） ─────────────────────────
            pose_res = model_pose(frame, verbose=False)[0]
            annotated = pose_res.plot()

            gesture     = None
            follow_dist = None
            follow_err  = None

            boxes    = pose_res.boxes
            kpts_all = pose_res.keypoints

            for i, box in enumerate(boxes):
                if int(box.cls) != 0:
                    continue
                kpts = kpts_all[i] if kpts_all is not None and i < len(kpts_all) else None

                # 过滤机器人自身手臂
                if not is_real_person(box, kpts):
                    # 在画面上标红提示被过滤
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated, "robot arm", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    continue

                # 手势检测（只对最近的真实人体做）
                if kpts is not None and gesture is None:
                    gesture = detect_arm_extend(kpts)

                # 跟随目标：取置信度最高的真实人体
                if follow_dist is None:
                    cx = float(box.xywh[0][0])
                    cy = float(box.xywh[0][1])
                    d  = depth_frame.get_distance(int(cx), int(cy))
                    if d > 0:
                        follow_dist = d
                        follow_err  = (cx - w / 2) / (w / 2)

            # ── 发送跟随指令 ───────────────────────────────────────────────
            if follow_dist is not None and follow_dist > 0.1:
                send_move(follow_dist, follow_err, blocked)

            # ── 发送手臂指令 ───────────────────────────────────────────────
            now = time.time()
            if gesture and (now - last_arm_time) > ACTION_COOLDOWN:
                last_arm_time = now
                action_id, label = ARM_ACTIONS[gesture]
                print(f"[GESTURE] {gesture} → {label}")
                send_arm(action_id, label)

            # ── HUD ────────────────────────────────────────────────────────
            bar_color = (0, 0, 255) if blocked else (0, 255, 0)
            cv2.putText(annotated, f"L:{dist_l:.1f} C:{dist_c:.1f} R:{dist_r:.1f}m",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
            if follow_dist:
                cv2.putText(annotated, f"follow dist={follow_dist:.2f}m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if gesture:
                cv2.putText(annotated, f"ARM: {gesture.upper()}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            with frame_lock:
                latest_frame = annotated.copy()

    finally:
        pipeline.stop()
        print("[CAM] 停止")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
