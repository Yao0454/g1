"""
movement.py — 手臂手势检测模块
用法: python movement.py

检测逻辑:
  - 手腕关键点出现在画面中 → 判定手臂伸出
  - 发 UDP (uint8 action_id) → localhost:9871 → C++ arm binary

调试流: http://192.168.123.164:6768/
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

STREAM_PORT        = 6768
ACTION_COOLDOWN    = 4.0
CONF_THRESHOLD     = 0.5
ARM_CONF_THRESHOLD = 0.6

ARM_HOST = "127.0.0.1"
ARM_PORT = 9871

KP_L_WRIST, KP_R_WRIST = 9, 10

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def do_action(action_id: int, name: str):
    _sock.sendto(struct.pack("B", action_id), (ARM_HOST, ARM_PORT))
    print(f"[ACTION] {name} (id={action_id})")

def kp(keypoints, idx):
    try:
        data = keypoints.data
        pt = data[0][idx] if data.ndim == 3 else data[idx]
        x, y, c = float(pt[0]), float(pt[1]), float(pt[2])
        return (x, y, c) if c >= CONF_THRESHOLD else None
    except (IndexError, TypeError, AttributeError):
        return None

def detect_arm_extend(keypoints):
    lw = kp(keypoints, KP_L_WRIST)
    rw = kp(keypoints, KP_R_WRIST)
    left  = lw is not None and lw[2] >= ARM_CONF_THRESHOLD
    right = rw is not None and rw[2] >= ARM_CONF_THRESHOLD
    if left and right: return "both"
    if left:           return "left"
    if right:          return "right"
    return None

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

def main():
    global latest_frame

    pose_model_path = "yolov8n-pose.engine" if os.path.exists("yolov8n-pose.engine") \
                      else "yolov8n-pose.pt"
    print(f"[MODEL] 加载 {pose_model_path} ...")
    model = YOLO(pose_model_path, task="pose")
    print("[MODEL] 热身中...")
    model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)
    print("[MODEL] 热身完成")

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    print("[CAM] RealSense 已启动")

    threading.Thread(
        target=lambda: HTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler).serve_forever(),
        daemon=True).start()
    print(f"[STREAM] http://192.168.123.164:{STREAM_PORT}/")

    last_action_time = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame    = np.asanyarray(color_frame.get_data())
            results  = model(frame, verbose=False)
            annotated = results[0].plot()

            gesture = None
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                gesture = detect_arm_extend(results[0].keypoints)

            now = time.time()
            if gesture and (now - last_action_time) > ACTION_COOLDOWN:
                last_action_time = now
                action_map = {"left": (26, "挥手"), "right": (27, "握手"), "both": (15, "双手举起")}
                action_id, label = action_map[gesture]
                print(f"[GESTURE] {gesture} → {label}")
                do_action(action_id, label)

            color  = (0, 255, 0) if gesture else (100, 100, 100)
            status = f"ARM: {gesture.upper()}" if gesture else "waiting..."
            cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cd = max(0.0, ACTION_COOLDOWN - (now - last_action_time))
            cv2.putText(annotated, f"cooldown: {cd:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            with frame_lock:
                latest_frame = annotated.copy()

    finally:
        pipeline.stop()
        print("[CAM] 停止")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
