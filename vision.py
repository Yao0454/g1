import os
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

MOVE_HOST = "127.0.0.1"
MOVE_PORT = 9870

detect_engine = "yolov8n.engine"
if not os.path.exists(detect_engine):
    YOLO("yolov8n.pt").export(format="engine", half=True)
model_detect = YOLO(detect_engine, task="detect")

# RealSense D435i
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(cfg)
align = rs.align(rs.stream.color)

# UDP socket → move
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

latest_frame = None
frame_lock = threading.Lock()

OBSTACLE_DIST = 0.6
ROI_TOP = 0.35
ROI_BOTTOM = 0.75


def send_to_move(dist, err_norm, blocked):
    data = struct.pack("fff", dist, err_norm, 1.0 if blocked else 0.0)
    sock.sendto(data, (MOVE_HOST, MOVE_PORT))


def check_obstacles(depth_frame, w, h):
    depth = np.asanyarray(depth_frame.get_data()).astype(float) / 1000.0
    roi = depth[int(h * ROI_TOP) : int(h * ROI_BOTTOM), :]
    left = roi[:, : w // 3]
    center = roi[:, w // 3 : 2 * w // 3]
    right = roi[:, 2 * w // 3 :]

    def min_valid(zone):
        valid = zone[(zone > 0.1) & (zone < 5.0)]
        return float(np.percentile(valid, 5)) if len(valid) > 0 else 5.0

    return min_valid(left), min_valid(center), min_valid(right)


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
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
                    + jpg.tobytes()
                    + b"\r\n"
                )
        except (BrokenPipeError, ConnectionResetError):
            pass


threading.Thread(
    target=lambda: HTTPServer(("0.0.0.0", 6767), StreamHandler).serve_forever(),
    daemon=True,
).start()
print("Stream: http://101.37.80.57:6767/")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        h, w = frame.shape[:2]

        dist_l, dist_c, dist_r = check_obstacles(depth_frame, w, h)
        blocked = dist_c < OBSTACLE_DIST

        bar_color = (0, 0, 255) if blocked else (0, 255, 0)
        cv2.putText(
            frame,
            f"L:{dist_l:.1f} C:{dist_c:.1f} R:{dist_r:.1f}m",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            bar_color,
            2,
        )
        if blocked:
            cv2.putText(
                frame,
                "OBSTACLE!",
                (w // 2 - 80, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        det_results = model_detect(frame, verbose=False)
        annotated = det_results[0].plot()
        boxes: Boxes = det_results[0].boxes

        persons = [box for box in boxes if int(box.cls) == 0]
        if persons:

            def get_dist(box):
                cx, cy = int(float(box.xywh[0][0])), int(float(box.xywh[0][1]))
                d = depth_frame.get_distance(cx, cy)
                return d if d > 0 else float("inf")

            nearest = min(persons, key=get_dist)
            x_center = float(nearest.xywh[0][0])
            y_center = float(nearest.xywh[0][1])
            error = x_center - w / 2
            dist = depth_frame.get_distance(int(x_center), int(y_center))

            send_to_move(dist, error / (w / 2), blocked)

            cv2.putText(
                annotated,
                f"Dist:{dist:.2f}m Err:{error:.0f}px",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        with frame_lock:
            latest_frame = annotated.copy()

finally:
    pipeline.stop()
    sock.close()
