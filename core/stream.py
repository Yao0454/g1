"""
core/stream.py — 一个共享的 MJPEG HTTP 流，多个组件可往里塞最新帧。

用法：
    stream = MjpegStream(port=6769)
    stream.start()
    stream.update(frame_bgr)        # 在主循环里随便塞
    # 浏览器打开 http://<ip>:6769/
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer as _HTTPServer

class HTTPServer(ThreadingMixIn, _HTTPServer):
    """每个连接独立线程，允许多个客户端同时接流（浏览器 + Flutter 同时连）。"""
    daemon_threads = True

import cv2


class MjpegStream:
    def __init__(self, port: int = 6769, jpeg_quality: int = 70):
        self.port = port
        self.jpeg_quality = jpeg_quality
        self._frame = None
        self._frame_id = 0
        self._cond = threading.Condition()
        self._server: HTTPServer | None = None

    def update(self, frame) -> None:
        # plot() 每帧给新 array，不必 copy；用 Condition 唤醒 HTTP 消费者
        with self._cond:
            self._frame = frame
            self._frame_id += 1
            self._cond.notify_all()

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if self.path != "/":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    last_id = -1
                    while True:
                        with owner._cond:
                            # 没新帧就阻塞，不烧 CPU
                            while owner._frame_id == last_id:
                                owner._cond.wait(timeout=1.0)
                            f = owner._frame
                            last_id = owner._frame_id
                        if f is None:
                            continue
                        ok, jpg = cv2.imencode(
                            ".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, owner.jpeg_quality]
                        )
                        if not ok:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + jpg.tobytes()
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = HTTPServer(("0.0.0.0", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"[STREAM] http://0.0.0.0:{self.port}/")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
