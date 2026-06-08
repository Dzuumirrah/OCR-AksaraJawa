from ast import parse
import ipaddress
import time
import numpy as np
from threading import Lock
import cv2

from urllib.parse import urlparse
from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot
from numpy.strings import startswith
from tensorflow.python.framework.test_ops import none

from src.params import camera 

class IPWebCamThread(QThread):
    """
    Background thread untuk capture video dari IP Webcam APP (android).
    - Frame dropping: hanya emit frame terbaru, skip frame lama
    - Non-blocking buffer drain: drain buffer OpenCV sebelum ambil frame
    - Support RTSP sebagai fallback untuk latency lebih rendah
    - Adaptive sleep berdasarkan target FPS
    IP webcam dapat diakses dari URL seperti http://<IP_ADDRESS>:8080/video
    """
    frame_ready = pyqtSignal(object)
    stream_connected = pyqtSignal(str)

    def __init__(self, ip_address=camera.IP_CAMERA_URL, port=camera.IP_CAMERA_PORT,
                 target_fps = camera.TARGET_FPS, prefer_rtsp=camera.USE_RTSP):
        super().__init__()
        self.ip_address = ip_address
        self.port = port
        self.target_fps = target_fps
        self.prefer_rtsp = prefer_rtsp
        self.running = False
        self.cap = None

        self._frame_lock = Lock()
        self._latest_frame = None

    @staticmethod
    def _build_mjpeg_url(ip_address: str, port: int):
        base = str(ip_address).strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            base = f"http://{base}"
        if base.endswith("/video"):
            return base
        parsed = urlparse(base)
        if parsed.port:
            return f"{base}/video"
        return f"{base}:{port}/video"
    
    @staticmethod
    def _build_rtsp_url(ip_address: str, port: int):
        base = str(ip_address).strip().rstrip("/")
        for scheme in ("http://", "https://"):
            if base.startswith(scheme):
                base = base[len(scheme):]

        host = base.split(":")[0]
        return f"rtsp://{host}:{port}/h264_ulaw.sdp"

    def _open_capture(self):
        """Utamakan RTSP, fallback ke MJPEG"""
        if self.prefer_rtsp:
            rtsp_url = self._build_rtsp_url(self.ip_address, self.port)
            print(f"[IPWebcam]: Mencoba RTSP: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            # tuning
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # timeout koneksi
            time.sleep(0.5)

            # cek koneksi kamera 
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print("[IPWebcam]: RTSP sucsess")
                    return cap, "rtsp"
                cap.release()
        # fallback jika RTSP gagal (MJPEG)
        mjpeg_url = self._build_mjpeg_url(self.ip_address, self.port)
        print(f"[IPWebcam]: Using MJPEG: {mjpeg_url}")
        cap = cv2.VideoCapture(mjpeg_url)
        
        # tuning
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, "mjpeg"
        
    def _drain_buffer(self):
        """
        Menguras buffer OpenCV untuk memastikan frame tidak menunmpu
        """
        if self.cap is None:
            return
        for _ in range(1):
            if not self.cap.grab():
                break

    def run(self):
        """Start streaming dari IP webcam."""
        try: 
            self.cap, stream_type = self._open_capture()
            
            if not self.cap.isOpened():
                print("[IPWebcam]: Cannot open camera stream")
                return
            
            self.running = True
            self.stream_connected.emit(stream_type)
            print(f"[IPWebcam]: Camera thread started. Stream on ({stream_type.upper()}) | target FPS {self.target_fps}")

            frame_interval = 1.0/self.target_fps
            frame_count = 0
            last_fps_time = time.time()

            while self.running:
                t_start = time.perf_counter()

                # drain buffer
                self._drain_buffer()

                ret, frame = self.cap.read()

                if not ret:
                    print("[IPWebcam]: Error reading frame, reconnecting...")
                    time.sleep(0.5)
                    self.cap.release()
                    self.cap, stream_type = self._open_capture()
                    continue
                
                self.frame_ready.emit(frame)
            
                frame_count += 1
                # Log FPS per 5 detik
                now = time.time()

                if now - last_fps_time > 5.0:
                    elapsed = now - last_fps_time
                    actual_fps = frame_count / elapsed
                    print(f"[IPWebcam]: Actual FPS ({actual_fps:.1f})")
                    frame_count = 0
                    last_fps_time = now
                
                # Adaptive sleep
                elapsed = time.perf_counter() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
        except Exception as e:
            print(f"[IPWebcam]: Error: {e}")
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            print("[IPWebcam]: Camera thread stopped")

    def stop(self):
        """Stop video capture."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.wait(2000)
                
