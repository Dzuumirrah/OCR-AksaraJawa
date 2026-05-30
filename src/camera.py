
import src.params as params
import cv2

from urllib.parse import urlparse
from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot
import time

class IPWebCamThread(QThread):
    """
    Background thread untuk capture video dari IP Webcam APP (android).
    
    IP webcam dapat diakses dari URL seperti http://<IP_ADDRESS>:8080/video
    """
    frame_ready = pyqtSignal(object)

    def __init__(self, ip_address=params.IP_CAMERA_URL, port=params.IP_CAMERA_PORT):
        super().__init__()
        self.ip_address = ip_address
        self.port = port
        self.url = self._build_stream_url(ip_address, port)
        self.running = False
        self.cap = None

    @staticmethod
    def _build_stream_url(ip_address, port):
        base_url = str(ip_address).strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        if base_url.endswith("/video"):
            return base_url
        parsed = urlparse(base_url)
        if parsed.port is not None:
            return f"{base_url}/video"
        return f"{base_url}:{port}/video"

    def run(self):
        """Start streaming dari IP webcam."""
        try: 
            print(f"[IPWebcam]: Connecting to IP webcam at == {self.url} ==...")
            
            # Coba akses URL video stream untuk memastikan koneksi
            self.cap = cv2.VideoCapture(self.url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                print("[IPWebcam]: Cannot open camera")
                return
            
            self.running = True
            print("[IPWebcam]: Camera thread started")

            frame_count = 0
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    # frame = cv2.flip(frame, 1)  # Mirror image
                    self.frame_ready.emit(frame)
                    time.sleep(0.001)  # ~100 FPS

                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[IPWebcam]: Received {frame_count} frames")
                else:
                    print("[IPWebcam]: Stream disconnected")
                    break
                
        except Exception as e:
            print(f"[IPWebcam]: Error in camera thread: {e}")
        finally:
            self.running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            print("[IPWebcam]: Camera thread stopped")

    def stop(self):
        """Stop video capture."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.wait(1500)
                
