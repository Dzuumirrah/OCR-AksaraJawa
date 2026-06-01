from pyclbr import Class
from sys import exception
from typing import Self

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QInputDialog,
    QGroupBox,
    QGridLayout,
    QPushButton,
    QLabel
)
from PyQt5.QtGui import(
    QPainter,
    QColor,
    QImage,
    QPixmap,
    QPen,
    QFont
)
from PyQt5.QtCore import (
    QTimer, 
    pyqtSlot,
    Qt,
    pyqtSignal,
    QMutex,
    QMutexLocker,
    QThread
)
import cv2
from threading import Lock as ThreadLock

from tensorflow.python.framework.test_util import lock

from ocr import OCRPipelineAksara
from src.camera import IPWebCamThread

from src.params import directory, gui, camera, model_conf, ocr_conf
COLOR = gui.COLORS

class OCRWorkerThread(QThread):
    """Worker thread agar OCR inference tidak membekukan GUI."""
    result_ready = pyqtSignal(list, tuple)  # hasil, frame_shape

    def __init__(self, ocr_pipeline: OCRPipelineAksara) -> None:
        super().__init__()
        
        self.ocr = ocr_pipeline
        self._pending_frame = None
        self._lock = ThreadLock()
        self.running = True

    def submit_frame(self, frame: cv2.typing.MatLike):
        with self._lock:
            self._pending_frame = frame.copy()

    def run(self):
        while self.running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is not None:
                try:
                    biner = self.ocr.binarisasi(frame)
                    boxes = self.ocr.segmentasi_karakter(biner)
                    hasil = self.ocr.klasifikasi_batch(frame, boxes)
                    self.result_ready.emit(hasil, frame.shape)
                except Exception as e:
                    print(f"[OCRWorker]: Error {e}")
            else:
                self.msleep(50)

    def stop(self):
        self.running = False
        self.wait(2000)

class  CameraWidget(QWidget):
    MARGIN = 40                         # Margin 
    RENDER_FPS = gui.RENDER_FPS         # maksimum render FPS

    def __init__(self):
        super().__init__()
        
        # GUI setup
        self.setMinimumSize(gui.W_CANVAS, gui.H_CANVAS)
        self.setStyleSheet(f"background-color:{COLOR['DARKER_BLUE']}")
        self.setFocusPolicy(Qt.StrongFocus)

        # Camera declaration
        self.show_camera = False
        self.camera_frame = None
        self.camera_thread = None
        self.camera_enabled = False

        # Camera rendering
        self._pixmap_lock = QMutex()
        self._cached_pixmap = None
        self._latest_frame = None

        # timer render
        self._render_timer = QTimer()
        self._render_timer.setInterval(1000 // self.RENDER_FPS)
        self._render_timer.timeout.connect(self.update)
        self._render_timer.start()
        self._new_frame_available = False

        # Hasil OCR
        self._ocr_results = []
        self._ocr_frame_shape = None

    @pyqtSlot(object)
    def _on_camera_frame(self, frame: cv2.typing.MatLike):
        """Receive frame dari camera thread lalu render dari frame ke pixmap"""
        if not self.show_camera:
            return
        
        # Flag kalau ada frame baru
        self._new_frame_available = True
        
        h, w, ch = frame.shape
        widget_w, widget_h = self.width(), self.height()

        # Resize frame sesuai ukuran widget
        scale = max(widget_w / w, widget_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize jika ukuran berbeda jauh
        if abs(new_h - h) > 2 or abs(new_w - w) > 2:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_frame.shape

        # Buat QImage
        qt_image = QImage(rgb_frame.data, w, h, w*3, QImage.Format_RGB888).copy()
        new_pixmap = QPixmap.fromImage(qt_image)

        # Simpan cache dengan lock
        with QMutexLocker(self._pixmap_lock):
            self._cached_pixmap = new_pixmap
            self._latest_frame = frame

        # # Triger repaint sesuai FPS dari timer
        # self.update()
        

    def paintEvent(self, event):
        """Render canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if not self.show_camera:
            painter.fillRect(self.rect(), QColor(COLOR['DARKER_BLUE']))
            return
        
        with QMutexLocker(self._pixmap_lock):
            pixmap = self._cached_pixmap

        if pixmap is not None:
            # center pixmap
            x = (self.width() - pixmap.width()) // 2
            y = (self.height() - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)
            if self._ocr_results and self._ocr_frame_shape:
                self.draw_ocr_boxes(painter, pixmap, x, y)
                
            # self._draw_camera_background(painter)
        else:
            painter.fillRect(self.rect(), QColor(COLOR['DARKER_BLUE']))
        
    def draw_ocr_overlay(self, hasil: list, frame_shape: tuple):
        self._ocr_results = hasil
        self._ocr_frame_shape = frame_shape

    def draw_ocr_boxes(self, painter: QPainter, pixmap: QPixmap, offset_x, offset_y):
        """Gambar bounding box dan label OCR di atas frame yang ditampilkan"""
        if not self._ocr_results or self._ocr_frame_shape is None:
            return
        
        orig_h, orig_w = self._ocr_frame_shape[:2]
        scale_x = pixmap.width() / orig_h
        scale_y = pixmap.height() / orig_w

        for h in self._ocr_results:
            bx, by, bw, bh = h['bbox']
            sx = int(bx * scale_x) + offset_x
            sy = int(by * scale_y) + offset_y
            sw = int(bw * scale_x)
            sh = int(bh * scale_y)

            color = QColor(COLOR['GREEN']) if h['valid'] else QColor(COLOR['RED'])
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(sx, sy, sw, sh)

            label = f"{h['kelas']} {h['confidence']:.1f}"
            painter.setFont(QFont("Arial", 9))
            painter.setPen(QColor(color))
            painter.drawText(sx, sy - 4, label)

    def _draw_camera_background(self, painter):
        """Draw camera frame sebagai background."""
        if self.camera_frame is None:
            return
        
        # Convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame untuk menyesuaikan dengan canvas
        h, w, ch = rgb_frame.shape
        widget_w, widget_h = self.width(), self.height()

        # jaga aspect ratio and cover the canvas
        scale = max(widget_w / w, widget_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))
        h, w, ch = rgb_frame.shape
        
        # Convert ke QImage
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_image)
        
        # center ke canvas
        pixmap_x = (widget_w - pixmap.width()) // 2
        pixmap_y = (widget_h - pixmap.height()) // 2

        painter.drawPixmap(pixmap_x, pixmap_y, pixmap)
   
class ControlPanel(QWidget):
    """Control panel dengan buttons dan status displat"""
    # tombol untuk aktivasi kamera
    camera_toggle = pyqtSignal(bool)  # True = aktifkan kamera, False = matikan kamera
    # tombol untuk ubah IP kamera
    change_ip_pressed = pyqtSignal()  # Tekan untuk ubah IP kamera

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background-color: {COLOR['DARKER_BLUE']}; color: white")
        self._build_layout()

        # State tracking
        self.current_status = "IDLE"
        self.last_target = None
        self.is_picked = False
        self.is_placed = False

    def _build_layout(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        # === STATUS SECTION ===
        status_group = self._build_status_section()
        main_layout.addWidget(status_group)

        # === COMMAND BUTTONS ===
        cmd_group = self._build_command_section()
        main_layout.addWidget(cmd_group)

        main_layout.addStretch()

    def _build_status_section(self):
        """Status indikator + state display"""
        group = QGroupBox("Status")
        group.setStyleSheet(self._groupbox_style())
        layout = QVBoxLayout()

        # Status light (colored circle)
        status_layout = QHBoxLayout()

        status_text_layout = QVBoxLayout()
        self.status_label = QLabel("IDLE")
        self.status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.status_detail = QLabel("Ready for command")
        self.status_detail.setFont(QFont('Arial', 10))
        self.status_detail.setStyleSheet(f"color: {COLOR['GRAY']}")

        status_text_layout.addWidget(self.status_label)
        status_text_layout.addWidget(self.status_detail)

        status_layout.addLayout(status_text_layout)
        status_layout.addStretch()

        layout.addLayout(status_layout)
        group.setLayout(layout)
        self.update()
        return group
    
    def _build_command_section(self):
        """Command buttons."""
        group = QGroupBox("Commands")
        group.setStyleSheet(self._groupbox_style())
        layout = QGridLayout()
        layout.setSpacing(10)

        # camera toggle button
        self.btn_camera = QPushButton("CAMERA")
        self.btn_camera.setCheckable(True)
        self.btn_camera.setFixedHeight(50)
        self.btn_camera.setStyleSheet(self._button_style(COLOR["LIGHT_BLUE"]))
        self.btn_camera.toggled.connect(self._on_camera_toggle)
        layout.addWidget(self.btn_camera, 1, 0, 1, 2)

        # ubah IP adress kamera
        self.btn_change_ip = QPushButton("Change Camera IP")
        self.btn_change_ip.setFixedHeight(40)
        self.btn_change_ip.setStyleSheet(self._button_style(COLOR["LIGHT_BLUE"]))
        self.btn_change_ip.clicked.connect(self._on_change_camera_ip)
        layout.addWidget(self.btn_change_ip, 2, 0, 1, 2)

        group.setLayout(layout)
        self.update()
        return group
    


    # === STYLING ===
    def _groupbox_style(self):
        return f"""
        QGroupBox {{
            border: 2px solid {COLOR['LIGHT_BLUE']};
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
            color: {COLOR['LIGHTER_BLUE']}      
        }}
        QGroupBox::title{{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px
            color: {COLOR['DARK_BLUE']}
        }}
        """
    def _button_style(self, bg_color, emergency=False):
        border_color = COLOR["RED"] if emergency else COLOR['DARKER_BLUE']
        return f"""
        QPushButton {{
            background-color: {bg_color};
            color: white;
            border: 2px solid {border_color};
            border-radius: 5px;
            font-weight: bold;
            font-size: 12px;
            padding: 5px;
        }}
        QPushButton:hover {{
            background-color: {self._lighten_color(bg_color)};
        }}
        QPushButton:pressed {{
            background-color: {self._darken_color(bg_color)};
        }}
        QPushButton:disabled {{
            background-color: {COLOR['DARKER_GRAY']};
            color: {COLOR['GRAY']};
        }}
        """
    
    @staticmethod
    def _lighten_color(hex_color):
        """Lighten color by 20%."""
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, int(c * 1.2)) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    @staticmethod
    def _darken_color(hex_color):
        """Darken color by 30%."""
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(int(c * 0.7) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
        
    # === COMMAND HANDLERS ===
    
    def _on_camera_toggle(self):
        """Handle camera toggle."""
        is_checked = self.btn_camera.isChecked()
        if is_checked:
            self.btn_camera.setText("CAMERA ON")
            self.btn_camera.setStyleSheet(self._button_style(COLOR["GREEN"]))
            self.camera_toggle.emit(True)
        else:
            self.btn_camera.setText("CAMERA OFF")
            self.btn_camera.setStyleSheet(self._button_style(COLOR["LIGHT_BLUE"]))
            self.camera_toggle.emit(False)

    def _on_change_camera_ip(self):
        """Handle change camera IP button click."""
        self.set_status("IDLE", "Changing camera IP...")
        self.change_ip_pressed.emit()
    # === PUBLIC METHODS ===
    
    def set_status(self, status, detail=""):
        """Update status display."""
        self.current_status = status
        self.status_label.setText(status)
        self.status_detail.setText(detail)
        
class MainWindow(QMainWindow):
    def __init__ (self):
        super().__init__()
        self.setWindowTitle("OCR Aksara Jawa dengan IPWebcam camera")
        self.setMinimumSize(gui.W_MAIN, gui.H_MAIN)
        self._build_layout()

        # Auto-idle timer setelah 5 detik
        self.idle_timer = QTimer()
        self.idle_timer.setSingleShot(True)
        self.idle_timer.timeout.connect(self._on_idle_timeout)
        self.IDLE_TIMEOUT_MS = 5000

        # konfigurasi IP camera
        self.ip_camera_config = {
            "ip_adress": camera.IP_CAMERA_URL,
            "port": 8080
        }

        # Thread untuk menampilkan kamera ke canvas
        self.canvas.camera_thread = IPWebCamThread(
            self.ip_camera_config["ip_adress"],
            self.ip_camera_config["port"]
        )

        # Thread untuk menampilkan hasil OCR ke canvas
        try:
            ocr_pipeline = OCRPipelineAksara(
                model_path = directory.MODEL_PATH,
                class_names = model_conf.CLASS_NAMES,
                confidence_threshold= ocr_conf.CONF_THRESH
            )
            
            self._ocr_worker = OCRWorkerThread(ocr_pipeline)
            self._ocr_worker.result_ready.connect(self._on_ocr_result)
            self._ocr_worker.start()
        except Exception as e:
            print(f"[MainWindow] OCR model cannot be loaded {e}")
            self._ocr_worker = None
  

    def _build_layout(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setSpacing(10)
        root_layout.setContentsMargins(10, 10, 10, 10)

        # Kiri: Camera area
        self.canvas = CameraWidget()
        # Kanan: Panel area
        self.panel = ControlPanel()
        self.panel.change_ip_pressed.connect(self._on_change_camera_ip)
        
        root_layout.addWidget(self.canvas, 3)
        root_layout.addWidget(self.panel, 1)

        self.panel.camera_toggle.connect(self._on_camera_toggle)

    @pyqtSlot(bool)
    def _on_camera_toggle(self, enabled):
        """Handle camera toggle from control panel"""
        try:
            if enabled:
                if self.canvas.camera_thread is not None and self.canvas.camera_thread.isRunning():
                    self.canvas.camera_thread.stop()
                self.canvas.camera_thread = IPWebCamThread(
                    self.ip_camera_config["ip_adress"],
                    port=self.ip_camera_config["port"]
                )
                self.canvas.camera_thread.frame_ready.connect(self.canvas._on_camera_frame)
                self.canvas.camera_thread.start()
                self.canvas.camera_enabled = True
                self.canvas.show_camera = True
                self.panel.set_status("IDLE", "CAMERA ON (phone)")
            else:
                self.canvas.show_camera = False
                self.canvas.camera_enabled = False
                self.canvas.camera_frame = None
                if self.canvas.camera_thread is not None and self.canvas.camera_thread.isRunning():
                    self.canvas.camera_thread.stop()
                self.panel.set_status("IDLE", "CAMERA OFF")
                self.canvas.update()
        except Exception as e:
            print(f"[CAMERA] Error occurred while toggling camera: {e}")
            self.panel.set_status("ERROR", f"Camera error: {e}")
            self.panel.btn_camera.setChecked(False)

    def closeEvent(self, event):
        """Stop background workers before Qt destroys the window."""
        if self.canvas.camera_thread is not None and self.canvas.camera_thread.isRunning():
            self.canvas.camera_thread.stop()
        super().closeEvent(event)
    
    @pyqtSlot()
    def _on_change_camera_ip(self):
        """Ubah IPWebcam URL berdasarkan input user"""
        current_ip = self.ip_camera_config["ip_adress"]
        new_ip, ok = QInputDialog.getText(self, "IPWebcam IP adress", 
                                          "Enter new IP address (contoh: 192.168.1.100):", text=current_ip
                                          )
        if ok and new_ip:
            self.ip_camera_config["ip_adress"] = new_ip
            print(f"[MAIN] Updated IP camera address to: {new_ip}")
            self.panel.set_status("IDLE", f"Updated camera IP to {new_ip}")
            self._start_idle_timer(2000, force=True)
    
    @pyqtSlot()
    def _on_scan_requested(self):
        if self._ocr_worker is None:
            return
        
        frame = self.canvas._latest_frame
        if frame is not None:
            self._ocr_worker.submit_frame(frame)

    @pyqtSlot(list, tuple)
    def _on_ocr_result(self, hasil, frame_shape):
        self.canvas.draw_ocr_overlay(hasil, frame_shape)
        valid = [h for h in hasil if h['valid']]



    @pyqtSlot()
    def _on_idle_timeout(self):
        """Auto return IDLE setelah timeout"""
        self.panel.set_status("IDLE", "Ready for command")
        self.panel.enable_commands(True)
        print("[Timer] Auto-idle timeout - kembali ke IDLE")

    def _start_idle_timer(self, timeMS=None, force=False):
        """Start hitung mundur ke auto-idle"""
        if self.use_serial_feedback and not force:
            return

        if not timeMS:
            self.idle_timer.start(self.IDLE_TIMEOUT_MS)
            return
        self.idle_timer.start(timeMS)

    