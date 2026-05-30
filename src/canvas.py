from curses import qiflush

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
    pyqtSignal
)
import cv2

from src.camera import IPWebCamThread

import src.camera as camera
import src.params as params
COLOR = params.COLORS

class  CameraWidget(QWidget):
    MARGIN = 40

    def __init__(self):
        super().__init__()
        self.setMinimumSize(params.W_CANVAS, params.H_CANVAS)
        self.setStyleSheet(f"background-color:{COLOR['DARKER_BLUE']}")
        self.setFocusPolicy(Qt.StrongFocus)

        # Camera declaration
        self.show_camera = False
        self.camera_frame = None
        self.camera_thread = None
        self.camera_enabled = False

    @pyqtSlot(object)
    def _on_camera_frame(self, frame):
        """Receive frame dari camera thread."""
        self.camera_frame = frame.copy()
        self.update() # Triger repaint


    def paintEvent(self, event):
        """Render canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.show_camera and self.camera_frame is not None:
            self._draw_camera_background(painter)
        else:
            painter.fillRect(self.rect(), QColor(COLOR['DARKER_BLUE']))
        

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
        self.setMinimumSize(params.W_MAIN, params.H_MAIN)
        self._build_layout()

        # Auto-idle timer setelah 5 detik
        self.idle_timer = QTimer()
        self.idle_timer.setSingleShot(True)
        self.idle_timer.timeout.connect(self._on_idle_timeout)
        self.IDLE_TIMEOUT_MS = 5000

        # konfigurasi IP camera
        self.ip_camera_config = {
            "ip_adress": params.IP_CAMERA_URL,
            "port": 8080
        }

        self.canvas.camera_thread = IPWebCamThread(
            self.ip_camera_config["ip_adress"],
            self.ip_camera_config["port"]
        )

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

    