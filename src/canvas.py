"""
Struktur GUI:
    CameraWidget: Area kamera (kiri)
    ControlPanel: Area panel (kanan)
    StatusBar: Area status (bawah)
    MainWindow : Window utama
"""
import time

from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QLayoutItem,
    QLineEdit,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QInputDialog,
    QGroupBox,
    QGridLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QSlider,
    QStatusBar,
    QSizePolicy,
)
from PyQt5.QtGui import(
    QPaintEvent,
    QPainter,
    QColor,
    QImage,
    QPixmap,
    QPen,
    QFont,
)
from PyQt5.QtCore import (
    QTimer, 
    pyqtSlot,
    Qt,
    pyqtSignal,
    QMutex,
    QMutexLocker,
    QThread,
    QSize
)
import cv2
from threading import Lock as ThreadLock


from matplotlib.pylab import box
from numpy import dot
from src.ocr import OCRPipelineAksara
from src.camera import IPWebCamThread

from src.params import directory, gui, camera, model_conf, ocr_config
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
            self._pending_frame = frame

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

def _lighten(hex_color: str, factor: float = 1.18) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        min(255, int(r * factor)),
        min(255, int(g * factor)),
        min(255, int(b * factor)),
    )
 
def _darken(hex_color: str, factor: float = 0.72) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(int(r*factor), int(g*factor), int(b*factor))

# Styling
def _btn_style(bg: str, text_color: str = "white", border: str = None) -> str:
    bd = border or ["BORDER"]
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {text_color};
            border: 1px solid {bd};
            border-radius: 5px;
            font-weight: 600;
            font-size: 11px;
            padding: 5px 8px;
        }}
        QPushButton:hover    {{ background-color: {_lighten(bg)}; }}
        QPushButton:pressed  {{ background-color: {_darken(bg)};  }}
        QPushButton:disabled {{ background-color: {COLOR['DARKER_GRAY']}; color: {COLOR['GRAY']}; }}
        QPushButton:checked  {{ border-color: {COLOR['ACCENT']}; }}
    """
 
def _groupbox_style(title_color: str = COLOR["ACCENT"]) -> str:
    return f"""
        QGroupBox {{
            border: 1px solid {COLOR['BORDER']};
            border-radius: 6px;
            margin-top: 14px;
            padding-top: 6px;
            color: {title_color};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 4px;
        }}
    """

class CameraWidget(QWidget):
    MARGIN = gui.MARGIN                 # Margin 
    RENDER_FPS = gui.RENDER_FPS         # maksimum render FPS

    def __init__(self) -> None:
        super().__init__()
        
        # GUI setup
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"""background-color:{COLOR['DARKER_BLUE']}""")
        self.setFocusPolicy(Qt.StrongFocus)

        self._frame_aspect_ratio = None

        # Camera declaration
        self.show_camera = False
        self.camera_frame = None
        self.camera_enabled = False

        # Camera rendering
        self._cached_pixmap = None
        self._pixmap_lock = QMutex()
        self._latest_frame = None
        
        # Frame aktual
        self._frame_count = 0
        self._fps_ts = time.monotonic()
        self.actual_fps = 0.0

        # timer auto painter        

        # timer render
        self._render_timer = QTimer()
        self._render_timer.setInterval(1000 // self.RENDER_FPS)
        self._render_timer.timeout.connect(self.update)
        self._render_timer.start()

        # Hasil OCR
        self._ocr_results = []
        self._ocr_frame_shape = None

        # Hasil saat scan di frame terakhir
        self._frozen = False        # True jika hasil OCR dibekukan untuk frame terakhir
        self._frozen_pixmap = None
        self._frozen_frame = None

    @pyqtSlot(object)
    def _on_camera_frame(self, frame: cv2.typing.MatLike) -> None:
        """Receive frame dari camera thread lalu render dari frame ke pixmap"""
        if not self.show_camera or self._frozen:
            return

        widget_w, widget_h = self.width(), self.height()
        h, w, ch = frame.shape
        
        if self._frame_aspect_ratio is None:
            self._frame_aspect_ratio = w / h
            self.updateGeometry()  # Trigger layout recalculation


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
            self._latest_frame = frame.copy()

        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_ts
        if elapsed >= 2:
            self.actual_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_ts = now
        # # Triger repaint sesuai FPS dari timer
        # self.update()
        
    def sizeHint(self):
        """Return preferred size maintaining frame aspect ratio"""
        if self._frame_aspect_ratio:
            parent_w = self.parent().width() if self.parent() else 800
            preferred_h = int(parent_w / self._frame_aspect_ratio)
            return QSize(parent_w, preferred_h)
        return super().sizeHint()

    def paintEvent(self, event) -> None:
        """Render canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        with QMutexLocker(self._pixmap_lock):
            pixmap = self._cached_pixmap

        if self.show_camera and pixmap is not None:
            # center pixmap
            x = (self.width() - pixmap.width()) // 2
            y = (self.height() - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)
            
            if self._ocr_results and self._ocr_frame_shape:
                self.draw_ocr_boxes(painter, pixmap, x, y)
                
            if self._frozen:
                painter.setFont(QFont("Arial", 9, QFont.Bold))
                painter.setPen(QColor(COLOR["ORANGE"]))
                painter.drawText(8, self.height() - 8, "⏸ FROZEN")
        else:
            # blank jika kamera mati
            painter.fillRect(self.rect(), QColor(COLOR['DARKER_BLUE']))
            painter.setPen(QColor(COLOR["LIGHTER_BLUE"]))
            
            # garis silang kamera
            cx, cy = self.width() // 2, self.height() // 2
            painter.setFont(QFont("Arial", 11))
            painter.setPen(QColor(COLOR["GRAY"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "Kamera mati")

    def draw_ocr_overlay(self, hasil: list, frame_shape: tuple):
        self._ocr_results = hasil
        self._ocr_frame_shape = frame_shape

    def draw_ocr_boxes(self, painter: QPainter, pixmap: QPixmap, offset_x, offset_y):
        """Gambar bounding box dan label OCR di atas frame yang ditampilkan"""
        if not self._ocr_results or self._ocr_frame_shape is None:
            return
        
        orig_h, orig_w = self._ocr_frame_shape[:2]
        scale_x = pixmap.width() / orig_w
        scale_y = pixmap.height() / orig_h

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

    def freeze (self) -> None:
        """Freeze tampilan di frame terakhir. """
        self._frozen = True

    def unfreeze(self) -> None:
        """Lanjutkan tampilan frame baru."""
        self._frozen = False
        self._ocr_results = []
        self._ocr_frame_shape = None

class AksaraChip(QFrame):
    """Kartu kecil menampilkan satu karakter OCR: nama + confidence bar."""

    def __init__(self, nama: str, confidence: float, valid: bool, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        border_color = COLOR["GREEN"] if valid else COLOR["RED"]
        self.setStyleSheet(f"""
            QFrame{{
                background-color: {COLOR['DARK_BLUE']};
                border: 1px solid {border_color};
                border-radius: 5px;
                padding: 2px
            }}            
            """
        )

        self.setFixedSize(52, 52)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(2)

        # nama karater
        label_name = QLabel(nama)
        label_name.setAlignment(Qt.AlignCenter)
        label_name.setStyleSheet("color: white; font-weight: bold; font-size: 13px; border: none;")

        # Confidence bar
        bar = _ConfBar(confidence, valid, self)

        # Skor confidence
        label_conf = QLabel(f"{confidence:.1f}")
        label_conf.setAlignment(Qt.AlignCenter)
        color = COLOR["GREEN"] if valid else COLOR["GRAY"]
        label_conf.setStyleSheet("color: white; font-size: 9px; border: none;")

        layout.addWidget(label_name)
        layout.addWidget(bar)
        layout.addWidget(label_conf)

class _ConfBar (QWidget):
    """Progress bar untuk confidence"""
    def __init__(self, value: float, valid: bool, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._value = max(0.0, min(1.0, value))
        self._color = COLOR["GREEN"] if valid else COLOR["RED"]
        self.setFixedHeight(4)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Background bar
        painter.fillRect(self.rect(), QColor(COLOR['DARKER_GRAY']))
        # Progress fill
        fill_width = int(self.width() * self._value)
        if fill_width > 0:
            painter.fillRect(0, 0, fill_width, self.height(), QColor(self._color))

class _FlowLayout(QGridLayout):
    """
    Layout untuk mengatur widget hasil OCR aksara dalam baris mengalir.
    """
    def __init__(self, parent: QWidget | None, margin: int = 6, spacing: int = 5):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

        self._cols = 4
        self._idx = 0
    
    def addWidget(self, widget: QWidget):   # type: ignore[override]
        row, col = divmod(self._idx, self._cols)
        super().addWidget(widget, row, col)
        self._idx += 1

    def takeAt(self, index: int) -> QLayoutItem | None:
        item = super().takeAt(index)
        if item:
            self._idx = max(0, self._idx - 1)
        return item

    def reset(self):
        self._idx = 0 

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
    """
    - Kamera: Toggle on/off, IP input, stream mode
    - Deteksi: Tombol scan + auto, slider confidence threshold
    - hasil: Chip per karakter, tombol salin/hapus
    """
    # tombol
    camera_toggle = pyqtSignal(bool)    # True = aktifkan kamera, False = matikan kamera    
    auto_scan_toggle = pyqtSignal(bool) # True = aktifkan auto-scan
    
    scan_requested = pyqtSignal()           # Trigger OCR sekali
    threshold_changed = pyqtSignal(float)   # Nilai threshold baru
    ip_changed = pyqtSignal(str)            # IP baru setelah diedit

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setFixedWidth(gui.W_PANEL)
        self.setStyleSheet(f"""
                           QWidget {{background-color: {COLOR['DARKER_BLUE']}; color: white;}}
                           QLabel {{background: transparent}}
                           """)

        self._cam_on = False
        self._auto_scan_on = False
        self._threshold = ocr_config.CONF_OCR

        self._build_layout()
        
    def _build_layout(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        main_layout.addWidget(self._build_camera_section())
        main_layout.addWidget(self._build_detection_section())
        main_layout.addWidget(self._build_result_section())
        main_layout.addStretch()

    def _build_camera_section(self) -> QGroupBox:
        box = QGroupBox("IPWebcam CAMERA")
        box.setStyleSheet(_groupbox_style())

        # layout utama
        layout = QVBoxLayout(box)
        layout.setSpacing(7)
        layout.setContentsMargins(8, 14, 8, 8)

        # === IP CAMERA SECTION ===================
        ip_section = QHBoxLayout()
        # Label judul section
        label_ip = QLabel("IP Camera")
        label_ip.setStyleSheet(f"""
                               color: {COLOR['GRAY']}; font-size: 10px
                               """)
        label_ip.setFixedWidth(16)
        
        # Kolom input IP (metode commit dengan enter dan button)
        self.ip_input = QLineEdit(camera.IP_CAMERA_URL)
        self.ip_input.setStyleSheet(f"""
            QLineEdit {{
                background: {COLOR['DARK_BLUE']};
                color: {COLOR['LIGHTER_BLUE']};
                border: 1px solid {COLOR['BORDER']};
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
                padding: 3px 6px;
            }}
            QLineEdit:focus {{ border-color: {COLOR['ACCENT']}; }}
        """)
        self.ip_input.setPlaceholderText("192.168.x.x")
        self.ip_input.returnPressed.connect(self._on_ip_commit)

        btn_apply = QPushButton("Change IP")
        btn_apply.setFixedSize(26, 26)
        btn_apply.setStyleSheet(_btn_style(COLOR["LIGHT_BLUE"]))
        btn_apply.setToolTip("Change IPWebcam IP adress")
        btn_apply.clicked.connect(self._on_ip_commit)
        
        ip_section.addWidget(label_ip)
        ip_section.addWidget(self.ip_input)
        ip_section.addWidget(btn_apply)

        # === PORT CAMERA SECTION ===================
        port_section = QHBoxLayout()
        label_port = QLabel("Port Camera")
        label_port.setStyleSheet(f"""color: {COLOR["GRAY"]}; font-size: 10px""")
        label_port.setFixedWidth(16)

        self.port_input = QLineEdit(str(camera.IP_CAMERA_PORT))
        self.port_input.setFixedWidth(52)
        self.port_input.setStyleSheet(f"""
            QLineEdit {{
                background: {COLOR['DARK_BLUE']};
                color: {COLOR['LIGHTER_BLUE']};
                border: 1px solid {COLOR['BORDER']};
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
                padding: 3px 6px;
            }}
            QLineEdit:focus {{ border-color: {COLOR['ACCENT']}; }}
        """)
        # Kolom badge stream (MJPEG | RTSP)
        self.stream_badge = QPushButton("MJEPG")
        self.stream_badge.setCheckable(True)
        self.stream_badge.setToolTip("Click to switch [MJPEG | RTSP]")
        self.stream_badge.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR['LIGHT_BLUE']};
                color: {COLOR['ACCENT']};
                border: 1px solid {COLOR['BORDER']};
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                padding: 2px 6px;
            }}
            QPushButton:checked {{ 
                border-color: {COLOR['GREEN']};
                color: {COLOR['GREEN']}
                background: {COLOR['DARKER_BLUE']} 
                }}
        """)
        self.stream_badge.toggled.connect(
            lambda on: self.stream_badge.setText("RTSP" if on else "MJPEG")
        )

        port_section.addWidget(label_port)
        port_section.addWidget(self.port_input)
        port_section.addStretch()
        port_section.addWidget(self.stream_badge)

        # === SEPARATOR ===================
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"""color: {COLOR['BORDER']};""")
        
        # === TOGGLE CAMERA SECTION =======
        self.btn_camera = QPushButton("Activate camera")
        self.btn_camera.setCheckable(True)
        self.btn_camera.setFixedHeight(36)
        self.btn_camera.setStyleSheet(_btn_style(COLOR['LIGHT_BLUE']))
        self.btn_camera.toggled.connect(self._on_camera_toggle)

        layout.addLayout(ip_section)
        layout.addLayout(port_section)

        layout.addWidget(sep)
        layout.addWidget(self.btn_camera)

        return box

    def _build_detection_section(self) -> QGroupBox:
        box = QGroupBox("DETECTION")
        box.setStyleSheet(_groupbox_style())
        # layout utama
        layout = QVBoxLayout(box)
        layout.setSpacing(7)
        layout.setContentsMargins(8, 14, 8, 8)

        # === SCAN TOGGLE SECTION ===================
        section_scan = QHBoxLayout()

        self.btn_scan = QPushButton("▶  Scan")
        self.btn_scan.setFixedHeight(34)
        self.btn_scan.setStyleSheet(_btn_style(COLOR["BLUE"]))
        self.btn_scan.setToolTip("Start Scan on Frame")
        self.btn_scan.clicked.connect(self.scan_requested.emit)
        
        self.btn_auto = QPushButton("↺ Auto")
        self.btn_auto.setFixedHeight(34)
        self.btn_auto.setCheckable(True)
        self.btn_auto.setStyleSheet(_btn_style(COLOR["LIGHT_BLUE"]))
        self.btn_auto.setToolTip("Start Auto Scan")
        self.btn_auto.clicked.connect(self._on_autoscan_toggle)

        section_scan.addWidget(self.btn_scan)
        section_scan.addWidget(self.btn_auto)

        # === THRESHOLD SLIDER SECTION ===================
        section_threshold = QHBoxLayout()
        # Label
        label_thresh = QLabel("Threshold")
        label_thresh.setStyleSheet(f"""color: {COLOR["GRAY"]}; font-size: 10px""")
        label_thresh.setFixedWidth(60)

        # Slider
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(10, 99)
        self.slider_thresh.setValue(int(self._threshold * 100))
        self.slider_thresh.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: {COLOR['BORDER']};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {COLOR['ACCENT']};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLOR['ACCENT']};
                border-radius: 2px;
            }}
        """)
        self.slider_thresh.valueChanged.connect(self._on_threshold_change)

        # Threshold value Label
        self.label_thresh_val = QLabel(f"{self._threshold:.1f}")
        self.label_thresh_val.setFixedWidth(36)
        self.label_thresh_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.label_thresh_val.setStyleSheet(f"""color: {COLOR['ACCENT']}; font-size: 11px; font-family: monospace;""")
        
        # Input numerik threshold
        self.thresh_input = QLineEdit(f"{self._threshold:.2f}")
        self.thresh_input.setFixedWidth(40)
        self.thresh_input.setStyleSheet(f"""
            QLineEdit {{
                background: {COLOR['DARK_BLUE']};
                color: {COLOR['ACCENT']};
                border: 1px solid {COLOR['BORDER']};
                border-radius: 3px;
                font-family: monospace;
                font-size: 11px;
                padding: 1px 4px;
            }}
            QLineEdit:focus {{ border-color: {COLOR['ACCENT']}; }}
        """)
        self.thresh_input.returnPressed.connect(self._on_thresh_input_commit)

        section_threshold.addWidget(label_thresh)
        section_threshold.addWidget(self.slider_thresh)
        section_threshold.addWidget(self.label_thresh_val)
        section_threshold.addWidget(self.thresh_input)

        layout.addLayout(section_scan)
        layout.addLayout(section_threshold)

        return box
    
    def _build_result_section(self) -> QGroupBox:
        box = QGroupBox("RESULT")
        box.setStyleSheet(_groupbox_style())

        # layout utama
        layout = QVBoxLayout(box)
        layout.setSpacing(7)
        layout.setContentsMargins(8, 14, 8, 8)

        # === CHIP AREA SECTION ===================
        # chip scroll area
        section_scroll_area = QScrollArea()
        section_scroll_area.setWidgetResizable(True)
        section_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        section_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        section_scroll_area.setFixedHeight(110)
        section_scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {COLOR['BORDER']};
                border-radius: 5px;
                background: {COLOR['DARK_BLUE']};
            }}
            QScrollBar:vertical {{
                width: 6px;
                background: {COLOR['DARKER_BLUE']};
            }}
            QScrollBar::handle:vertical {{
                background: {COLOR['LIGHT_BLUE']};
                border-radius: 3px;
                min-height: 20px;
            }}
        """ )

        # chip container
        self._chip_container = QWidget()
        self._chip_container.setStyleSheet(f"""background: {COLOR['DARK_BLUE']}""")
        self._chip_layout = _FlowLayout(self._chip_container)
        self._chip_container.setLayout(self._chip_layout)

        section_scroll_area.setWidget(self._chip_container)

        # label placeholder ketika belum ada hasil OCR
        self._label_empty = QLabel("OCR result empty")
        self._label_empty.setAlignment(Qt.AlignCenter)
        self._label_empty.setStyleSheet(f"""color: {COLOR['GRAY']}; font-size: 11px""")
        
        self._chip_layout.addWidget(self._label_empty)

        # === COPY PASTE SECTION ===================
        section_result_action = QHBoxLayout()

        self.btn_copy = QPushButton("⎘  Copy")
        self.btn_copy.setFixedHeight(28)
        self.btn_copy.setStyleSheet(_btn_style(COLOR['LIGHT_BLUE']))
        self.btn_copy.clicked.connect(self._on_copy)
        
        self.btn_clear = QPushButton("✕  Clear")
        self.btn_clear.setFixedHeight(28)
        self.btn_clear.setStyleSheet(_btn_style(COLOR['DARKER_GRAY']))
        self.btn_clear.clicked.connect(self.clear_results)

        section_result_action.addWidget(self.btn_copy)
        section_result_action.addWidget(self.btn_clear)

        layout.addWidget(section_scroll_area)
        layout.addLayout(section_result_action)

        # Simpan hasil OCR untuk copy
        self._ocr_names: list[str] = []

        return box

    # === COMMAND HANDLERS ============
    

    def _on_ip_commit(self):
        """Handle change camera IP button click."""
        ip = self.ip_input.text().strip()
        if ip:
            self.ip_changed.emit(ip)
    
    def _on_camera_toggle(self):
        """Handle camera toggle."""
        is_checked = self.btn_camera.isChecked()
        if is_checked:
            self.btn_camera.setText("CAMERA ON")
            self.btn_camera.setStyleSheet(_btn_style(COLOR["GREEN"], COLOR['GREEN']))
            self.camera_toggle.emit(True)
        else:
            self.btn_camera.setText("CAMERA OFF")
            self.btn_camera.setStyleSheet(_btn_style(COLOR["LIGHT_BLUE"]))
            self.camera_toggle.emit(False)
    
    def _on_autoscan_toggle(self):
        """Handle auto scan toggle"""
        is_checked = self.btn_auto.isChecked()
        if is_checked:
            self.btn_auto.setText("↺ Auto ON")
            self.btn_auto.setStyleSheet(_btn_style(COLOR["BLUE"], COLOR["ACCENT"]))
        else:
            self.btn_auto.setText("↺ Auto")
            self.btn_auto.setStyleSheet(_btn_style(COLOR["LIGHT_BLUE"]))
        self.auto_scan_toggle.emit(is_checked)

    def _on_threshold_change(self):
        """Handle perubahan threshold dari slider"""
        value = self.slider_thresh.value()
        self._threshold = value / 100.0
        self.label_thresh_val.setText(f"{self._threshold:.1f}")
        self.thresh_input.setText(f"{self._threshold:.2f}")
        self.threshold_changed.emit(self._threshold)

    def _on_thresh_input_commit(self):
        """Handle perubahan threshold dari input numerik"""
        try:
            val = float(self.thresh_input.text())
            val = max(0.1, min(0.99, val))
            self._threshold = val
            self.slider_thresh.blockSignals(True)
            self.slider_thresh.setValue(int(val * 100))
            self.slider_thresh.blockSignals(False)
            self.label_thresh_val.setText(f"{val:.2f}")
            self.thresh_input.setText(f"{val:.2f}")
            self.threshold_changed.emit(val)
        except ValueError:
            self.thresh_input.setText(f"{self._threshold:.2f}")
    
    def _on_copy(self):
        """Handle button copy result OCR"""
        if self._ocr_names:
            QApplication.clipboard().setText(" ".join(self._ocr_names))
    
    # === PUBLIC METHODS ===========    
    def set_camera_button_error(self):
        """Reset tombol kamera ke off dan tampilkan warna error sesaat"""
        self.btn_camera.setChecked(False)
        self.btn_camera.setText("  Activate Kamera")
        self.btn_camera.setStyleSheet(_btn_style(COLOR['RED']))
        QTimer.singleShot(
            2000,
            lambda: self.btn_camera.setStyleSheet(_btn_style(COLOR['LIGHT_BLUE']))
        )   
    
    def update_result(self, hasil: list):
        """
        Tampilkan list hasil OCR sebagai chip
        Params:
            - dict list {'kelas', 'confidence', 'valid'}
        """
        self.clear_results()
        self._ocr_names = []

        if not hasil:
            self._label_empty.show()
            return

        self._label_empty.hide()
        for h in hasil:
            chip = AksaraChip(h['kelas'], h['confidence'], h['valid'])
            self._chip_layout.addWidget(chip)
            if h['valid']:
                self._ocr_names.append(h['kelas'])
    def clear_results(self):
        """Hapus semua chip dari area hasil"""
        while self._chip_layout.count():
            item = self._chip_layout.takeAt(0)
            if item and item.widget() and item.widget() is not self._label_empty:
                item.widget().deleteLater()
        
        self._label_empty.show()
        self._ocr_names = []
        
    def get_threshold(self) -> float:
        return self._threshold

    def get_ip(self) -> str:
        return self.ip_input.text().strip()
    
    def get_port(self):
        try:
            return int(self.port_input.text().strip())
        except ValueError:
            return camera.IP_CAMERA_PORT

class AppStatusBar(QStatusBar):
    """
    Status bar di bawah window
    UI:
        - Koneksi
        - FPS
        - Jumlah Karakter
        - Nama Model
    """
    def __init__(self, parent: QWidget | None = ...) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"""
            QStatusBar {{
                background: {COLOR['DARK_BLUE']};
                color: {COLOR['GRAY']};
                border-top: 1px solid {COLOR['BORDER']};
                font-size: 11px;
            }}
            QLabel {{ background: transparent; color: {COLOR['GRAY']}; padding: 0 8px; }}
        """)
        self.setSizeGripEnabled(False)
        
        # === UI ====================================
        self._dot = QLabel("●")
        self._label_connection  = QLabel("Disconnected")
        self._label_sep1  = QLabel("|")
        self._label_fps   = QLabel("— FPS")
        self._label_sep2  = QLabel("|")
        self._label_chars = QLabel("0 char")
        self._label_sep3  = QLabel("|")
        self._label_model = QLabel("—")
        self._label_sep4  = QLabel("|")
        self._label_ip    = QLabel("—")

        # Styling element
        self._dot.setStyleSheet(f"""color: {COLOR['GRAY']}; padding: 0 4px;""")
        self._label_sep1.setStyleSheet(f"""color: {COLOR['BORDER']};""")
        self._label_sep2.setStyleSheet(f"""color: {COLOR['BORDER']};""")
        self._label_sep3.setStyleSheet(f"""color: {COLOR['BORDER']};""")
        self._label_sep4.setStyleSheet(f"""color: {COLOR['BORDER']};""")

        # looping untuk add widget
        for widget in (
            self._dot, 
            self._label_connection,
            self._label_sep1,
            self._label_fps,
            self._label_sep2,
            self._label_chars,
            self._label_sep3, 
            self._label_model,
            self._label_sep4,
            self._label_ip
        ):
            self.addWidget(widget)

    def set_connected(self, ok: bool):
        color = COLOR['GREEN'] if ok else COLOR['GRAY']
        self._dot.setStyleSheet(f"""color: {color}; padding: 0 4px""")
        self._label_connection.setText("Connected" if ok else "Disconnected")
        self._label_connection.setStyleSheet(f"""color:{color}; background: transparent;
                                             padding: 0 8px""")
    
    def set_fps(self, fps: float):
        self._label_fps.setText(f"{fps:.1f} FPS" if fps > 0 else "- FPS")

    def set_char_count(self, n: int):
        self._label_chars.setText(f"{n} char")

    def set_model_name(self, name: str):
        self._label_model.setText(name)
    
    def set_ip(self, ip: str):
        self._label_ip.setText(ip)

class MainWindow(QMainWindow):
    def __init__ (self):
        super().__init__()
        self.setWindowTitle("OCR Aksara Jawa dengan IPWebcam camera")
        if gui.RESIZABLE:
            self.setMinimumSize(gui.W_MAIN, gui.H_MAIN)
            self.resize(gui.W_MAIN, gui.H_MAIN)
        else:
            self.setFixedSize(gui.W_MAIN, gui.H_MAIN)
        
        self.setStyleSheet(f"background-color:{COLOR['DARKER_BLUE']}")
        
        # konfigurasi IP camera
        self.ip_camera_config = {
            "ip_address": camera.IP_CAMERA_URL,
            "port": camera.IP_CAMERA_PORT
        }
        # konfigurasi OCR
        self._threshold = ocr_config.CONF_OCR

        # Timer auto scan
        self._auto_scan_timer = QTimer(self)
        self._auto_scan_timer.setInterval(200)   # Dalam ms
        self._auto_scan_timer.timeout.connect(self._on_scan_requested)
        
        # timer update FPS di status bar
        self._fps_update_timer = QTimer(self)
        self._fps_update_timer.setInterval(1000)
        self._fps_update_timer.timeout.connect(self._on_update_status_fps)
        self._fps_update_timer.start()

        self._build_layout()
        self._connect_signals()

        
        # Thread untuk menampilkan kamera ke canvas
        self.canvas.camera_thread = IPWebCamThread(
                    self.ip_camera_config["ip_address"],
                    port=self.ip_camera_config["port"]
                )
        _model_path = directory.ONNX_PATH if directory.ONNX_PATH.exists() else directory.MODEL_PATH
        model_name = _model_path.name

        self.status_bar.set_model_name(model_name)
        self.status_bar.set_ip(f"{self.ip_camera_config['ip_address']}:{self.ip_camera_config['port']}")

        # Thread untuk menampilkan hasil OCR ke canvas
        try:
            ocr_pipeline = OCRPipelineAksara(
                model_path = _model_path,
                class_names = model_conf.CLASS_NAMES,
                confidence_threshold= ocr_config.CONF_OCR
            )
            
            self._ocr_worker = OCRWorkerThread(ocr_pipeline)
            self._ocr_worker.result_ready.connect(self._on_ocr_result)
            self._ocr_worker.start()
        except Exception as e:
            print(f"[MainWindow] OCR model cannot be loaded: {e}")
            self._ocr_worker = None
  
    def _build_layout(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # Kiri: Camera area
        self.canvas = CameraWidget()
        # Kanan: Panel area
        self.panel = ControlPanel()

        # Separator vertika
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {COLOR['BORDER']}")

        root_layout.addWidget(self.canvas, 1)
        root_layout.addWidget(sep)
        root_layout.addWidget(self.panel)

        # Status bar
        self.status_bar = AppStatusBar(self)
        self.setStatusBar(self.status_bar)

    def _connect_signals(self):
        self.panel.camera_toggle.connect(self._on_camera_toggle)
        self.panel.scan_requested.connect(self._on_scan_requested)
        self.panel.auto_scan_toggle.connect(self._on_auto_scan_toggled)
        self.panel.threshold_changed.connect(self._on_threshold_changed)
        self.panel.ip_changed.connect(self._on_ip_changed)

    @pyqtSlot(bool)
    def _on_camera_toggle(self, enabled):
        """Handle camera toggle from control panel"""
        try:
            if enabled:
                if self.canvas.camera_thread is not None and self.canvas.camera_thread.isRunning():
                    self.canvas.camera_thread.stop()
                    self.canvas.camera_thread.wait()  # Wait for thread to finish before starting a new one
                
                # Buat thread baru dengan config terkini
                self.canvas.camera_thread = IPWebCamThread(
                    self.ip_camera_config["ip_address"],
                    port=self.ip_camera_config["port"]
                )
                self.canvas.camera_thread.frame_ready.connect(self.canvas._on_camera_frame)
                self.canvas.camera_thread.frame_ready.connect(self._on_first_frame)
                self.canvas.camera_thread.stream_connected.connect(self._on_stream_connected)
                self.canvas.camera_thread.start()

                self.canvas.camera_enabled = True
                self.canvas.show_camera = True

                 # Timeout fallback: jika 5 detik tidak ada frame → error
                self._conn_timeout = QTimer(self)
                self._conn_timeout.setSingleShot(True)
                self._conn_timeout.timeout.connect(self._on_connection_timeout)
                self._conn_timeout.start(5000)
                
            else:
                self._cancel_conn_timeout()
                self.canvas.show_camera = False
                self.canvas.camera_enabled = False
                self.canvas.unfreeze()
                
                if (self.canvas.camera_thread is not None 
                    and self.canvas.camera_thread.isRunning()):
                    self.canvas.camera_thread.stop()
                self.canvas._cached_pixmap = None
                self.canvas.actual_fps = 0.0
                self.canvas.update()

                self.status_bar.set_connected(False)
                self.status_bar.set_fps(0.0)

        except Exception as e:
            print(f"[MainWindow] Error occurred while toggling camera: {e}")
            self.panel.set_camera_button_error()
            self.status_bar.set_connected(False)


    # === Handler =================================    
    @pyqtSlot(bool)
    def _on_auto_scan_toggled(self, enabled: bool):
        if enabled:
            self.canvas.unfreeze()  # Pastikan canvas tidak beku saat auto-scan diaktifkan
            self._auto_scan_timer.start()
        else:
            self._auto_scan_timer.stop()
            self.canvas.unfreeze()  # Unfreeze canvas saat auto-scan dimatikan, agar bisa scan manual lagi

    @pyqtSlot(float)
    def _on_threshold_changed(self, value: float):
        self._threshold = value
        if self._ocr_worker and self._ocr_worker.ocr:
            self._ocr_worker.ocr.conf_thresh = value

    @pyqtSlot(str)
    def _on_ip_changed(self, ip: str):
        """Ubah IPWebcam URL berdasarkan input user"""
        self.canvas.camera_thread.stop()
        self.canvas.camera_thread.wait()  # Wait for thread to finish
    
        # Update config
        self.ip_camera_config["ip_address"] = ip
        
        # Create and start new thread
        self.canvas.camera_thread = IPWebCamThread(
            self.ip_camera_config["ip_address"],
            port=self.ip_camera_config["port"]
        )
        self.canvas.camera_thread.start()

        self.status_bar.set_ip(ip)
        print(f"[MainWindow] IP Camera sucsessfully chaged: {self.ip_camera_config['ip_address']}:{self.ip_camera_config['port']}")
    
        
    def _on_update_status_fps(self):
        if self.canvas.camera_enabled:
            self.status_bar.set_fps(self.canvas.actual_fps)
      
    @pyqtSlot()
    def _on_scan_requested(self):
        """
        Trigger OCR pada frame terakhir.
        """
        if self._ocr_worker is None:
            return
        
        is_auto_scan = self._auto_scan_timer.isActive()
        # Unfreeze canvas sekali lalu OCR frame terakhir untuk scan manual
        if self.canvas._frozen and not is_auto_scan:
            self.canvas.unfreeze()
            return
        
        frame = self.canvas._latest_frame
        if frame is not None:
            self._ocr_worker.submit_frame(frame)

            if not is_auto_scan:
                self.canvas.freeze()  # Bekukan canvas setelah scan manual, agar hasilnya tidak berubah sampai user toggle lagi

    @pyqtSlot(list, tuple)
    def _on_ocr_result(self, hasil, frame_shape):
        self.canvas.draw_ocr_overlay(hasil, frame_shape)
        valid = [h for h in hasil if h['valid']]
        self.panel.update_result(hasil)
        self.status_bar.set_char_count(len(valid))

    def _cancel_conn_timeout(self):
        if hasattr(self, "_conn_timeout") and self._conn_timeout is not None:
            self._conn_timeout.stop()
            self._conn_timeout = None
    
    @pyqtSlot()
    def _on_first_frame(self):
        """Handle frame pertama diterima untuk update status koneksi"""
        self._cancel_conn_timeout()
        self.status_bar.set_connected(True)
        try:
            self.canvas.camera_thread.frame_ready.disconnect(self._on_first_frame)
        except Exception:
            pass

    @pyqtSlot()
    def _on_connection_timeout(self):
        """Handle timeout ketika tidak ada frame diterima setelah toggle kamera"""
        print(f"[MainWindow] Connection timeout triggered.")
        self.panel.set_camera_button_error()
        self.status_bar.set_connected(False)
        if self.canvas.camera_thread is not None and self.canvas.camera_thread.isRunning():
            self.canvas.camera_thread.stop()

        self.canvas.camera_enabled = False
        self.canvas.show_camera = False

    @pyqtSlot(str)
    def _on_stream_connected(self, stream_type: str):
        """Handle update status ketika stream berhasil terhubung"""
        print(f"[MainWindow] Stream connected: {stream_type}")
        is_rtsp = stream_type.lower() == "rtsp"
        self.panel.stream_badge.blockSignals(True)
        self.panel.stream_badge.setChecked(is_rtsp)
        self.panel.stream_badge.setText("RTSP" if is_rtsp else "MJPEG")
        self.panel.stream_badge.blockSignals(False)


    def closeEvent(self, event):
        """Stop background workers before Qt destroys the window."""
        if self.canvas.camera_thread is not None and self.canvas.camera_thread.isRunning():
            self.canvas.camera_thread.stop()
        self._auto_scan_timer.stop()
        self._fps_update_timer.stop()
        if self._ocr_worker is not None:
            self._ocr_worker.stop()
        super().closeEvent(event)