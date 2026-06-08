import os
import json

from pathlib import Path
from numpy import indices
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import TypedDict
    
# path dataset
class directory:
    ROOT_DIR = Path.cwd()
    INPUT_DIR = ROOT_DIR / "dataset/OPSI 1"
    OUTPUT_DIR = ROOT_DIR / "dataset/dataset_aksara_split/"
    MODEL_PATH = ROOT_DIR / "model_aksara.keras"
    ONNX_PATH = ROOT_DIR / "model_aksara.onnx"

# Menulis nama kelas 
if not os.path.exists('class_names.json'):
    dataset_folder = directory.INPUT_DIR
    kelas = sorted(os.listdir(dataset_folder))
    counts = {k: len(os.listdir(os.path.join(dataset_folder, k))) for k in kelas if os.path.isdir(os.path.join(dataset_folder, k))}
    names = []
    for kelas, _ in counts.items():
        names.append(kelas)

    with open('class_names.json', 'w') as f:
        json.dump(names, f, indent=4)

# Mengambil nama kelas
with open('class_names.json') as f:
    name = json.load(f)

names = name

# Konfigurasi model dan data
class model_conf:
    IMG_SIZE = 128
    BATCH_SIZE = 32
    NUM_CLASSES = 20    # Kelas data set saat ini
    EPOCS = 60
    SEED = 42
    SPLIT_RATIO = (0.70, 0.15, 0.15)  # Train, Validation, Test
    CLASS_NAMES = names

# konfigurasi OCR
class ocr_config:
    CONF_OCR = 0.6

# konfigurasi GUI
class gui:
    # Window
    MARGIN = 40
    W_MAIN, H_MAIN = 1280, 760          # keseluruhan window
    W_PANEL, H_PANEL = 280, H_MAIN
    W_CAMERA, H_CAMERA = W_MAIN - W_PANEL - MARGIN, H_MAIN    # area kamera
    RESIZABLE = True
    
    RENDER_FPS = 30
    class ColorsDict(TypedDict):
        DARKER_BLUE: str
        DARK_BLUE: str
        LIGHTER_BLUE: str
        LIGHT_BLUE: str
        DARKER_GRAY: str
        GRAY: str
        DARK_RED: str
        BLUE: str
        ORANGE: str
        GREEN: str
        RED: str
        PURPLE: str
        ACCENT: str
        BORDER: str

    COLORS:ColorsDict = {
        'DARKER_BLUE': "#1a1a2e",
        'DARK_BLUE': "#16213e",
        'LIGHTER_BLUE': '#00d4ff',
        'LIGHT_BLUE': '#3a4563',
        'DARKER_GRAY': '#555555',
        'GRAY' : "#aaaaaa",
        'DARK_RED' : "#8b0000",
        "BLUE": "#3498db",      
        "ORANGE": "#f39c12",    
        "GREEN": "#2ecc71",      
        "RED": "#e74c3c",       
        "PURPLE": "#9b59b6",
        "ACCENT": "#00d4ff",
        "BORDER": '#3a4563',
    }

class camera:
    IP_CAMERA_URL = "http://192.168.0.107"
    IP_CAMERA_PORT = 8080
    TARGET_FPS = 30
    USE_RTSP = True
# warna HEX for styling
from typing import TypedDict


