from os import path
import sys

from src.ocr import OCRPipelineAksara

# import library
import os
from pathlib import Path
from src.canvas import MainWindow

from PyQt5.QtWidgets import QApplication

# Konfigurasi global
ROOT_DIR = Path.cwd() 

MODEL_PATH = ROOT_DIR / "model_aksara.keras"


if __name__ == '__main__':
    # Inisialisasi GUI
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    # ocr = OCRPipelineAksara(
    #     model_path=MODEL_PATH,
    #     class_names= CLASS_NAMES,
    #     confidence_threshold=0.6
    # )
    # hasil = ocr.proses('dataset_aksara_split/test/01-Ha/8.png', visualisasi=True)
                