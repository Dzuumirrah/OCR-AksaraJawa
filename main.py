import sys

from src.ocr import OCRPipelineAksara

# import library
from src.canvas import MainWindow

import src.params as params

from PyQt5.QtWidgets import QApplication


MODEL_PATH = params.directory.ROOT_DIR / "model_aksara.keras"

CLASS_NAMES = params.model_conf.CLASS_NAMES


if __name__ == '__main__':
    # Inisialisasi GUI
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if window.isVisible():
        print("[MainWindow] : STARTED.")
    sys.exit(app.exec_())
    # 
    # ocr = OCRPipelineAksara(
    #     model_path=MODEL_PATH,
    #     class_names= CLASS_NAMES,
    #     confidence_threshold=0.6
    # )
    # hasil = ocr.proses('test/images.jpg', visualisasi=True)
    # print(CLASS_NAMES)
                