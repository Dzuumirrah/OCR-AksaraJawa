from pydoc import classname
from tabnanny import verbose

import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class OCRPipelineAksara:
    def __init__(self, model_path, class_names, img_size=128, confidence_threshold=0.6):
        self.class_names = class_names
        self.img_size = img_size
        self.conf_thresh = confidence_threshold

        model_path = str(model_path)
        self._use_onnx = model_path.endswith('.onnx')
        if self._use_onnx:
            import onnxruntime as ort 
            sess_opts: ort.SessionOptions = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                model_path,
                sess_opts,
                providers=['CPUExecutionProvider']
            )
            self._input_name = self._session.get_inputs()[0].name
            self.model = None
            print(f"[OCRPipelineAksara] : ONNX model loaded: {model_path}")
        else:
            self._session = None
            self.model = tf.keras.models.load_model(model_path)
            print(f"[OCRPipelineAksara] : Keras model loaded: {model_path}")


    def binarisasi(self, img_bgr):
        """Ubah foro bgr menjadi biner"""
        # Resize ke lebar tetap untuk stabilkan adaptive threshold
        TARGET_WIDTH = 1200
        h, w = img_bgr.shape[:2]
        if w != TARGET_WIDTH:
            scale = TARGET_WIDTH / w
            img_bgr = cv2.resize(img_bgr, (TARGET_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Clahe untuk normalisasi kontras
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        biner = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=25, C=8
        )

        # Opening untuk membuang noise kecil
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        biner = cv2.morphologyEx(biner,cv2.MORPH_OPEN, kernel)

        # Closing untuk menyambung stroke yang hampir menyambung dalam satu karakter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
        biner = cv2.morphologyEx(biner,cv2.MORPH_CLOSE, kernel)

        return biner
    
    def segmentasi_karakter(self, biner):
        """
        Temukan bounding box dengan contour detection
        Terdapat Non Max Suppression (NMS) sederhana dengan filter area dan pengurutan berdasarkan posisi
        Return: list of (x, y, w, h) dengan urutan kiri->kanan, atas->bawah
        """

        img_area = biner.shape[0] * biner.shape[1]
        contours, _ = cv2.findContours(
            biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bounding_box = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            area = w * h
            # filter: abaikan kontour yang terlalu kecil atau terlalu besar
            if not (500 < area < img_area * 0.5):
                continue
            # filter: abaikan kontour yang terlalu pipih (bukan karakter) atau terlalu lancip
            aspect_ratio = w / h
            if not (0.15 < aspect_ratio < 4.0):
                continue
            # filter: abaikan bounding box yang terlalu tipis absolute
            if w < 10 or h < 10:
                continue
            bounding_box.append((x, y, w, h))

        # urutkan baris atas dulu (y), lalu kiri ke kanan (x)
        bounding_box.sort(key=lambda b: (b[1] // 50, b[0]))
        
        # Terapkan NMS
        bounding_box = self._nms(bounding_box, iou_threshold=0.3)
        return bounding_box
    
    def _nms(self, boxes, iou_threshold: float =0.3):
        """
        Non Maximum Suppression sederhana untuk menghilangkan bounding box yang tumpang tindih.
        
        Args: 
            boxes: list of (x, y, w, h)
            iou_threshold: ambang batas IOU untuk menganggap dua box sebagai tumpang tindih
        Return:
            list of (x, y, w, h) setelah NMS
        """
        if not boxes:
            return []
        
        # Konversi ke format (x1, y1, x2, y2)
        rects = np.array([[x, y, x+w, y+h] for x, y, w, h in boxes], dtype=np.float32)
        areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])

        # urutkan berdasarkan area (besar ke kecil)
        order = areas.argsort()[::-1]

        kept = []
        suppressed = np.zeros(len(rects), dtype=bool)

        for i in order:
            if suppressed[i]:
                continue
            kept.append(i)
            
            # Hitung IoU dengan box lain
            xx1 = max(rects[i, 0], rects[order, 0])
            yy1 = max(rects[i, 1], rects[order, 1])
            xx2 = min(rects[i, 2], rects[order, 2])
            yy2 = min(rects[i, 3], rects[order, 3])

            w = np.max(0.0, xx2 - xx1)
            h = np.max(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order] - inter + 1e-6
            iou = inter / union

            # suppress box yang overlap dengan box i
            to_suppress = order[iou > iou_threshold]
            suppressed[to_suppress] = True
            suppressed[i] = False  # jangan suppress box i sendiri
        
        result = []
        for i in kept:
            x1, y1, x2, y2 = rects[i]
            result.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        
        return result
    

    def _predict(self, batch: np.ndarray):
        """
        Prediksi batch dengan model yang sesuai (Keras atau ONNX)
        Input: batch dengan shape (N, img_size, img_size, 3) dan tipe float32
        Output: array dengan shape (N, num_classes) berisi probabilitas prediksi
        """
        if self._use_onnx:
            return self._session.run(None, {self._input_name: batch})[0]
        return self.model.predict(batch, verbose=0)

    def klasifikasi_satu(self, img_bgr: cv2.typing.MatLike, bbox):
        """Potong satu karater dan prediksi kelasnya"""
        x, y, w, h = bbox

        # padding untuk memperlebar potongan
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_bgr.shape[1], x + w + pad)
        y2 = min(img_bgr.shape[0], y + h + pad)
        crop = img_bgr[y1:y2, x1:x2]

        # preprocessing
        resized = cv2.resize(crop, (self.img_size, self.img_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype('float32') / 255.0
        batch = np.expand_dims(normalized, axis=0)

        # prediksi
        probs = self._predict(batch)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])

        return{
            'kelas': self.class_names[pred_idx],
            'confidence': round(confidence, 3),
            'valid': confidence >= self.conf_thresh,
            'bbox': bbox
        }
    
    def klasifikasi_batch(self, img_bgr: cv2.typing.MatLike, bboxes):
        """Klasifikasi semua karakter dalam satu batch"""
        if not bboxes:
            return []
        
        # Crops segmen
        crops = []
        for x, y, w, h in bboxes:
            pad = 4
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(img_bgr.shape[1], x + w + pad), min(img_bgr.shape[0], y + h + pad)
            crop = img_bgr[y1:y2, x1:x2]
            resized = cv2.resize(crop, (self.img_size, self.img_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            crops.append(rgb.astype('float32') / 255.0)

        # predict
        batch = np.stack(crops, axis=0)
        all_probs = self._predict(batch)

        # return hasil
        hasil = []
        for probs, bbox in zip(all_probs, bboxes):
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
            hasil.append({
                'kelas': self.class_names[pred_idx],
                'confidence': round(confidence, 3),
                'valid': confidence >= self.conf_thresh,
                'bbox': bbox
            })
        return hasil

    def proses(self, image_path, visualisasi=True):
        """
        Jalankan seluruh pipeline OCR pada satu gambar
        
        Return: list hasil prediksi per karakter
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Gambar tidak ditemukan: {image_path}")

        biner = self.binarisasi(img)
        boxes = self.segmentasi_karakter(biner)

        if not boxes:
            print("Tidak ada kaarater yang terdeteksi...")
            return []
        
        hasil = [self.klasifikasi_satu(img, box) for box in boxes]

        if visualisasi:
            self._visualisasi(img, hasil)

        # Ringkasan hasil
        valid = [h for h in hasil if h['valid']]
        print(f"\nDitemukan {len(boxes)} karakter | {len(valid)} di atas threshold {(self.conf_thresh)} ")
        print("Prediksi: ", " ".join([h['kelas'] for h in valid]))

        return hasil
    
    def _visualisasi(self, img, hasil):
        """Tampilkan gambar beserta bounding box dan label prediksi"""

        vis = img
        for h in hasil:
            x, y, w, hh = h['bbox']
            warna = (0, 200, 0) if h['valid'] else (0, 0, 200)
            if h['valid']:
                cv2.rectangle(vis, (x, y), (x+w, y+hh), warna, 2)
                label = f"{h['kelas']} {h['confidence']:.2f}"
                cv2.putText(vis, label, (x, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna)
            
        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Hasil OCR aksara Jawa")
        plt.tight_layout()
        plt.show()
