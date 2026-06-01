from pydoc import classname
from tabnanny import verbose

import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class OCRPipelineAksara:
    def __init__(self, model_path, class_names, img_size=128, confidence_threshold=0.6):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_size = img_size
        self.conf_thresh = confidence_threshold

    def binarisasi(self, img_bgr):
        """Ubah foro bgr menjadi biner"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        biner = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11,2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        biner = cv2.morphologyEx(biner,cv2.MORPH_CLOSE, kernel)
        return biner
    
    def segmentasi_karakter(self, biner):
        """
        Temukan bounding box dengan contour detection
        
        Return: list of (x, y, w, h) dengan urutan kiri->kanan, atas->bawah
        """

        contours, _ = cv2.findContours(
            biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bounding_box = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # filter: abaikan kontour yang terlalu kecil atau terlalu besar
            area = w * h
            if 200 < area < (biner.shape[0] * biner.shape[1] * 0.5):
                bounding_box.append((x, y, w, h))

        # urutkan baris atas dulu (y), lalu kiri ke kanan (x)
        bounding_box.sort(key=lambda b: (b[1] // 40, b[0]))
        return bounding_box
        
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
        probs = self.model.predict(batch, verbose=0)[0]
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
        all_probs = self.model.predict(batch, verbose=0)

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

        vis = img.copy()
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
