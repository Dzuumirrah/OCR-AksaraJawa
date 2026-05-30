# import library
import cv2
import numpy as np
import os
import splitfolders
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter 
import pandas as pd

from src.params import directory, model_conf

INPUT_DIR = directory.INPUT_DIR
OUTPUT_DIR = directory.OUTPUT_DIR
MODEL_PATH = directory.MODEL_PATH

IMG_SIZE = model_conf.IMG_SIZE
BATCH_SIZE = model_conf.BATCH_SIZE
NUM_CLASSES = model_conf.NUM_CLASSES    # Kelas data set saat ini
EPOCS = model_conf.EPOCS
SEED = model_conf.SEED
SPLIT_RATIO = model_conf.SPLIT_RATIO  # Train, Validation, Test

try:
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory '{INPUT_DIR}' does not exist.")
except FileNotFoundError as e:
    print(e)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Split dataset menjadi 3 bagian: train, validation, test
splitfolders.ratio(
    INPUT_DIR,
    OUTPUT_DIR,
    SEED,
    SPLIT_RATIO,
    group_prefix=None,
    move=False,  # Set to False to copy files instead of moving
)
if not os.path.exists(OUTPUT_DIR / "train"):
    raise FileNotFoundError(f"Train directory '{OUTPUT_DIR / 'train'}' was not created.")
if not os.path.exists(OUTPUT_DIR / "val"):
    raise FileNotFoundError(f"Validation directory '{OUTPUT_DIR / 'val'}' was not created.")
if not os.path.exists(OUTPUT_DIR / "test"):
    raise FileNotFoundError(f"Test directory '{OUTPUT_DIR / 'test'}' was not created.")

# Verifikasi distribusi kelas
def cek_distribusi(folder):
    kelas = sorted(os.listdir(folder))
    counts = {k: len(os.listdir(os.path.join(folder, k))) for k in kelas if os.path.isdir(os.path.join(folder, k))}
    print(f"Distribusi kelas di '{folder}':")
    for k, v in counts.items():
        print(f"  {k:<20}: {v:>8}")
    print(f"\nTotal: {sum(counts.values())} | Min: {min(counts.values())} | Max: {max(counts.values())}\n")
print("=== Distribusi Kelas","="*50)
cek_distribusi(OUTPUT_DIR / "train")
print("="*70)

