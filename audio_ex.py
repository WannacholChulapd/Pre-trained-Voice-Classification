from numba import none
from feature_extract import AudioFeatureExtractor
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import os
import torch
import librosa
import librosa.display

PD_Dir = os.listdir("PD")
HC_Dir = os.listdir("HC")

print("จำนวนไฟล์ PD:", len(PD_Dir))
print("จำนวนไฟล์ HC:", len(HC_Dir))

# สร้างโฟลเดอร์เก็บ Mel-spectrogram ถ้ายังไม่มี
os.makedirs("PD_melspec", exist_ok=True)
os.makedirs("HC_melspec", exist_ok=True)

# ------------------ PD ------------------
for filename in PD_Dir:
    filepath = os.path.join("PD", filename)
    extractor = AudioFeatureExtractor(filepath, sr=16000)
    # ดึง Mel-spectrogram (dB)
    mel_db = extractor.get_melspectrogram()
    print(f"{filename} -> {mel_db.shape}")
    # mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    """plt.imsave(
        os.path.join("PD_melspec", filename.replace(".wav", "_norm.png")),
        mel_norm,
        cmap="viridis",
    )"""
    plt.figure(figsize=(10, 4))

    librosa.display.specshow(mel_db, sr=16000, hop_length=extractor.hop_length)
    plt.axis("off")
    plt.tight_layout()
    # เซฟเป็นไฟล์ .png
    save_path = os.path.join("PD_melspec", filename.replace(".wav", ".png"))
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()  # ปิด figure หลัง save


for filename in HC_Dir:
    filepath = os.path.join("HC", filename)
    extractor = AudioFeatureExtractor(filepath, sr=16000)
    # ดึง Mel-spectrogram (dB)
    mel_db = extractor.get_melspectrogram()
    print(f"{filename} -> {mel_db.shape}")
    # mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    """plt.imsave(
        os.path.join("HC_melspec", filename.replace(".wav", "_norm.png")),
        mel_norm,
        cmap="viridis",
    )"""
    plt.figure(figsize=(10, 4))

    librosa.display.specshow(mel_db, sr=16000, hop_length=extractor.hop_length)
    plt.axis("off")
    plt.tight_layout()
    # เซฟเป็นไฟล์ .png
    save_path = os.path.join("HC_melspec", filename.replace(".wav", ".png"))
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()  # ปิด figure หลัง save
