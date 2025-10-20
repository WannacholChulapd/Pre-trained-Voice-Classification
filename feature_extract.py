import torch
import torchaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torchaudio import transforms as T
from torchaudio import functional as F


class AudioFeatureExtractor:
    def __init__(self, wavfile, sr=16000, n_fft=1024, hop_length=51, n_mels=256):
        self.wavfile = wavfile
        self.target_sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # โหลดไฟล์เสียง
        waveform, original_sr = torchaudio.load(self.wavfile)

        # resample ถ้าจำเป็น
        if original_sr != self.target_sr:
            resampler = T.Resample(orig_freq=original_sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        self.waveform = waveform
        self.sr = self.target_sr

        # เตรียม transform ที่ใช้บ่อย
        self.spectrogram_transform = T.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        self.griffin_lim = T.GriffinLim(n_fft=self.n_fft)
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            norm="slaney",
            n_mels=self.n_mels,
            mel_scale="htk",
            win_length=None,
            pad_mode="reflect",
            center=True,
            window_fn=torch.hann_window,
        )

    def get_spectrogram(self, to_db=True):
        spec = self.spectrogram_transform(self.waveform)[0].numpy()
        if to_db:
            spec = librosa.power_to_db(spec, ref=np.max)
        return spec

    def reconstruct_waveform(self):
        spec = self.spectrogram_transform(self.waveform)
        return self.griffin_lim(spec)

    def get_mel_filters(self):
        return F.melscale_fbanks(
            int(self.n_fft // 2 + 1),
            n_mels=self.n_mels,
            f_min=0.0,
            f_max=self.sr / 2.0,
            sample_rate=self.sr,
            norm="slaney",
        )

    def get_melspectrogram(self):
        melspec = self.mel_spectrogram_transform(self.waveform)[0].numpy()
        melspec = librosa.power_to_db(melspec, ref=np.max)
        # librosa.power_to_db()
        return melspec

    def normalize(self, spec):
        spec_min, spec_max = spec.min(), spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-6)
        return spec_norm

    def to_grayscale(self, spec):
        return np.expand_dims(spec, axis=0)  # shape: (1, H, W)

    def get_normalized_melspec(self):
        melspec = self.get_melspectrogram()
        melspec_norm = self.normalize(melspec)
        return self.to_grayscale(melspec_norm)

    # ---------------- PLOT FUNCTIONS ----------------
    def plot_waveforms(self):
        reconstructed = self.reconstruct_waveform()
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title("Original Waveform")
        plt.plot(self.waveform.t().numpy())
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.title("Reconstructed Waveform (Griffin-Lim)")
        plt.plot(reconstructed.t().detach().numpy())
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    def plot_spectrogram(self):
        spec = self.get_spectrogram()
        plt.figure(figsize=(10, 4))
        plt.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
        plt.title("Spectrogram (dB)")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format="%+2.0f dB")
        plt.show()

    def plot_filters(self):
        mel_filters = self.get_mel_filters()
        plt.figure(figsize=(10, 6))
        plt.imshow(mel_filters.numpy(), origin="lower", aspect="auto", cmap="viridis")
        plt.title("Mel Filter Banks")
        plt.xlabel("FFT bins")
        plt.ylabel("Mel filter index")
        plt.colorbar(label="Amplitude")
        plt.show()

    def plot_melspectrogram(self):
        melspec = self.get_melspectrogram()
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(melspec, sr=self.sr, hop_length=self.hop_length)
        plt.axis("off")  # ปิดแกน x/y
        plt.tight_layout()
        plt.show()

    def save_melspectrogram(self, out_path="melspec.png"):
        melspec = self.get_melspectrogram()
        plt.figure(figsize=(10, 4))
        import librosa.display

        librosa.display.specshow(melspec, sr=self.sr, hop_length=self.hop_length)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        return out_path
