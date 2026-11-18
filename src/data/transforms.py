"""Görüntü (224x224 Bicubic, ImageNet Norm, Augmentation) ve
Ses (16 kHz Mono, RMS Norm, 128-bin Log-Mel Spektrogram, SpecAugment) ön işleme modülü.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy.signal import stft

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def build_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 128,
    f_min: float = 50.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Slaney/HTK tarzı üçgensel Mel filtre bankası matrisi üretir: şekil [n_mels, n_fft // 2 + 1]."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        up = (fft_freqs - f_left) / max(f_center - f_left, 1e-8)
        down = (f_right - fft_freqs) / max(f_right - f_center, 1e-8)
        fb[m - 1] = np.maximum(0.0, np.minimum(up, down))
        # Alan normalizasyonu
        enorm = 2.0 / max(f_right - f_left, 1e-8)
        fb[m - 1] *= enorm
    return fb


_MEL_FB_CACHE: Dict[Tuple[int, int, int, float, float], np.ndarray] = {}


def get_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 128,
    f_min: float = 50.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    key = (sr, n_fft, n_mels, f_min, f_max)
    if key not in _MEL_FB_CACHE:
        _MEL_FB_CACHE[key] = build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max)
    return _MEL_FB_CACHE[key]


class BirdImagePreprocessor:
    """Görüntüleri 224x224 boyutuna getirir, isteğe bağlı artırma uygular ve ImageNet normalize eder."""

    def __init__(self, target_size: int = 224, augment: bool = False, seed: int = 42) -> None:
        self.target_size = target_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __call__(self, img_path: Path | str) -> torch.Tensor:
        with Image.open(img_path) as im:
            im = im.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)
            if self.augment:
                if self.rng.random() < 0.5:
                    im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                im = ImageEnhance.Brightness(im).enhance(float(self.rng.uniform(0.88, 1.12)))
                im = ImageEnhance.Contrast(im).enhance(float(self.rng.uniform(0.88, 1.12)))
                im = ImageEnhance.Color(im).enhance(float(self.rng.uniform(0.92, 1.08)))
            arr = np.asarray(im, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        norm = (chw - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(norm.astype(np.float32))


def load_wav_mono_fixed(wav_path: Path | str, target_sr: int = 16000, target_len: int = 48000) -> Tuple[np.ndarray, int]:
    """WAV dosyasını okur, mono kanala indirger ve 3.0 sn (48.000 örnek) sabit uzunluğa hizalar."""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr and len(data) > 0:
        new_len = int(round(len(data) * target_sr / sr))
        old_x = np.linspace(0.0, 1.0, len(data), endpoint=False)
        new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
        data = np.interp(new_x, old_x, data).astype(np.float32)

    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)), mode="constant")
    elif len(data) > target_len:
        start = (len(data) - target_len) // 2
        data = data[start:start + target_len]

    return data.astype(np.float32), target_sr


class BirdAudioPreprocessor:
    """16 kHz mono sesten [1, 128, 151] boyutunda Log-Mel spektrogram tensörü üretir."""

    def __init__(
        self,
        sample_rate: int = 16000,
        duration_sec: float = 3.0,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        augment: bool = False,
        time_mask_param: int = 18,
        freq_mask_param: int = 16,
        seed: int = 42,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_len = int(round(sample_rate * duration_sec))
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.augment = augment
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.rng = np.random.default_rng(seed)
        self.mel_fb = get_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)

    def compute_stft_and_logmel(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hem doğrusal STFT güç spektrogramını hem 128-bin Log-Mel spektrogramını döndürür."""
        # RMS enerji normalizasyonu
        rms = float(np.sqrt(np.mean(waveform ** 2)) + 1e-8)
        norm_wav = waveform * (0.1 / rms) if rms > 1e-5 else waveform

        if self.augment:
            gain = float(self.rng.uniform(0.85, 1.15))
            norm_wav = norm_wav * gain + self.rng.normal(0, 0.003, size=norm_wav.shape).astype(np.float32)

        pad = self.n_fft // 2
        padded = np.pad(norm_wav, (pad, pad), mode="reflect")
        _, _, Zxx = stft(
            padded,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
            boundary=None,
            padded=False,
        )
        power_spec = (np.abs(Zxx) ** 2).astype(np.float32)  # [513, 151]
        mel_spec = self.mel_fb @ power_spec  # [128, 151]
        log_mel_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))

        # Örnek bazlı standardizasyon (mean=0, std=1)
        log_mel_norm = (log_mel_db - log_mel_db.mean()) / (log_mel_db.std() + 1e-6)

        if self.augment:
            # SpecAugment: Frekans ve Zaman maskelemesi
            f_len = int(self.rng.integers(4, self.freq_mask_param + 1))
            f0 = int(self.rng.integers(0, max(1, self.n_mels - f_len)))
            log_mel_norm[f0:f0 + f_len, :] = 0.0

            t_frames = log_mel_norm.shape[1]
            t_len = int(self.rng.integers(4, self.time_mask_param + 1))
            t0 = int(self.rng.integers(0, max(1, t_frames - t_len)))
            log_mel_norm[:, t0:t0 + t_len] = 0.0

        return power_spec, log_mel_norm.astype(np.float32)

    def __call__(self, wav_path: Path | str) -> torch.Tensor:
        waveform, _ = load_wav_mono_fixed(wav_path, self.sample_rate, self.target_len)
        _, log_mel = self.compute_stft_and_logmel(waveform)
        return torch.from_numpy(log_mel[None, :, :])  # [1, 128, 151]


def extract_image_eda_metrics(img_path: Path | str) -> Dict[str, float]:
    """Tek bir görüntü dosyasından parlaklık, RMS kontrast, RGB ortalaması ve entropi ölçer."""
    with Image.open(img_path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    hist, _ = np.histogram(lum.ravel(), bins=256, range=(0, 256))
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    return {
        "luminance": float(lum.mean()),
        "rms_contrast": float(lum.std()),
        "color_entropy": entropy,
        "mean_r": float(arr[:, :, 0].mean()),
        "mean_g": float(arr[:, :, 1].mean()),
        "mean_b": float(arr[:, :, 2].mean()),
    }


def extract_audio_eda_metrics(wav_path: Path | str) -> Dict[str, float]:
    """Tek bir WAV dosyasından Spectral Centroid, Bandwidth, RMS dB, ZCR ve sessizlik oranı ölçer."""
    wav, sr = load_wav_mono_fixed(wav_path)
    rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-10)
    rms_db = float(20.0 * np.log10(rms))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(wav).astype(np.float32)))))
    # Kısa çerçeve enerjisi ile sessizlik oranı (tepe enerjinin %10'unun altı)
    frames = wav.reshape(-1, 400)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    thresh = max(0.015, 0.12 * float(frame_rms.max()))
    silence_ratio = float(np.mean(frame_rms < thresh))

    freqs = np.fft.rfftfreq(len(wav), d=1.0 / sr)
    mag = np.abs(np.fft.rfft(wav))
    mag_sum = float(mag.sum() + 1e-10)
    centroid = float(np.sum(freqs * mag) / mag_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / mag_sum))
    return {
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "rms_energy_db": rms_db,
        "zcr": zcr,
        "silence_ratio": silence_ratio,
    }
