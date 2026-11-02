"""Trích MFCC + LFCC (13 hệ số, mean/std theo thời gian) -> features/mfcc_features.npz

Vector 52 chiều: mfcc_mean(13) | mfcc_std(13) | lfcc_mean(13) | lfcc_std(13)
Chạy:  python src/features_handcrafted.py [--metadata data/metadata_combined.csv]
"""
import argparse, sys
import numpy as np
import pandas as pd
import librosa
from scipy.fftpack import dct

SR, N_COEF, N_FFT, HOP = 16000, 13, 2048, 512
FEATURE_NAMES = [f"{kind}_{stat}_{i}" for kind in ("mfcc", "lfcc")
                 for stat in ("mean", "std") for i in range(N_COEF)]


def extract_lfcc(y, sr, n_lfcc=N_COEF, n_fft=N_FFT, hop=HOP):
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    edges = np.linspace(0, sr / 2.0, n_lfcc + 2)            # filterbank tam giác, cách đều theo Hz
    fb = np.zeros((n_lfcc, len(freqs)))
    for i in range(n_lfcc):
        lo, ce, up = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.maximum(0, np.minimum((freqs - lo) / (ce - lo), (up - freqs) / (up - ce)))
    log_e = np.log(np.maximum(fb @ power, 1e-10))
    return dct(log_e, axis=0, type=2, norm="ortho")[:n_lfcc]


def extract_vector(audio_path):
    """Vector 52 chiều cho MỘT file. Dùng chung cho train và inference (demo)."""
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    if len(y) < N_FFT:
        raise ValueError(f"Audio quá ngắn ({len(y)} mẫu): {audio_path}")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_COEF)
    lfcc = extract_lfcc(y, sr)
    vec = np.concatenate([mfcc.mean(1), mfcc.std(1), lfcc.mean(1), lfcc.std(1)])
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"Đặc trưng không hữu hạn: {audio_path}")
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata_combined.csv")
    ap.add_argument("--out", default="features/mfcc_features.npz")
    a = ap.parse_args()
    df = pd.read_csv(a.metadata)
    for c in ("clip_id", "path"):
        if c not in df.columns:
            sys.exit(f"[LỖI] metadata thiếu cột {c}. Chạy src/prepare_metadata.py trước.")
    ids, rows, errors = [], [], []
    for cid, p in zip(df["clip_id"].astype(str), df["path"]):
        try:
            rows.append(extract_vector(p)); ids.append(cid)
        except Exception as e:
            errors.append((p, str(e)))
    if errors:                                              # không nuốt lỗi trong im lặng
        sys.exit(f"[LỖI] {len(errors)}/{len(df)} clip lỗi, ví dụ: {errors[:3]}")
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, clip_id=np.array(ids), X=np.array(rows), feature_names=np.array(FEATURE_NAMES))
    print(f"[OK] {a.out}: X={np.array(rows).shape}")


if __name__ == "__main__":
    main()