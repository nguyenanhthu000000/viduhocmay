import os
import numpy as np
import pandas as pd
import librosa
import librosa.filters
from scipy.fftpack import dct

def extract_lfcc(y, sr, n_lfcc=13, n_fft=2048, hop_length=512):
    """Trích xuất LFCC (Linear Frequency Cepstral Coefficients)"""
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    lin_filterbank = np.linspace(0, sr / 2.0, n_lfcc + 2)
    
    filter_bank = np.zeros((n_lfcc, len(fft_freqs)))
    for i in range(n_lfcc):
        lower = lin_filterbank[i]
        center = lin_filterbank[i+1]
        upper = lin_filterbank[i+2]
        filter_bank[i] = np.maximum(
            0,
            np.minimum(
                (fft_freqs - lower) / (center - lower + 1e-10),
                (upper - fft_freqs) / (upper - center + 1e-10)
            )
        )
    
    lfcc_feat = np.dot(filter_bank, stft)
    lfcc_feat = np.log(np.maximum(lfcc_feat, 1e-10))
    lfcc_cep = dct(lfcc_feat, axis=0, type=2, norm='ortho')[:n_lfcc]
    return lfcc_cep

def extract_features():
    metadata_paths = ["data/metadata_combined.csv", "data/metadata_real.csv", "data/metadata_fake.csv"]
    metadata_path = None
    for path in metadata_paths:
        if os.path.exists(path):
            metadata_path = path
            break
            
    if not metadata_path:
        raise FileNotFoundError("Không tìm thấy file metadata hợp lệ trong thư mục data/!")
        
    print(f"Đang sử dụng file metadata: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Xác định cột chứa đường dẫn file âm thanh tùy theo schema của file real hay fake
    path_col = None
    for col in ['path', 'file_name', 'filename']:
        if col in df.columns:
            path_col = col
            break
    if not path_col:
        path_col = df.columns[0]
            
    os.makedirs("features", exist_ok=True)
    clip_ids = []
    features_list = []
    missing_count = 0
    
    for idx, row in df.iterrows():
        raw_path = str(row[path_col])
        file_name_only = os.path.basename(raw_path)
        
        audio_path = os.path.join("data", "WAV đã chuẩn hóa", file_name_only)
        if not os.path.exists(audio_path):
            if os.path.exists(raw_path):
                audio_path = raw_path
            else:
                missing_count += 1
                continue
            
        # Lấy clip_id chuẩn nếu có, nếu không tự định nghĩa từ index
        clip_id = str(row['clip_id']) if 'clip_id' in df.columns else f"clip_{idx}"
        
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            lfcc = extract_lfcc(y, sr, n_lfcc=13)
            lfcc_mean = np.mean(lfcc, axis=1)
            lfcc_std = np.std(lfcc, axis=1)
            
            feat_vector = np.concatenate([mfcc_mean, mfcc_std, lfcc_mean, lfcc_std])
            
            clip_ids.append(clip_id)
            features_list.append(feat_vector)
        except Exception as e:
            print(f"[Lỗi] Xử lý file {audio_path} thất bại: {e}")
            
    if missing_count > 0:
        print(f"[CẢNH BÁO] Có {missing_count} file âm thanh không tìm thấy.")
    if len(features_list) == 0:
        raise ValueError("Không tìm thấy file âm thanh hợp lệ nào để trích xuất đặc trưng!")
            
    X = np.array(features_list)
    clip_ids = np.array(clip_ids)
    
    np.savez("features/mfcc_features.npz", clip_id=clip_ids, X=X)
    print(f"[THÀNH CÔNG] Đã trích xuất xong {len(clip_ids)} mẫu, shape X: {X.shape}")

if __name__ == "__main__":
    extract_features()