"""Hàm dùng chung: nạp metadata + đặc trưng, ghép THEO clip_id, kiểm tra nghiêm ngặt."""
import numpy as np
import pandas as pd

REQUIRED = ["clip_id", "label", "speaker_id", "split"]
LABEL_MAP = {"real": 0, "fake": 1}


def load_split(metadata_path, feature_path, splits):
    meta = pd.read_csv(metadata_path)
    miss = [c for c in REQUIRED if c not in meta.columns]
    if miss:
        raise ValueError(f"metadata thiếu cột {miss}. Chạy src/prepare_metadata.py trước.")
    meta["clip_id"] = meta["clip_id"].astype(str)
    if meta["clip_id"].duplicated().any():
        raise ValueError("clip_id trùng trong metadata")
    bad = set(meta["label"]) - set(LABEL_MAP)
    if bad:
        raise ValueError(f"label không hợp lệ: {bad} (chỉ nhận real/fake)")

    npz = np.load(feature_path, allow_pickle=True)
    ids = [str(c) for c in npz["clip_id"]]
    if len(set(ids)) != len(ids):
        raise ValueError("clip_id trùng trong file đặc trưng")
    X_all = npz["X"]
    if not np.all(np.isfinite(X_all)):
        raise ValueError("Đặc trưng chứa NaN/inf")
    names = [str(n) for n in npz["feature_names"]] if "feature_names" in npz.files \
        else [f"dim_{i}" for i in range(X_all.shape[1])]
    pos = {c: i for i, c in enumerate(ids)}

    sub = meta[meta["split"].isin(splits)].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"Không có dòng nào thuộc split {splits}")
    missing = [c for c in sub["clip_id"] if c not in pos]
    if missing:
        raise ValueError(f"{len(missing)} clip_id có trong metadata nhưng thiếu đặc trưng, ví dụ: {missing[:3]}")
    X = X_all[[pos[c] for c in sub["clip_id"]]]            # ghép theo ID, không theo thứ tự hàng
    y = sub["label"].map(LABEL_MAP).values
    return sub, X, y, sub["speaker_id"].astype(str).values, names