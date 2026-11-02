"""Gộp metadata_real.csv + metadata_fake.csv thành data/metadata_combined.csv (đúng 9 cột).

Chạy:  python src/prepare_metadata.py [--audio-dir "data/WAV đã chuẩn hóa"]
Không tự bịa nhãn, người nói hay split: thiếu hoặc sai là DỪNG và báo lỗi.
"""
import argparse, os, sys
import pandas as pd

COLS = ["clip_id", "path", "label", "speaker_id", "source", "generator", "channel", "split", "duration"]
def base(p):
    return str(p).replace("\\", "/").split("/")[-1]


SPLITS = {"train", "dev", "test_seen", "test_unseen", "test_phone"}
ALIASES = {"path": ["path", "file_name", "filename"],
           "speaker_id": ["speaker_id", "speaker"],
           "generator": ["generator", "tool"],
           "clip_id": ["clip_id"], "label": ["label"], "split": ["split"],
           "source": ["source"], "channel": ["channel"], "duration": ["duration"]}


def pick(df, name):
    for c in ALIASES[name]:
        if c in df.columns:
            return df[c]
    return None


def normalize(df, expected_label, audio_dir, tag):
    out = pd.DataFrame(index=df.index)
    for name in ["path", "speaker_id", "split", "label"]:
        s = pick(df, name)
        if s is None:
            raise ValueError(f"[{tag}] thiếu cột bắt buộc '{name}' (chấp nhận: {ALIASES[name]})")
        out[name] = s.astype(str).str.strip()
    out["label"] = out["label"].str.lower().replace({"bona fide": "real", "bonafide": "real", "spoof": "fake"})
    bad = set(out["label"]) - {"real", "fake"}
    if bad:
        raise ValueError(f"[{tag}] nhãn không hợp lệ: {bad}")
    if set(out["label"]) != {expected_label}:
        raise ValueError(f"[{tag}] file này phải chỉ chứa nhãn '{expected_label}', thực tế: {set(out['label'])}")
    gen = pick(df, "generator")
    out["generator"] = "none" if expected_label == "real" else (gen.astype(str) if gen is not None else None)
    if out["generator"].isna().any():
        raise ValueError(f"[{tag}] file giọng giả phải có cột 'tool' hoặc 'generator'")
    for name, default in [("source", tag), ("channel", "clean")]:
        s = pick(df, name)
        out[name] = s.astype(str) if s is not None else default
    d = pick(df, "duration")
    out["duration"] = d if d is not None else float("nan")

    # Tìm file thật: đường dẫn trong metadata trước, rồi mới tới --audio-dir (cấm trùng basename)
    resolved, missing = [], []
    index = {}
    if audio_dir:
        for root, _, files in os.walk(audio_dir):
            for f in files:
                index.setdefault(f, []).append(os.path.join(root, f))
    for p in out["path"]:
        if os.path.exists(p):
            resolved.append(p)
        elif base(p) in index:
            cands = index[base(p)]
            if len(cands) > 1:
                raise ValueError(f"[{tag}] basename trùng giữa các thư mục: {p} -> {cands[:3]}")
            resolved.append(cands[0])
        else:
            resolved.append(None); missing.append(p)
    if missing:
        raise FileNotFoundError(f"[{tag}] thiếu {len(missing)} file audio, ví dụ: {missing[:3]}")
    out["path"] = [r.replace("\\", "/") for r in resolved]

    cid = pick(df, "clip_id")
    out["clip_id"] = (cid.astype(str) if cid is not None else
                      out["label"] + "_" + out["generator"] + "_" + out["path"].map(lambda p: os.path.splitext(base(p))[0]))
    return out[COLS]


def subsample_real(df, n, seed=42):
    """Lấy tối đa n clip real mỗi người nói (cố định seed). Không đổi nhãn/người nói/split."""
    real = df[df["label"] == "real"]
    keep = real.groupby("speaker_id", group_keys=False).apply(lambda g: g.sample(min(len(g), n), random_state=seed))
    return pd.concat([keep, df[df["label"] == "fake"]], ignore_index=True)


def validate(df):
    if df["clip_id"].duplicated().any():
        raise ValueError(f"clip_id trùng: {df.loc[df['clip_id'].duplicated(), 'clip_id'].head().tolist()}")
    bad = set(df["split"]) - SPLITS
    if bad:
        raise ValueError(f"split không hợp lệ: {bad}")
    # Một người nói chỉ nằm trong một split (xem mục 6 tài liệu đề tài)
    n = df.groupby("speaker_id")["split"].nunique()
    if (n > 1).any():
        raise ValueError(f"Người nói nằm ở nhiều split (rò rỉ): {n[n > 1].index.tolist()[:10]}")
    tr = df[df["split"] == "train"]
    n_fake_spk = tr.loc[tr["label"] == "fake", "speaker_id"].nunique()
    if n_fake_spk < 2:
        raise ValueError(
            f"Giọng giả trong train chỉ có {n_fake_spk} speaker_id ({tr.loc[tr['label']=='fake','speaker_id'].unique()[:3].tolist()}). "
            "GroupKFold theo người nói cần >= 2 người nói có giọng giả. speaker_id của giọng giả phải là NGƯỜI BỊ CLONE "
            "(người nói VIVOS làm giọng mẫu), không phải tên giọng của công cụ TTS. Nhờ Task 2 gán lại, hoặc dùng bộ tạm (xem hướng dẫn).")
    for sp in ["train", "test_seen"]:
        labels = set(df.loc[df["split"] == sp, "label"])
        if labels != {"real", "fake"}:
            raise ValueError(f"split '{sp}' phải có đủ real và fake, hiện có: {labels}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="data/metadata_real.csv")
    ap.add_argument("--fake", default="data/metadata_fake.csv")
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--max-real-per-speaker", type=int, default=None)
    ap.add_argument("--out", default="data/metadata_combined.csv")
    a = ap.parse_args()
    real = normalize(pd.read_csv(a.real), "real", a.audio_dir, "real")
    fake = normalize(pd.read_csv(a.fake), "fake", a.audio_dir, "fake")
    df = pd.concat([real, fake], ignore_index=True)
    if a.max_real_per_speaker:
        df = subsample_real(df, a.max_real_per_speaker)
    validate(df)
    df.to_csv(a.out, index=False)
    print(f"[OK] {a.out}: {len(df)} clip")
    print(df.groupby(["split", "label"]).agg(clips=("clip_id", "count"), speakers=("speaker_id", "nunique")))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"[LỖI] {e}")