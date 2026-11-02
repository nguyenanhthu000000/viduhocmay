"""Đánh giá mô hình đã lưu trên split test_seen bằng NHÃN GỐC.
Xuất: ROC, confusion matrix, bảng chỉ số, bảng dự đoán vào results/.
Chạy:  python src/evaluate.py [--split test_seen]
"""
import argparse, os, sys
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
from common import load_split


def compute_eer(y, score):
    fpr, tpr, thr = roc_curve(y, score)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i]), fpr, tpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata_combined.csv")
    ap.add_argument("--features", default="features/mfcc_features.npz")
    ap.add_argument("--model", default="models/best_model.pkl")
    ap.add_argument("--split", default="test_seen")
    a = ap.parse_args()

    meta, X, y, groups, _ = load_split(a.metadata, a.features, [a.split])
    if len(set(y)) < 2:
        sys.exit(f"[LỖI] Split '{a.split}' chỉ có một lớp -> AUC/EER không xác định. Không tự tạo nhãn.")
    model = joblib.load(a.model)
    p = model.predict_proba(X)[:, list(model.classes_).index(1)]
    pred = (p >= 0.5).astype(int)

    auc = roc_auc_score(y, p); f1 = f1_score(y, pred); eer, eer_thr, fpr, tpr = compute_eer(y, p)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{"split": a.split, "n_samples": len(y), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
                   "n_speakers": len(set(groups)), "ROC_AUC": auc, "EER": eer, "EER_threshold": eer_thr,
                   "F1": f1, "F1_threshold": 0.5}]).to_csv("results/evaluation_metrics_summary.csv", index=False)
    pd.DataFrame({"clip_id": meta["clip_id"], "speaker_id": groups, "true_label": y,
                  "pred_prob_fake": p, "pred_label": pred}).to_csv(f"results/{a.split}_predictions.csv", index=False)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Dự đoán"); plt.ylabel("Thực tế"); plt.title(f"Confusion matrix ({a.split})")
    plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=200); plt.close()
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}, EER={eer:.3f}"); plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({a.split})"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig("results/roc_curve.png", dpi=200); plt.close()
    print(f"[{a.split}] n={len(y)} AUC={auc:.4f} EER={eer:.4f} F1={f1:.4f}\n{cm}")


if __name__ == "__main__":
    main()