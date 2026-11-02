"""Huấn luyện LR / SVM / RF / XGBoost, chọn tham số bằng GroupKFold theo speaker_id (chỉ split=train).
Chạy:  python src/train_classical.py [--metadata ...] [--features ...]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common import load_split

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def candidates():
    scaled = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cfg = [
        ("LogisticRegression", scaled(LogisticRegression(max_iter=2000, random_state=42)),
         [{"clf__C": c} for c in (0.1, 1.0, 10.0)]),
        ("SVM", scaled(SVC(kernel="rbf", probability=True, random_state=42)),
         [{"clf__C": c, "clf__gamma": g} for c in (0.1, 1.0, 10.0) for g in ("scale", 0.01)]),
        ("RandomForest", RandomForestClassifier(random_state=42),
         [{"n_estimators": n, "max_depth": d} for n in (100, 300) for d in (None, 8)]),
    ]
    if XGBClassifier is not None:
        cfg.append(("XGBoost", XGBClassifier(eval_metric="logloss", random_state=42),
                    [{"n_estimators": n, "max_depth": d} for n in (100, 300) for d in (3, 5)]))
    else:
        print("[CẢNH BÁO] Chưa cài xgboost -> bỏ qua XGBoost (pip install xgboost)")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata_combined.csv")
    ap.add_argument("--features", default="features/mfcc_features.npz")
    a = ap.parse_args()

    meta, X, y, groups, names = load_split(a.metadata, a.features, ["train"])
    if len(set(y)) < 2:
        sys.exit("[LỖI] Tập train chỉ có một lớp. Cần cả real và fake, không tự đổi nhãn.")
    n_spk = len(set(groups))
    if n_spk < 2:
        sys.exit("[LỖI] Cần ít nhất 2 người nói trong train để chạy GroupKFold.")
    gkf = GroupKFold(n_splits=min(5, n_spk))
    print(f"Train: {len(y)} clip, {n_spk} người nói, real={int((y==0).sum())}, fake={int((y==1).sum())}")

    rows = []
    for algo, base, grid in candidates():
        for params in grid:
            scores = []
            for tr, va in gkf.split(X, y, groups):
                if len(set(y[va])) < 2 or len(set(y[tr])) < 2:
                    continue                                 # fold một lớp không tính được AUC
                m = clone(base).set_params(**params).fit(X[tr], y[tr])
                scores.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
            if not scores:
                continue
            rows.append({"algo": algo, "params": json.dumps(params), "cv_auc_mean": np.mean(scores),
                         "cv_auc_std": np.std(scores), "n_folds_used": len(scores)})
            print(f"{algo} {params} -> AUC {np.mean(scores):.4f} (±{np.std(scores):.4f}, {len(scores)} fold)")
    if not rows:
        sys.exit("[LỖI] Không fold nào có đủ hai lớp. Kiểm tra lại phân bố người nói / nhãn.")
    cv = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
    os.makedirs("results", exist_ok=True); os.makedirs("models", exist_ok=True)
    cv.to_csv("results/cv_results.csv", index=False)

    # Kết quả tốt nhất của TỪNG mô hình (để so sánh 4 mô hình)
    best_per_algo = cv.groupby("algo").head(1)
    best_per_algo.to_csv("results/cv_best_per_model.csv", index=False)
    top = cv.iloc[0]
    base = {n: b for n, b, _ in candidates()}[top["algo"]]
    final = clone(base).set_params(**json.loads(top["params"])).fit(X, y)   # refit trên TOÀN BỘ train
    joblib.dump(final, "models/best_model.pkl")
    with open("models/model_config.json", "w", encoding="utf-8") as f:
        json.dump({"algo": top["algo"], "params": json.loads(top["params"]), "cv_auc": float(top["cv_auc_mean"]),
                   "feature_names": names, "label_map": {"real": 0, "fake": 1},
                   "sklearn_version": sklearn.__version__, "n_train": int(len(y)), "n_speakers": n_spk}, f,
                  ensure_ascii=False, indent=2)
    print(f"[OK] Mô hình tốt nhất: {top['algo']} {top['params']} (CV AUC {top['cv_auc_mean']:.4f})")

    # Feature importance: hệ số (LR) / importance (RF, XGB) / permutation (SVM) -> luôn có kết quả
    clf = final.named_steps["clf"] if hasattr(final, "named_steps") else final
    if hasattr(clf, "coef_"):
        imp = pd.DataFrame({"feature": names, "value": clf.coef_[0]}); how = "coefficient"
    elif hasattr(clf, "feature_importances_"):
        imp = pd.DataFrame({"feature": names, "value": clf.feature_importances_}); how = "impurity_importance"
    else:
        try:
            _, Xe, ye, _, _ = load_split(a.metadata, a.features, ["dev"])
            dev_ok = len(set(ye)) == 2
        except ValueError:
            dev_ok = False
        if not dev_ok:
            Xe, ye = X, y                                    # không có dev hợp lệ: dùng train (ghi rõ trong cột method)
        pi = permutation_importance(final, Xe, ye, scoring="roc_auc", n_repeats=10, random_state=42)
        imp = pd.DataFrame({"feature": names, "value": pi.importances_mean}); how = "permutation_" + ("dev" if dev_ok else "train")
    imp["type"] = imp["feature"].str.split("_").str[0]       # mfcc / lfcc
    imp["stat"] = imp["feature"].str.split("_").str[1]       # mean / std
    imp["method"] = how
    imp["abs_value"] = imp["value"].abs()
    imp.sort_values("abs_value", ascending=False).drop(columns="abs_value").to_csv(
        "results/feature_importance_weights.csv", index=False)
    print("[OK] results/feature_importance_weights.csv, cv_results.csv, models/best_model.pkl")


if __name__ == "__main__":
    main()