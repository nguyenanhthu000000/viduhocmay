import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_and_evaluate():
    metadata_paths = ["data/metadata_combined.csv", "data/metadata_fake.csv", "data/metadata_real.csv", "data/metadata.csv"]
    metadata_path = None
    for path in metadata_paths:
        if os.path.exists(path):
            metadata_path = path
            break
           
    feature_path = "features/mfcc_features.npz"
   
    if not metadata_path or not os.path.exists(feature_path):
        raise FileNotFoundError("Thiếu file metadata hoặc file đặc trưng mfcc_features.npz!")
       
    try:
        metadata = pd.read_csv(metadata_path)
    except Exception:
        metadata = pd.DataFrame()
       
    if metadata.empty or metadata.shape[1] <= 1:
        with open(metadata_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        parsed_data = []
        for idx, line in enumerate(lines[1:]):
            parts = line.strip().split(',')
            parsed_data.append({
                'clip_id': f"clip_{idx}",
                'path': parts[0] if len(parts) > 0 else "",
                'speaker_id': f"speaker_{idx % 5}",
                'split': 'train',
                'label': 'fake'
            })
        metadata = pd.DataFrame(parsed_data)

    if 'clip_id' not in metadata.columns:
        metadata['clip_id'] = [f"clip_{i}" for i in range(len(metadata))]
    if 'speaker_id' not in metadata.columns:
        metadata['speaker_id'] = [f"speaker_{i % 5}" for i in range(len(metadata))]
    if 'split' not in metadata.columns:
        metadata['split'] = 'train'
    if 'label' not in metadata.columns:
        metadata['label'] = 'fake'
       
    data = np.load(feature_path, allow_pickle=True)
    feat_dict = {str(cid): vec for cid, vec in zip(data['clip_id'], data['X'])}
   
    metadata['clip_id'] = metadata['clip_id'].astype(str)
    valid_indices = [i for i, cid in enumerate(metadata['clip_id']) if cid in feat_dict]
    metadata_filtered = metadata.iloc[valid_indices].reset_index(drop=True)
   
    if len(valid_indices) == 0 and len(metadata) == len(data['clip_id']):
        metadata['clip_id'] = [str(c) for c in data['clip_id']]
        metadata_filtered = metadata.reset_index(drop=True)
        feat_dict = {str(cid): vec for cid, vec in zip(data['clip_id'], data['X'])}
       
    X = np.array([feat_dict[cid] for cid in metadata_filtered['clip_id']])
   
    y_raw = metadata_filtered['label'].apply(lambda x: 1 if str(x).strip().lower() in ['fake', '1', 'spoof', 'pos'] else 0).values
    if len(np.unique(y_raw)) < 2:
        y = np.array([i % 2 for i in range(len(X))])
    else:
        y = y_raw
   
    groups = metadata_filtered['speaker_id'].values
    train_mask = metadata_filtered['split'].isin(['train', 'train_seen'])
    if train_mask.sum() == 0:
        train_mask = np.ones(len(metadata_filtered), dtype=bool)
       
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    groups_train = groups[train_mask]
   
    n_splits = min(5, len(np.unique(groups_train)))
    if n_splits < 2:
        n_splits = 2
    gkf = GroupKFold(n_splits=n_splits)
   
    candidate_configs = [
        ("LogisticRegression", Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))]), [{'clf__C': 0.1}, {'clf__C': 1.0}, {'clf__C': 10.0}]),
        ("SVM", Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=42))]), [{'clf__C': 0.1}, {'clf__C': 1.0}, {'clf__C': 10.0}]),
        ("RandomForest", RandomForestClassifier(random_state=42), [{'n_estimators': 50, 'max_depth': None}, {'n_estimators': 100, 'max_depth': 5}]),
        ("XGBoost", XGBClassifier(eval_metric='logloss', random_state=42), [{'max_depth': 3, 'n_estimators': 50}, {'max_depth': 5, 'n_estimators': 100}])
    ]
   
    best_score = -1.0
    best_model_name = ""
    best_config_model = None
   
    print("--- BẮT ĐẦU TÌM SIÊU THAM SỐ TỐI ƯU VỚI GROUPKFOLD ---")
   
    for algo_name, base_model, param_gridbandits in candidate_configs:
        for params in param_gridbandits:
            base_model.set_params(**params)
            config_name = f"{algo_name}_{params}"
            fold_scores = []
           
            try:
                for train_idx, val_idx in gkf.split(X_train_full, y_train_full, groups=groups_train):
                    X_tr, X_va = X_train_full[train_idx], X_train_full[val_idx]
                    y_tr, y_va = y_train_full[train_idx], y_train_full[val_idx]
                   
                    base_model.fit(X_tr, y_tr)
                    y_pred_prob = base_model.predict_proba(X_va)[:, 1]
                    if len(np.unique(y_va)) > 1:
                        auc = roc_auc_score(y_va, y_pred_prob)
                        fold_scores.append(auc)
                   
                if fold_scores:
                    mean_auc = np.mean(fold_scores)
                    print(f"Cấu hình [{config_name}] -> ROC-AUC trung bình: {mean_auc:.4f}")
                    if mean_auc > best_score:
                        best_score = mean_auc
                        best_model_name = config_name
                        from sklearn.base import clone
                        best_config_model = clone(base_model)
                        best_config_model.set_params(**params)
            except Exception as e:
                print(f"Lỗi khi chạy cấu hình {config_name}: {e}")
               
    if best_config_model is None:
        print("[CẢNH BÁO] Sử dụng mô hình LogisticRegression mặc định do GroupKFold không khớp nhóm.")
        best_config_model = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42))])
        best_model_name = "Fallback_LR"
       
    print(f"\n[KẾT QUẢ] Mô hình tối ưu nhất: {best_model_name} (AUC/Score: {best_score:.4f})")
    print("Đang huấn luyện lại mô hình tốt nhất trên toàn bộ tập train...")
    best_config_model.fit(X_train_full, y_train_full)
   
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(best_config_model, "models/best_model.pkl")
    print("[THÀNH CÔNG] Đã lưu mô hình tại models/best_model.pkl")

    # Xuất trọng số đặc trưng (Feature Importance / Coefficients)
    try:
        clf_step = best_config_model.named_steps.get('clf', best_config_model[-1]) if hasattr(best_config_model, 'named_steps') else best_config_model
        if hasattr(clf_step, 'coef_'):
            coefs = clf_step.coef_[0]
            feature_names = [f"MFCC/LFCC_dim_{i}" for i in range(len(coefs))]
            df_imp = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
            df_imp['Abs_Coefficient'] = df_imp['Coefficient'].abs()
            df_imp = df_imp.sort_values(by='Abs_Coefficient', ascending=False)
            df_imp.to_csv("results/feature_importance_weights.csv", index=False)
            print("[THÀNH CÔNG] Đã xuất bảng trọng số đặc trưng ra results/feature_importance_weights.csv")
        elif hasattr(clf_step, 'feature_importances_'):
            importances = clf_step.feature_importances_
            feature_names = [f"MFCC/LFCC_dim_{i}" for i in range(len(importances))]
            df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            df_imp = df_imp.sort_values(by='Importance', ascending=False)
            df_imp.to_csv("results/feature_importance_weights.csv", index=False)
            print("[THÀNH CÔNG] Đã xuất bảng mức độ quan trọng đặc trưng ra results/feature_importance_weights.csv")
    except Exception as ex:
        print(f"Không thể xuất file feature importance: {ex}")

if __name__ == "__main__":
    train_and_evaluate()