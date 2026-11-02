"""Wrapper cho demo (Task 5): HandcraftedPredictor(model).predict_audio(wav) -> dict.
Pipeline đã chứa StandardScaler nên KHÔNG cần scaler.pkl và không scale lần hai."""
import joblib
try:
    from src.features_handcrafted import extract_vector
except ImportError:
    from features_handcrafted import extract_vector


class HandcraftedPredictor:
    def __init__(self, model_path="models/best_model.pkl"):
        self.model = joblib.load(model_path)
        self.fake_idx = list(self.model.classes_).index(1)   # 1 = fake

    def predict_fake_proba(self, audio_path):
        vec = extract_vector(audio_path).reshape(1, -1)
        return float(self.model.predict_proba(vec)[0, self.fake_idx])

    def predict_audio(self, audio_path):
        try:
            p = self.predict_fake_proba(audio_path)
            return {"status": "success", "prediction": "fake" if p >= 0.5 else "real", "prob_fake": p}
        except Exception as e:
            return {"status": "error", "message": str(e)}