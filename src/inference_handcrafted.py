import joblib
import numpy as np
from src.features_handcrafted import extract_features
class HandcraftedPredictor:
    def __init__(self, model_path="models/best_model.pkl", scaler_path="models/scaler.pkl"):
        """
        Khởi tạo, load sẵn mô hình cổ điển (SVM/RF/XGB) và Standard Scaler đã train bằng GroupKFold.
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_audio(self, audio_path):
        """
        Hàm inference độc lập: Nhận file .wav -> Trích xuất -> Scale -> Trả về kết quả (Real/Fake).
        """
        try:
            # Trích xuất đặc trưng thủ công từ file âm thanh thô
            features = extract_features(audio_path)
            
            # mảng 2D và chuẩn hóa bằng mean/std lúc train
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Dự đoán nhãn và lấy xác suất
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            return {
                "status": "success",
                "prediction": str(prediction),
                "probability": probability.tolist()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }