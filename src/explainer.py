import shap
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Không cần GUI

class SHAPExplainer:
    """
    Wrap model IDS v14 để giải thích từng alert bằng SHAP.
    Dùng KernelSHAP vì model là PyTorch (model-agnostic).
    """

    def __init__(self, model, scaler, feature_names, background_data, device='cpu'):
        """
        model          : IDSBackbone đã train (eval mode)
        scaler         : RobustScaler đã fit từ v14
        feature_names  : list 55 tên feature của UNSW-NB15
        background_data: numpy array ~100-200 samples (normal traffic) để SHAP tính baseline
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.device = device

        # Wrap model thành function nhận numpy → trả numpy probability
        def model_predict(X_numpy):
            X_scaled = self.scaler.transform(X_numpy)
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            with torch.no_grad():
                logits = self.model(X_tensor)          # shape: (N, n_classes)
                probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

        self.predict_fn = model_predict

        # KernelExplainer cần background = mẫu "bình thường"
        # Dùng kmeans để giảm xuống 50 mẫu đại diện (nhanh hơn)
        background_scaled = scaler.transform(background_data)
        self.explainer = shap.KernelExplainer(
            self.predict_fn,
            shap.kmeans(background_scaled, 50)
        )

    def explain_alert(self, alert_features_raw, top_k=10):
        """
        Giải thích 1 alert cụ thể.

        alert_features_raw : numpy array shape (1, 55) — raw, chưa scale
        top_k              : số feature quan trọng nhất muốn hiển thị

        Return dict chứa:
            - shap_values     : array SHAP values
            - top_features    : list (feature_name, shap_value) top_k features
            - summary_text    : string mô tả ngắn để gửi vào LLM
        """
        alert_scaled = self.scaler.transform(alert_features_raw)

        # nsamples=100: cân bằng giữa tốc độ và độ chính xác
        shap_values = self.explainer.shap_values(alert_scaled, nsamples=100)

        # shap_values là list [class_0, class_1, ...] — lấy class có prob cao nhất
        probs = self.predict_fn(alert_features_raw)[0]
        predicted_class = np.argmax(probs)
        sv = shap_values[predicted_class][0]  # shape: (55,)

        # Lấy top_k features ảnh hưởng nhiều nhất (abs value)
        top_indices = np.argsort(np.abs(sv))[::-1][:top_k]
        top_features = [
            (self.feature_names[i], float(sv[i]), float(alert_features_raw[0][i]))
            for i in top_indices
        ]

        # Tạo text tóm tắt để đưa vào LLM prompt
        summary_lines = []
        for fname, sval, fval in top_features:
            direction = "tăng nguy cơ" if sval > 0 else "giảm nguy cơ"
            summary_lines.append(f"  - {fname}: giá trị={fval:.4f}, SHAP={sval:+.4f} ({direction})")

        summary_text = "\n".join(summary_lines)

        return {
            "shap_values": sv,
            "top_features": top_features,
            "predicted_class_idx": int(predicted_class),
            "class_probabilities": probs.tolist(),
            "summary_text": summary_text,
        }

    def plot_waterfall(self, alert_features_raw, save_path="shap_waterfall.png"):
        """Vẽ waterfall chart — đẹp để đưa vào report."""
        result = self.explain_alert(alert_features_raw)
        sv = result["shap_values"]

        # Lấy base value (expected value của predicted class)
        predicted_class = result["predicted_class_idx"]
        base_val = float(self.explainer.expected_value[predicted_class])

        explanation = shap.Explanation(
            values=sv,
            base_values=base_val,
            data=self.scaler.transform(alert_features_raw)[0],
            feature_names=self.feature_names
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SHAP] Waterfall chart saved → {save_path}")
        return save_path