# vòng lặp huấn luyện (training loop)

# tính phương sai (variance)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

def get_uncertainty(model, data, num_passes=20):
    """
    Sử dụng MC Dropout để tính:
    - reconstruction_errors: MSE loss trung bình của num_passes lần chạy
    - uncertainties: Phương sai giữa num_passes lần chạy (Epistemic Uncertainty)
    
    Trả về: (reconstruction_errors, uncertainties) — cả hai đều là numpy array shape [batch]
    """
    model.train()  # BẬT dropout để kích hoạt MC Dropout

    predictions = []

    with torch.no_grad():
        for _ in range(num_passes):
            recon_batch, _, _ = model(data)
            predictions.append(recon_batch.cpu().numpy())

    # predictions shape: [num_passes, batch, seq_length, feature_size]
    predictions = np.array(predictions)

    # Uncertainty = phương sai giữa các lần chạy -> đo Epistemic Uncertainty
    uncertainty_map = np.var(predictions, axis=0)
    uncertainties = np.mean(uncertainty_map, axis=(1, 2))  # shape: [batch]

    # Reconstruction Error = MSE giữa giá trị trung bình của dự đoán và đầu vào gốc
    mean_prediction = np.mean(predictions, axis=0)  # shape: [batch, seq_len, features]
    data_np = data.cpu().numpy()
    # Tính MSE per sample
    reconstruction_errors = np.mean((mean_prediction - data_np) ** 2, axis=(1, 2))  # shape: [batch]

    return reconstruction_errors, uncertainties


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_results(reconstruction_error, uncertainty_scores, latent_z=None, labels=None):
    """
    Vẽ biểu đồ kết quả phát hiện:
    - Line plot: lỗi tái tạo theo thời gian
    - Boxplot: độ bất định theo từng nhãn
    - Scatter (t-SNE): không gian tiềm ẩn (nếu có)
    """
    plt.figure(figsize=(15, 5))

    # 1. BIỂU ĐỒ LỖI TÁI TẠO (RECONSTRUCTION ERROR)
    plt.subplot(1, 2, 1)
    plt.plot(reconstruction_error, label='Reconstruction Error', color='blue', alpha=0.6)
    threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.title('Phát hiện bất thường qua Sai số tái tạo')
    plt.xlabel('Mẫu dữ liệu (Time)')
    plt.ylabel('MSE Loss')
    plt.legend()

    # 2. BIỂU ĐỒ ĐỘ BẤT ĐỊNH (UNCERTAINTY)
    plt.subplot(1, 2, 2)
    if labels is not None:
        sns.boxplot(x=labels, y=uncertainty_scores)
    else:
        plt.plot(uncertainty_scores, color='orange', alpha=0.6)
    plt.title('Độ bất định (Uncertainty) theo từng kịch bản')
    plt.xlabel('Nhãn')
    plt.ylabel('Uncertainty Score')

    plt.tight_layout()
    plt.show()

    # 3. BIỂU ĐỒ KHÔNG GIAN TIỀM ẨN (LATENT SPACE - t-SNE)
    if latent_z is not None and labels is not None:
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        z_2d = tsne.fit_transform(latent_z)

        plt.figure(figsize=(8, 6))
        label_ids, label_names = pd.factorize(labels)

        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=label_ids, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(scatter)
        cbar.set_ticks(range(len(label_names)))
        cbar.set_ticklabels(label_names)

        plt.title('Trực quan hóa không gian tiềm ẩn (t-SNE)')
        plt.show()


def auto_labeler(errors, uncertainties, threshold_re, threshold_u):
    """
    Gán nhãn tự động dựa trên Reconstruction Error và Uncertainty:
    - RE thấp                  -> Normal
    - RE cao, Uncertainty thấp -> Known Anomaly (tấn công đã biết)
    - RE cao, Uncertainty cao  -> Unknown Attack (tấn công mới / Zero-day)
    """
    labels = []
    for e, u in zip(errors, uncertainties):
        if e > threshold_re:
            if u > threshold_u:
                labels.append("Unknown Attack")
            else:
                labels.append("Known Anomaly")
        else:
            labels.append("Normal")
    return np.array(labels)


def package_to_elk(loader, uncertainties, errors, labels,
                   es=None, index_name="ddos-detection"):
    """
    Đóng gói dữ liệu và gửi đến Elasticsearch.
    Yêu cầu biến 'es' (Elasticsearch client) và 'ELASTICSEARCH_INDEX' đã được khởi tạo.
    """
    meta_data = loader.original_df_subset
    results_to_send = []

    for i in range(len(labels)):
        if labels[i] != "Normal":
            record = {
                "timestamp": meta_data.iloc[i]['timestamp'],
                "ip_source": meta_data.iloc[i]['ip'],
                "recon_error": float(errors[i]),
                "uncertainty": float(uncertainties[i]),
                "predicted_label": labels[i],
                "status": "pending_review" # Trạng thái chờ chuyên gia xác nhận
            }
            results_to_send.append(record)

    if results_to_send:
        if es is None: # Kiểm tra xem client Elasticsearch đã được khởi tạo chưa
            print("Lỗi: Elasticsearch client chưa được khởi tạo. Vui lòng cấu hình 'es'.")
            return None

        print(f"--- Đang gửi {len(results_to_send)} cảnh báo đến Elasticsearch index: {index_name} ---")
        try:
            # Sử dụng bulk helper để gửi nhiều tài liệu hiệu quả hơn
            from elasticsearch.helpers import bulk
            actions = [
                {
                    "_index": index_name,
                    "_source": doc
                }
                for doc in results_to_send
            ]
            success, failed = bulk(es, actions)
            print(f"--- Đã gửi thành công {success} tài liệu, {failed} tài liệu lỗi. ---")
            return True # Trả về True nếu thành công
        except Exception as e:
            print(f"Lỗi khi gửi dữ liệu đến Elasticsearch: {e}")
            return False # Trả về False nếu có lỗi
    else:
        print("Không có cảnh báo nào để gửi.")
        return None
    