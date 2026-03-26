import time
import torch
from scripts.data_pipeline import parse_apache_log
from utils.dataloader import LogDataLoader
from utils.threshold_tuning import calculate_dynamic_thresholds
from core.detect import get_uncertainty, auto_labeler, package_to_elk
from core.train import self_learning_process

def start_monitoring(model, optimizer, config, es=None):
    """
    Vòng lặp giám sát liên tục. Mỗi chu kỳ:
    1. Parse log Apache -> CSV tạm
    2. Load dữ liệu từ CSV -> Tensor
    3. Chạy MC Dropout để tính reconstruction_error & uncertainty
    4. Gán nhãn tự động: Normal / Known Anomaly / Unknown Attack
    5. Thu thập mẫu "Unknown Attack" vào buffer
    6. Kích hoạt tự học (Self-learning) nếu buffer đủ 100 mẫu
    7. Gửi cảnh báo lên ELK Stack (nếu có Elasticsearch client)
    """
    # FIX: data_path được lấy từ config, không để hardcode
    loader = LogDataLoader(
        data_path=config['monitor']['csv_temp'],
        seq_length=config['model']['seq_length']
    )
    buffer_new_scenario = []  # Bộ nhớ đệm cho kịch bản "Unknown Attack"

    while True:
        print("\n--- Đang quét log mới... ---")
        try:
            # 1. Parse Apache log -> CSV tạm
            success = parse_apache_log(
                log_path=config['monitor']['log_path'],
                output_csv=config['monitor']['csv_temp']
            )
            if not success:
                print(f"[WARN] Không tìm thấy file log tại: {config['monitor']['log_path']}")
                time.sleep(config['monitor']['interval'])
                continue

            # 2. Load và chuyển đổi dữ liệu thành tensor
            # FIX: reload raw_data từ CSV mới nhất trước khi gọi get_tensor_data
            import pandas as pd
            fresh_df = pd.read_csv(config['monitor']['csv_temp'])
            X_new = loader.get_tensor_data(df=fresh_df)

            if X_new is None or len(X_new) == 0:
                print("[INFO] Không có dữ liệu mới.")
                time.sleep(config['monitor']['interval'])
                continue

            X_new_cuda = X_new.cuda()

            # 3. FIX: get_uncertainty giờ trả về (errors, uncertainties) — 2 giá trị
            reconstruction_errors, uncertainties = get_uncertainty(
                model,
                X_new_cuda,
                num_passes=config['detection']['num_passes']
            )

            # 4. FIX: auto_labeler nhận đủ cả errors lẫn uncertainties
            labels = auto_labeler(
                errors=reconstruction_errors,
                uncertainties=uncertainties,
                threshold_re=config['detection']['threshold_re'],
                threshold_u=config['detection']['threshold_u']
            )

            # Thống kê nhanh
            unique, counts = __import__('numpy').unique(labels, return_counts=True)
            print(f"[INFO] Kết quả: {dict(zip(unique, counts))}")

            # 5. Thu thập mẫu Unknown Attack vào buffer
            for i, label in enumerate(labels):
                if label == "Unknown Attack":
                    buffer_new_scenario.append(X_new[i].numpy())

            # 6. FIX: self_learning_process cần save_path — lấy từ config
            if len(buffer_new_scenario) >= 100:
                print(f"[INFO] Buffer đủ {len(buffer_new_scenario)} mẫu -> bắt đầu tự học...")
                success = self_learning_process(
                    model=model,
                    optimizer=optimizer,
                    new_patterns=buffer_new_scenario,
                    save_path=config['model']['model_path']
                )
                if success:
                    buffer_new_scenario = []  # Reset buffer sau khi học
                    print("[INFO] Self-learning hoàn tất, buffer đã được reset.")

            # 7. Gửi dữ liệu bất thường lên ELK (es=None sẽ bỏ qua nếu chưa cấu hình)
            package_to_elk(
                loader=loader,
                uncertainties=uncertainties,
                errors=reconstruction_errors,
                labels=labels,
                es=es,
                index_name=config.get('elk', {}).get('index', 'ddos-detection')
            )

        except Exception as ex:
            print(f"[ERROR] Lỗi trong vòng lặp giám sát: {ex}")

        time.sleep(config['monitor']['interval'])