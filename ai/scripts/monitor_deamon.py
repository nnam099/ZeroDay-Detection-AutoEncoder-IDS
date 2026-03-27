import time
import pickle
import pandas as pd

from scripts.data_pipeline import parse_apache_log
from utils.dataloader import LogDataLoader
from utils.threshold_tuning import calculate_dynamic_thresholds
from core.detect import get_uncertainty, auto_labeler, package_to_elk
from core.train import self_learning_process


def start_monitoring(model, optimizer, config, es=None):
    """
    Vòng lặp giám sát liên tục để phát hiện tấn công DDoS.

    Mỗi chu kỳ `interval` giây:
    1.  Parse Apache log  → CSV tạm
    2.  Load & tiền xử lý dữ liệu → Tensor
    3.  MC Dropout  → reconstruction_errors, uncertainties
    4.  Gán nhãn   → Normal / Known Anomaly / Unknown Attack
    5.  Thu thập mẫu "Unknown Attack" vào buffer
    6.  Kích hoạt tự học khi buffer đủ queue_limit mẫu
    7.  Gửi cảnh báo lên ELK Stack (nếu đã cấu hình)
    """
    # ── Khởi tạo loader ─────────────────────────────────────────────────────
    loader = LogDataLoader(
        data_path=None,                              # Không load trước; sẽ load mỗi vòng
        seq_length=config['model']['seq_length']
    )

    buffer_new_scenario = []   # Bộ nhớ đệm cho "Unknown Attack"

    # ── Vòng lặp giám sát ───────────────────────────────────────────────────
    while True:
        print("\n" + "="*60)
        print(f"[MONITOR] Đang quét log mới...")

        try:
            # 1. Parse Apache log thô → CSV tạm
            ok = parse_apache_log(
                log_path=config['monitor']['log_path'],
                output_csv=config['monitor']['csv_temp']
            )
            if not ok:
                print(f"[WARN] Bỏ qua chu kỳ — không đọc được log.")
                time.sleep(config['monitor']['interval'])
                continue

            # 2. Load CSV + trích xuất đặc trưng + sliding window
            fresh_df = pd.read_csv(config['monitor']['csv_temp'])
            X_new = loader.get_tensor_data(df=fresh_df, fit=False)

            if X_new is None or len(X_new) == 0:
                print("[INFO] Không có dữ liệu mới đủ để phân tích.")
                time.sleep(config['monitor']['interval'])
                continue

            device = next(model.parameters()).device
            X_new_dev = X_new.to(device)

            # 3. MC Dropout → reconstruction_errors + uncertainties
            reconstruction_errors, uncertainties = get_uncertainty(
                model,
                X_new_dev,
                num_passes=config['detection']['num_passes']
            )

            # 4. Gán nhãn tự động
            labels = auto_labeler(
                errors=reconstruction_errors,
                uncertainties=uncertainties,
                threshold_re=config['detection']['threshold_re'],
                threshold_u=config['detection']['threshold_u']
            )

            # Log thống kê nhanh
            import numpy as np
            unique, counts = np.unique(labels, return_counts=True)
            stat = dict(zip(unique, counts))
            print(f"[RESULT] {stat}")
            print(f"[RESULT] RE   : mean={reconstruction_errors.mean():.5f}, "
                  f"max={reconstruction_errors.max():.5f}")
            print(f"[RESULT] Uncert: mean={uncertainties.mean():.6f}, "
                  f"max={uncertainties.max():.6f}")

            # 5. Gom mẫu "Unknown Attack" vào buffer để tự học
            for i, label in enumerate(labels):
                if label == "Unknown Attack":
                    buffer_new_scenario.append(X_new[i].numpy())

            print(f"[BUFFER] Unknown Attack buffer: {len(buffer_new_scenario)}"
                  f" / {config['self_learning']['queue_limit']}")

            # 6. Tự học nếu đủ mẫu
            sl_cfg = config['self_learning']
            if len(buffer_new_scenario) >= sl_cfg['queue_limit']:
                success = self_learning_process(
                    model=model,
                    optimizer=optimizer,
                    new_patterns=buffer_new_scenario,
                    save_path=config['model']['model_path'],
                    queue_limit=sl_cfg['queue_limit'],
                    epochs=sl_cfg.get('epochs', 20),
                    beta=sl_cfg.get('beta', 0.5)
                )
                if success:
                    buffer_new_scenario.clear()
                    print("[SELF-LEARNING] Buffer đã được reset.")

            # 7. Gửi cảnh báo lên ELK (bỏ qua nếu chưa cấu hình)
            elk_cfg = config.get('elk', {})
            if elk_cfg.get('enabled', False) and es is not None:
                package_to_elk(
                    loader=loader,
                    uncertainties=uncertainties,
                    errors=reconstruction_errors,
                    labels=labels,
                    es=es,
                    index_name=elk_cfg.get('index', 'ddos-detection')
                )

        except Exception as ex:
            import traceback
            print(f"[ERROR] Lỗi trong vòng lặp giám sát: {ex}")
            traceback.print_exc()

        time.sleep(config['monitor']['interval'])