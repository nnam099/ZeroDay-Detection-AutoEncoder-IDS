import os
import sys
import yaml
import torch

from models.transformer_vae import TransformerVAE
from scripts.monitor_deamon import start_monitoring


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # ── Load cấu hình ────────────────────────────────────────────────────────
    config = load_config()
    model_cfg = config['model']

    # ── Chọn device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] Sử dụng device: {device}")

    # ── Khởi tạo Model ───────────────────────────────────────────────────────
    model = TransformerVAE(
        feature_size=model_cfg['feature_size'],
        seq_length=model_cfg['seq_length'],
        latent_dim=model_cfg['latent_dim'],
        nhead=model_cfg.get('nhead', 1),
        num_layers=model_cfg.get('num_layers', 2),
        dropout=model_cfg.get('dropout', 0.1)
    ).to(device)

    # ── Load trọng số đã huấn luyện ──────────────────────────────────────────
    model_path = model_cfg['model_path']
    if not os.path.exists(model_path):
        print(f"[ERROR] Không tìm thấy model tại: {model_path}")
        print("        Vui lòng chạy notebook Colab để huấn luyện và tải model về.")
        sys.exit(1)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()
    print(f"[MAIN] Đã load model từ: {model_path}")

    # ── Optimizer cho Self-learning ──────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg.get('lr', 0.0005)
    )

    # ── Kết nối Elasticsearch (nếu được bật) ─────────────────────────────────
    es = None
    elk_cfg = config.get('elk', {})
    if elk_cfg.get('enabled', False):
        try:
            from elasticsearch import Elasticsearch
            es = Elasticsearch(
                f"http://{elk_cfg.get('host', 'localhost')}:{elk_cfg.get('port', 9200)}",
                basic_auth=(
                    elk_cfg.get('username', ''),
                    elk_cfg.get('password', '')
                ) if elk_cfg.get('username') else None
            )
            print(f"[MAIN] Kết nối Elasticsearch: {elk_cfg.get('host')}:{elk_cfg.get('port')}")
        except Exception as e:
            print(f"[WARN] Không thể kết nối Elasticsearch: {e}")
            es = None

    # ── Bắt đầu giám sát ─────────────────────────────────────────────────────
    print("[MAIN] Bắt đầu giám sát DDoS...")
    start_monitoring(model, optimizer, config, es=es)


if __name__ == "__main__":
    main()