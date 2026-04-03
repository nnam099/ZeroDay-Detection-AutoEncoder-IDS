import os

# 1. Cấu hình Kafka (Docker)
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "snort-logs")

# 2. Cấu hình pfSense SSH
PFSENSE_IP = os.getenv("PFSENSE_IP", "10.0.0.1")
PFSENSE_USER = os.getenv("PFSENSE_USER", "admin")
PFSENSE_PASS = os.getenv("PFSENSE_PASS", "pfsense")

# 3. Đường dẫn Model
MODEL_PATH = "model/model_hybrid_best.pth"
SCALER_PATH = "model/scaler_hybrid.pkl"