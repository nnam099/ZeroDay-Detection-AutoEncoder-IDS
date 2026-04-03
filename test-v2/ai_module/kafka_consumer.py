from kafka import KafkaConsumer
import json
from config import KAFKA_BROKER, KAFKA_TOPIC

def get_kafka_consumer():
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset='latest', # Chỉ lấy log mới nhất, bỏ qua log cũ khi khởi động lại
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        print(f"✅ Đã kết nối thành công tới Kafka Topic: {KAFKA_TOPIC}")
        return consumer
    except Exception as e:
        print(f"❌ Lỗi kết nối Kafka: {e}")
        return None