# import json
# import torch
# from kafka import KafkaConsumer
# import redis

# # 1. Khởi tạo các kết nối
# # (Giả định bạn đã train xong một mô hình bằng PyTorch và lưu thành file .pth)
# # model = torch.load("transformer_ddos_model.pth")
# # model.eval()

# # Kết nối Redis để đếm tần suất gói tin (Feature Engineering theo thời gian thực)
# r = redis.Redis(host='redis', port=6379, db=0)

# # Lắng nghe dữ liệu đẩy ra từ Kafka (Thay thế hoàn toàn việc query Elasticsearch)
# consumer = KafkaConsumer(
#     'pfsense-logs-topic', # Tên topic mà Logstash sẽ đẩy log vào
#     bootstrap_servers=['kafka:9092'],
#     value_deserializer=lambda m: json.loads(m.decode('utf-8'))
# )

# print("Đang lắng nghe luồng log thời gian thực từ Kafka...")

# # Vòng lặp này chạy vĩnh viễn, hễ có log rơi vào Kafka là nó chạy ngay lập tức
# for message in consumer:
#     log = message.value
#     src_ip = log.get("src_ip")
    
#     if src_ip:
#         # 2. Xử lý Feature Engineering với Redis
#         # Đếm số lượng gói tin từ IP này (Tự động reset sau 1 giây)
#         packet_count = r.incr(f"rate:{src_ip}")
#         if packet_count == 1:
#             r.expire(f"rate:{src_ip}", 1) 
            
#         # 3. Chuẩn bị dữ liệu cho mô hình (Đưa về dạng Tensor của PyTorch)
#         # Bỏ qua chuỗi IP thô, ta lấy các thông số kỹ thuật số để AI học
#         length = float(log.get("length", 0))
#         protocol_id = float(log.get("protocol_id", 0))
        
#         # Mảng feature đầu vào: [Kích thước, Giao thức, Tốc độ gửi]
#         features = [length, protocol_id, float(packet_count)]
#         tensor_data = torch.tensor([features])
        
#         # 4. Đưa vào Transformer dự đoán (Ví dụ minh họa)
#         # with torch.no_grad():
#         #     prediction = model(tensor_data)
#         #     is_ddos = prediction.item() > 0.8 # Ngưỡng tự tin 80%
        
#         # Nếu phát hiện DDoS, in ra cảnh báo (Hoặc gọi API đẩy lệnh block ngược lại pfSense)
#         # if is_ddos:
#         #     print(f"🚨 CẢNH BÁO DDoS: Đang bị tấn công từ {src_ip} | Tốc độ: {packet_count} pps")
        
#         # In log ra màn hình để test luồng chạy
#         print(f"Nhận log từ {src_ip} | Chiều dài: {length} | Lượng gói tin/s: {packet_count}")




import json
import torch
import torch.nn as nn
from kafka import KafkaConsumer
import redis

# 1. Định nghĩa lại class cấu trúc Model để load tệp .pth
class DDoSDetector(nn.Module):
    def __init__(self):
        super(DDoSDetector, self).__init__()
        # Lưu ý: Input ở đây là 3 (length, protocol_id, packet_rate)
        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# Load Model
model = DDoSDetector()
# Bỏ comment dòng dưới sau khi bạn chạy train.py xong
# model.load_state_dict(torch.load("model.pth"))
model.eval()

# 2. Kết nối Redis và Kafka
r = redis.Redis(host='redis', port=6379, db=0)
consumer = KafkaConsumer(
    'pfsense-logs-topic', 
    bootstrap_servers=['kafka:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("Đang lắng nghe luồng log thời gian thực từ Kafka...")


print("Đang kết nối tới Kafka...")

# Vòng lặp chờ Kafka khởi động
while True:
    try:
        consumer = KafkaConsumer(
            'pfsense-logs-topic', 
            bootstrap_servers=['kafka:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        print("✅ Đã kết nối thành công tới Kafka! Đang lắng nghe luồng log thời gian thực...")
        break # Kết nối thành công thì thoát vòng lặp chờ
    except NoBrokersAvailable:
        print("⏳ Kafka chưa sẵn sàng, đang thử kết nối lại sau 5 giây...")
        time.sleep(5)
        

# 3. Vòng lặp xử lý streaming
for message in consumer:
    log = message.value
    src_ip = log.get("src_ip")
    
    if src_ip:
        # Tính toán packet_rate trong 1 giây qua bằng Redis
        packet_count = r.incr(f"rate:{src_ip}")
        if packet_count == 1:
            r.expire(f"rate:{src_ip}", 1) 
            
        # Trích xuất đặc trưng
        length = float(log.get("length", 0))
        protocol_id = float(log.get("protocol_id", 0))
        
        # Đưa vào Tensor [Chiều dài, Giao thức, Tốc độ gửi]
        features = torch.tensor([[length, protocol_id, float(packet_count)]], dtype=torch.float32)
        
        # Dự đoán
        with torch.no_grad():
            prediction = model(features)
            is_ddos = prediction.item() > 0.8 # Ngưỡng tự tin 80%
        
        if is_ddos:
            print(f"🚨 PHÁT HIỆN DDoS từ {src_ip} | Rate: {packet_count} pps | Pred: {prediction.item():.2f}")