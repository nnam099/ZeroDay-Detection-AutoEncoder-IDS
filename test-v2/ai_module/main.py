import json
from config import MODEL_PATH, SCALER_PATH
from ai_predictor import DDoSPredictor
from kafka_consumer import get_kafka_consumer
from pfsense_action import block_ip_on_pfsense

def extract_features(snort_log):
    """
    Hàm này bóc tách JSON từ Snort thành mảng số thực để đưa vào Model
    """
    try:
        # Lấy IP tấn công
        src_ip = snort_log.get("src_ip", "")
        
        # TODO: Bạn cần trích xuất dữ liệu để đưa vào list 'features' đúng với thứ tự các cột CSV lúc train
        # Dưới đây chỉ là ví dụ mẫu trích xuất từ chuẩn EVE JSON của Snort
        features = [
            snort_log.get("flow", {}).get("pkts_toserver", 0),
            snort_log.get("flow", {}).get("bytes_toserver", 0),
            snort_log.get("tcp", {}).get("syn", False) * 1, # Chuyển True/False thành 1/0
            # Thêm các chỉ số khác ở đây...
        ]
        
        return src_ip, features
    except Exception as e:
        return "", []

def main():
    print("========================================")
    print("⏳ KHỞI ĐỘNG HỆ THỐNG IDS/IPS BẰNG AI")
    print("========================================")
    
    # 1. Đánh thức AI
    predictor = DDoSPredictor(MODEL_PATH, SCALER_PATH)
    
    # 2. Mở cổng Kafka đón luồng log
    consumer = get_kafka_consumer()
    if not consumer:
        return

    print("\n🛡️ HỆ THỐNG ĐANG LẮNG NGHE LOG TỪ PFSENSE CHỜ ĐỢI TẤN CÔNG...")
    
    # Bộ nhớ tạm để tránh spam gọi pfSense khóa 1 IP hàng trăm lần trong 1 giây
    blocked_ips = set()

    # 3. Vòng lặp Real-time vô tận
    for message in consumer:
        log_data = message.value
        
        # Chỉ xử lý các log có chứa IP nguồn
        src_ip, features = extract_features(log_data)
        if not src_ip or not features:
            continue
            
        # 4. Não bộ phân tích
        is_ddos = predictor.predict(features)
        
        # 5. Ra quyết định
        if is_ddos == 1:
            print(f"\n🚨 [CẢNH BÁO MỨC ĐỘ CAO] Phát hiện luồng DDoS từ IP: {src_ip}")
            
            if src_ip not in blocked_ips:
                success = block_ip_on_pfsense(src_ip)
                if success:
                    blocked_ips.add(src_ip)
        else:
            # Nếu là gói tin bình thường, in một dấu chấm để biết hệ thống vẫn đang làm việc
            print(".", end="", flush=True)

if __name__ == "__main__":
    main()