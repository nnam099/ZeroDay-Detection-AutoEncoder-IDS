import socket
import time
import random
from datetime import datetime

# Cấu hình kết nối tới Logstash
HOST = '127.0.0.1'
PORT = 5000

def generate_pfsense_log(is_ddos=False):
    # Lấy thời gian hiện tại chuẩn Syslog (VD: Mar 29 13:45:00)
    now = datetime.now().strftime("%b %d %H:%M:%S")
    
    if not is_ddos:
        # Lưu lượng bình thường (UDP, pass)
        src_ip = f"192.168.1.{random.randint(2, 50)}"
        log = f"{now} pfSense filterlog: 1,2,,,em0,match,pass,in,4,0,,64,123,0,none,17,udp,46,{src_ip},10.0.0.1,54321,53,26,,,,,,\n"
    else:
        # Lưu lượng DDoS (TCP SYN Flood, block, nhắm vào cùng 1 IP đích)
        src_ip = "10.10.10.99" # Giả lập IP của kẻ tấn công
        log = f"{now} pfSense filterlog: 5,0,,,em0,match,block,in,4,0,,128,456,0,none,6,tcp,60,{src_ip},10.0.0.1,12345,80,0,S,12345678,,65535,,\n"
    return log.encode('utf-8')

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Đã kết nối tới Logstash. Bắt đầu gửi dữ liệu...")

        print("\n--- GIAI ĐOẠN 1: Lưu lượng mạng bình thường ---")
        for i in range(10):
            s.sendall(generate_pfsense_log(is_ddos=False))
            print("Đã gửi gói tin bình thường...")
            time.sleep(1) # Gửi chậm rãi

        print("\n🔥 --- GIAI ĐOẠN 2: BẮT ĐẦU TẤN CÔNG DDoS (SYN Flood) --- 🔥")
        for i in range(500):
            s.sendall(generate_pfsense_log(is_ddos=True))
            # Không sleep để tạo tốc độ cực cao, ép Redis đếm packet_rate tăng vọt
        
        print("\nHoàn tất giả lập tấn công!")

except ConnectionRefusedError:
    print("Lỗi: Không thể kết nối. Hãy kiểm tra xem Logstash đã mở cổng 5000 chưa.")