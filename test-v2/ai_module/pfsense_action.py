import paramiko
from config import PFSENSE_IP, PFSENSE_USER, PFSENSE_PASS

def block_ip_on_pfsense(target_ip):
    try:
        print(f"🚀 Tiến hành block IP: {target_ip} trên mặt WAN của pfSense...")
        
        # Khởi tạo client SSH
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # Tự động chấp nhận key
        
        # Kết nối
        ssh.connect(PFSENSE_IP, username=PFSENSE_USER, password=PFSENSE_PASS, timeout=5)
        
        # Ra lệnh tạo Rule chặn
        command = f"easyrule block wan {target_ip}"
        stdin, stdout, stderr = ssh.exec_command(command)
        
        # Đọc kết quả trả về từ pfSense
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        
        if error:
            print(f"⚠️ Cảnh báo từ pfSense: {error}")
        else:
            print(f"✅ Đã block thành công! pfSense phản hồi: {output}")
            
        ssh.close()
        return True
    except Exception as e:
        print(f"❌ Lỗi kết nối SSH tới pfSense: {e}")
        return False