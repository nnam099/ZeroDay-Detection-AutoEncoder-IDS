# export_model.py
# Dat file nay trong: ZeroDay-Detection-AutoEncoder-IDS/
# Chay: python export_model.py

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ids_v14_unswnb15 import CFG, run_demo

# Output thang vao DDoS-Mitigation
DDOS_DIR = r"D:\Kì 2 Năm 3\Thực Tập Cơ Sở\AI_Train\DDoS-Mitigation"

CFG.save_dir = os.path.join(DDOS_DIR, "checkpoints")
CFG.plot_dir = os.path.join(DDOS_DIR, "plots")
CFG.demo     = True   # Dung du lieu gia lap, nhe may
CFG.epochs   = 10
CFG.patience = 5

os.makedirs(CFG.save_dir, exist_ok=True)
os.makedirs(CFG.plot_dir, exist_ok=True)

print("Dang chay Demo Mode (nhe - khong can GPU, ~2-3 phut)...")
run_demo(CFG)

print("\n=== XONG ===")
print(f"File da duoc luu tai: {CFG.save_dir}")