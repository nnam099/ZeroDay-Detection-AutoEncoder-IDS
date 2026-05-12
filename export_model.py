# export_model.py
# Dat file nay trong: ZeroDay-Detection-AutoEncoder-IDS/
# Chay: python export_model.py

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ids_v14_unswnb15 import CFG, run_full

# Output vao dung project hien tai
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CFG.save_dir = os.path.join(BASE_DIR, "checkpoints")
CFG.plot_dir = os.path.join(BASE_DIR, "plots")
CFG.data_dir = os.path.join(BASE_DIR, "data", "quick_train") # Chay tren dataset nho cho nhanh
CFG.demo     = False  # Train tren data that, KHONG phai data gia lap!
CFG.epochs   = 5
CFG.patience = 3

CFG.num_workers = 0 # Fix PyTorch Windows multiprocessing crash

os.makedirs(CFG.save_dir, exist_ok=True)
os.makedirs(CFG.plot_dir, exist_ok=True)

if __name__ == '__main__':
    print("Dang train model bang du lieu THAT (UNSW-NB15) de tuong thich voi Dashboard...")
    run_full(CFG)

    print("\n=== XONG ===")
    print(f"File da duoc luu tai: {CFG.save_dir}")