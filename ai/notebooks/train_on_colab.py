# ══════════════════════════════════════════════════════════════
# TRAIN TransformerVAE — PHIÊN BẢN KAGGLE
# Không cần Google Drive, không cần upload zip
# CSV files đã có sẵn trong /kaggle/input/
# ══════════════════════════════════════════════════════════════

import os, glob, pickle, gc
import torch, torch.nn as nn
import torch.nn.functional as F
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────
# CẤU HÌNH & MÔI TRƯỜNG (KAGGLE / COLAB)
# ─────────────────────────────────────────────────────────────
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or 'KAGGLE_URL_BASE' in os.environ:
    IN_COLAB = False
    print("🌍 Môi trường: Kaggle")
    DATA_DIR = "/kaggle/input"
    WORK_DIR = "/kaggle/working"
else:
    try:
        import google.colab
        IN_COLAB = True
        print("🌍 Môi trường: Google Colab")
        from google.colab import drive
        drive.mount('/content/drive')
        # Lưu ý: Sửa đường dẫn này đến thư mục chứa file CSV trên Drive của bạn
        DATA_DIR = "/content/drive/MyDrive/CIC-IDS2017" 
        WORK_DIR = "/content/drive/MyDrive/DDoS_Models"
        os.makedirs(WORK_DIR, exist_ok=True)
    except ImportError:
        IN_COLAB = False
        print("🌍 Môi trường: Local")
        DATA_DIR = "./data"
        WORK_DIR = "./working"
        os.makedirs(WORK_DIR, exist_ok=True)

CKPT_PATH   = f"{WORK_DIR}/checkpoint_dtae_proto40.pth"
MODEL_BEST  = f"{WORK_DIR}/model_dtae_proto40_best.pth"
SCALER_PATH = f"{WORK_DIR}/scaler_dtae_proto40.pkl"

SEQ          = 10
LATENT       = 4
NHEAD        = 2
LAYERS       = 1
DROPOUT      = 0.15
BETA         = 0.1
TOTAL_EPOCHS = 80
BATCH        = 512
LR           = 3e-4

FEATS_RAW = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Fwd IAT Mean',
    'Packet Length Mean', 'SYN Flag Count', 'ACK Flag Count',
    'Init_Win_bytes_forward', 'Active Mean', 'Idle Mean', 'Bwd Packet Length Std'
]
LOG_FEATS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Fwd IAT Mean',
    'Packet Length Mean', 'Init_Win_bytes_forward',
    'Active Mean', 'Idle Mean', 'Bwd Packet Length Std'
]

# ─────────────────────────────────────────────────────────────
# KIỂM TRA GPU
# ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError(
        "⛔ Không có GPU! Vào Settings → Accelerator → GPU T4 ×2"
    )
print("✅ GPU sẵn sàng!")

# ─────────────────────────────────────────────────────────────
# ĐỌC DỮ LIỆU
# ─────────────────────────────────────────────────────────────
csv_files = glob.glob(f"{DATA_DIR}/**/*.csv", recursive=True)
print(f"Tìm thấy {len(csv_files)} file CSV trong {DATA_DIR}")
if len(csv_files) == 0:
    if IN_COLAB:
        raise FileNotFoundError(f"Không tìm thấy CSV! Vui lòng upload dataset lên Google Drive tại thư mục: {DATA_DIR} hoặc sửa lại biến DATA_DIR ở trên.")
    else:
        raise FileNotFoundError("Không có CSV! Hãy click + Add Input → chọn dataset CIC-IDS2017")

chunks = []
for f in csv_files:
    header = pd.read_csv(f, nrows=0)
    cols_needed = [c for c in header.columns
                   if any(c.strip() == r.strip() for r in FEATS_RAW)
                   or c.strip() == 'Label']
    if len(cols_needed) < 2:
        continue
    tmp = pd.read_csv(f, usecols=cols_needed, low_memory=False)
    tmp.columns = tmp.columns.str.strip()
    chunks.append(tmp)
    print(f"  ✓ {os.path.basename(f)}: {len(tmp):,} rows")

df = pd.concat(chunks, ignore_index=True)
del chunks; gc.collect()
df['Label'] = df['Label'].str.strip()
FEATS = [c for c in df.columns if c != 'Label']
FEATURE_SIZE = len(FEATS)
print(f"\nTổng: {len(df):,} rows | Features: {FEATURE_SIZE}")
print(df['Label'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────
# TIỀN XỬ LÝ
# ─────────────────────────────────────────────────────────────
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS)
X_all = df[FEATS].values.astype(np.float32)

# Lọc bỏ triệt để các giá trị inf/nan (như string 'Infinity' còn sót)
finite_mask = np.isfinite(X_all).all(axis=1)
if not finite_mask.all():
    X_all = X_all[finite_mask]
    df = df.iloc[finite_mask]

y_all = (df['Label'] != 'BENIGN').astype(np.uint8).values
del df; gc.collect()

# Log1p transform
LOG_IDX = [FEATS.index(f) for f in LOG_FEATS if f in FEATS]
X_all[:, LOG_IDX] = np.log1p(np.abs(X_all[:, LOG_IDX]))
print(f"Log1p transform: {len(LOG_IDX)} features.")

# Trộn & Tách index (Supervised 70/15/15)
idx_all = np.arange(len(y_all))
idx_tr_vl, idx_ts = train_test_split(idx_all, test_size=0.15, random_state=42, stratify=y_all)
idx_tr, idx_vl = train_test_split(idx_tr_vl, test_size=0.15/0.85, random_state=42, stratify=y_all[idx_tr_vl])

print(f"Train: {len(idx_tr):,} | Val: {len(idx_vl):,} | Test: {len(idx_ts):,}")

# Scaler
if os.path.exists(SCALER_PATH):
    print("✅ Load scaler đã có...")
    with open(SCALER_PATH, 'rb') as f: sc = pickle.load(f)
    Xtr_raw = sc.transform(X_all[idx_tr])
else:
    sc = MinMaxScaler()
    Xtr_raw = sc.fit_transform(X_all[idx_tr])
    with open(SCALER_PATH, 'wb') as f: pickle.dump(sc, f)
    print("✅ Scaler mới đã lưu.")

Xvl_raw = sc.transform(X_all[idx_vl])
Xts_raw = sc.transform(X_all[idx_ts])

# Lưu lại nhãn y tương ứng
y_tr_raw = y_all[idx_tr].astype(np.float32)
y_vl_raw = y_all[idx_vl].astype(np.float32)
y_ts_raw = y_all[idx_ts].astype(np.float32)

del X_all, y_all, idx_tr, idx_vl, idx_ts, idx_tr_vl, idx_all; gc.collect()

# ─────────────────────────────────────────────────────────────
# SLIDING WINDOWS
# ─────────────────────────────────────────────────────────────
def make_windows_with_labels(data, labels, s):
    n, f = data.shape
    if n < s: return np.empty((0, s, f), dtype=np.float32), np.empty((0,), dtype=np.float32)
    shape   = (n - s + 1, s, f)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    W = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).copy().astype(np.float32)
    
    shape_l = (n - s + 1, s)
    strides_l = (labels.strides[0], labels.strides[0])
    L_win = np.lib.stride_tricks.as_strided(labels, shape=shape_l, strides=strides_l).copy()
    L = L_win.max(axis=1) # Có 1 attack là window thành attack
    return W, L

print("Tạo sliding windows có nhãn...")
Wtr, Ytr = make_windows_with_labels(Xtr_raw, y_tr_raw, SEQ); del Xtr_raw, y_tr_raw; gc.collect(); print(f"  Train: X={Wtr.shape}, Y={Ytr.shape}")
Wvl, Yvl = make_windows_with_labels(Xvl_raw, y_vl_raw, SEQ); del Xvl_raw, y_vl_raw; gc.collect(); print(f"  Val  : X={Wvl.shape}, Y={Yvl.shape}")
Wts, Yts = make_windows_with_labels(Xts_raw, y_ts_raw, SEQ); del Xts_raw, y_ts_raw; gc.collect(); print(f"  Test : X={Wts.shape}, Y={Yts.shape}")

Tr_data = TensorDataset(torch.tensor(Wtr), torch.tensor(Ytr).unsqueeze(1)); del Wtr, Ytr; gc.collect()
Vl_data = TensorDataset(torch.tensor(Wvl), torch.tensor(Yvl).unsqueeze(1)); del Wvl, Yvl; gc.collect()
Ts_data = TensorDataset(torch.tensor(Wts), torch.tensor(Yts).unsqueeze(1)); del Wts, Yts; gc.collect()
print("✅ Windows xong!")

# ─────────────────────────────────────────────────────────────
# SUPERVISED TRANSFORMER CLASSIFIER
# ─────────────────────────────────────────────────────────────
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class SupervisedTransformerClassifier(nn.Module):
    def __init__(self, F, S, H, N, D):
        super().__init__(); self.F=F; self.S=S
        
        self.pos_encoder = PositionalEncoding(F, max_len=S)
        self.local_cnn = nn.Conv1d(in_channels=F, out_channels=F, kernel_size=3, padding=1)
        self.act_cnn = nn.GELU()
        
        enc = nn.TransformerEncoderLayer(F, H, F*4, D, batch_first=True, activation='gelu')
        self.encoder  = nn.TransformerEncoder(enc, N)
        self.dropout_z = nn.Dropout(D)
        
        # Đưa sequence tensor về 1 value (logit)
        self.fc_pool = nn.Linear(F*S, F*2)
        self.fc_out  = nn.Linear(F*2, 1)

    def forward(self, x):
        # Local CNN Feature Extraction
        x_cnn = x.transpose(1, 2)
        x_cnn = self.act_cnn(self.local_cnn(x_cnn))
        x_cnn = x_cnn.transpose(1, 2)
        
        x_emb = self.pos_encoder(x + x_cnn)
        h = self.encoder(x_emb).contiguous().view(x_emb.size(0), -1)
        h = self.dropout_z(h)
        h = self.act_cnn(self.fc_pool(h))
        return self.fc_out(h)

model = SupervisedTransformerClassifier(FEATURE_SIZE, SEQ, NHEAD, LAYERS, DROPOUT).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS, eta_min=1e-5)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params | Supervised Sequence Transformer")

# Hàm loss dùng Weighted BCE để phạt nặng FN
# pos_weight = 10.0 (Tăng cường bắt Attack, phạt x10 nếu bỏ sót)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(DEVICE))

# ─────────────────────────────────────────────────────────────
# LOAD CHECKPOINT NẾU CÓ
# ─────────────────────────────────────────────────────────────
start_epoch = 1
best        = float('inf')
hist        = {'tr': [], 'vl': [], 'acc': []}

CKPT_PATH   = f"{WORK_DIR}/checkpoint_sup_x10.pth"
MODEL_BEST  = f"{WORK_DIR}/model_sup_x10_best.pth"

if os.path.exists(CKPT_PATH):
    print(f"\n>>> Tìm thấy checkpoint! Đang resume...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    sch.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best        = ckpt.get('best_val', float('inf'))
    hist        = ckpt.get('history', hist)
    print(f">>> Resume từ epoch {start_epoch}/{TOTAL_EPOCHS}")
else:
    print(f"\n>>> Train từ đầu (Supervised, epoch 1/{TOTAL_EPOCHS})")

# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────
if start_epoch <= TOTAL_EPOCHS:
    ldr = DataLoader(Tr_data, batch_size=BATCH, shuffle=True,
                     drop_last=True, pin_memory=True, num_workers=4)
    vldr = DataLoader(Vl_data, batch_size=BATCH*2, shuffle=False)
    print(f"Training | {len(ldr)} batches/epoch\n")

    for ep in range(start_epoch, TOTAL_EPOCHS + 1):
        model.train(); tl = 0.0
        for b_x, b_y in ldr:
            b_x = b_x.to(DEVICE, non_blocking=True)
            b_y = b_y.to(DEVICE, non_blocking=True)
            
            opt.zero_grad()
            logits = model(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()

        nb = len(ldr)
        atr = tl/nb
        hist['tr'].append(atr)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for vb_x, vb_y in vldr:
                vb_x = vb_x.to(DEVICE)
                vb_y = vb_y.to(DEVICE)
                logits = model(vb_x)
                loss = criterion(logits, vb_y)
                vl += loss.item()
        
        avl = vl / len(vldr)
        hist['vl'].append(avl); sch.step()

        if avl < best:
            best = avl
            torch.save(model.state_dict(), MODEL_BEST)

        torch.save({
            'epoch': ep, 'model': model.state_dict(),
            'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(),
            'best_val': best, 'history': hist,
        }, CKPT_PATH)

        lr_now = opt.param_groups[0]['lr']
        print(f"[{ep:3d}/{TOTAL_EPOCHS}] "
              f"train={atr:.5f} "
              f"val={avl:.5f} best={best:.5f} lr={lr_now:.1e}")

    print(f"\n✅ Train xong!")

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    axes.plot(hist['tr'], label='Train Loss (BCE)')
    axes.plot(hist['vl'], label='Val Loss (BCE)', ls='--')
    axes.set_title('Learning Curve (Supervised)')
    axes.legend(); axes.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────
# ĐÁNH GIÁ TRÊN TẬP TEST
# ─────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_BEST, map_location=DEVICE))
model.eval()

def predict(model, dataloader):
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(DEVICE)
            logits = model(bx)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(by.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)

print("Đánh giá Attack trên tập Test...")
ts_ldr = DataLoader(Ts_data, batch_size=BATCH*2, shuffle=False)
probs, y_true = predict(model, ts_ldr)

# Ranh giới phân loại mặc định là 50%
THR_PROB = 0.5 
y_pred = (probs > THR_PROB).astype(int)

print(f"\n{'='*65}")
print(f"KẾT QUẢ — SUPERVISED CLASSIFIER (pos_weight=10.0)")
print(f"{'='*65}")
print(classification_report(y_true, y_pred,
      target_names=['BENIGN','Attack'], digits=4))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            ax=axes[0], xticklabels=['Normal','Attack'],
            yticklabels=['BENIGN','Attack'])
axes[0].set_title(f'Confusion Matrix (Thr={THR_PROB:.2f})')
axes[1].hist(probs[y_true==0], bins=50, alpha=0.6, label='BENIGN', density=False)
axes[1].hist(probs[y_true==1], bins=50, alpha=0.6, label='Attack', color='tomato', density=False)
axes[1].axvline(THR_PROB, color='red', ls='--', label=f'thr={THR_PROB:.2f}')
axes[1].legend(); axes[1].set_title('Prediction Probability: BENIGN vs Attack')
plt.tight_layout(); plt.show()

print(f"""
╔══════════════════════════════════════════════════════╗
║  Model Định Tuyến Supervised đã lưu                 ║
║  Vào Output panel bên phải để tải về                ║
╚══════════════════════════════════════════════════════╝
""")
