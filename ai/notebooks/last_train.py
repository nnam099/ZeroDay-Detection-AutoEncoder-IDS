# ══════════════════════════════════════════════════════════════════════════════
# DDoS DETECTION v2 — CNN-BiLSTM-Transformer Hybrid
# Dataset : CIC-DDoS2019 (Parquet)
# Platform: Kaggle GPU T4
# ══════════════════════════════════════════════════════════════════════════════

import os, glob, pickle, gc, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score,
                             precision_recall_curve, roc_curve)
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# 1. MÔI TRƯỜNG & ĐƯỜNG DẪN
# ─────────────────────────────────────────────────────────────────────────────
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or 'KAGGLE_URL_BASE' in os.environ:
    ENV      = "kaggle"
    DATA_DIR = "/kaggle/input"
    WORK_DIR = "/kaggle/working"
else:
    try:
        import google.colab
        ENV = "colab"
        from google.colab import drive
        drive.mount('/content/drive')
        DATA_DIR = "/content/drive/MyDrive/CIC-DDoS2019"
        WORK_DIR = "/content/drive/MyDrive/DDoS_CICDDoS2019"
        os.makedirs(WORK_DIR, exist_ok=True)
    except ImportError:
        ENV      = "local"
        DATA_DIR = "./data"
        WORK_DIR = "./working"
        os.makedirs(WORK_DIR, exist_ok=True)

print(f"🌍 Môi trường: {ENV.upper()}")

CKPT_PATH   = f"{WORK_DIR}/ckpt_cicddos2019.pth"
MODEL_BEST  = f"{WORK_DIR}/model_cicddos2019_best.pth"
SCALER_PATH = f"{WORK_DIR}/scaler_cicddos2019.pkl"
RESULT_DIR  = WORK_DIR

# ─────────────────────────────────────────────────────────────────────────────
# 2. HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
SEQ          = 16
HIDDEN       = 64
NHEAD        = 4
T_LAYERS     = 2
DROPOUT      = 0.3
TOTAL_EPOCHS = 80
BATCH        = 512
LR           = 2e-4
PATIENCE     = 12
FOCAL_ALPHA  = 0.80
FOCAL_GAMMA  = 2.5
LABEL_SMOOTH = 0.05
TARGET_FPR   = 0.03

# ── CIC-DDoS2019 features ──────────────────────────────────────────────────
# Dataset này có tên cột khác CIC-IDS2017 — dùng tên chuẩn CIC-DDoS2019
# Code sẽ tự động detect features có sẵn trong file và lọc
FEATS_CANDIDATE = [
    # Flow identifiers / timing
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min',
    # Packet counts
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    # Forward packet stats
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    # Backward packet stats
    'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    # Rates
    'Flow Bytes/s', 'Flow Packets/s',
    'Fwd Packets/s', 'Bwd Packets/s',
    # IAT
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min',
    # Flags
    'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count',
    # Header
    'Fwd Header Length', 'Bwd Header Length',
    # Window
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    # Active/Idle
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    # Other
    'Destination Port', 'Protocol',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance',
    'Down/Up Ratio',
]

# Features cần log1p (phân phối lệch nặng)
LOG_FEATS_CANDIDATE = [
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
    'Fwd Header Length', 'Bwd Header Length',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'Active Mean', 'Active Max', 'Idle Mean', 'Idle Max',
    'Destination Port', 'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
    'Average Packet Size', 'Packet Length Mean', 'Packet Length Variance',
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. KIỂM TRA GPU
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError("⛔ Cần GPU! Vào Settings → Accelerator → GPU T4")
print("✅ GPU sẵn sàng!\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ĐỌC FILE PARQUET — TỰ ĐỘNG DETECT FEATURES & LABEL
# ─────────────────────────────────────────────────────────────────────────────
parquet_files = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True)
print(f"Tìm thấy {len(parquet_files)} file Parquet")
if not parquet_files:
    raise FileNotFoundError(
        "Không tìm thấy file .parquet!\n"
        "Kaggle: Add Input → tìm 'CIC-DDoS2019'\n"
        "Kiểm tra DATA_DIR = " + DATA_DIR
    )

# ── Detect tên cột Label từ file đầu tiên ──
schema_cols = pq.read_schema(parquet_files[0]).names
schema_cols_stripped = [c.strip() for c in schema_cols]

# Tìm cột label (thử các tên phổ biến)
LABEL_COL = None
for candidate in ['Label', 'label', 'Attack', 'Class', 'class', 'attack']:
    if candidate in schema_cols_stripped:
        LABEL_COL = schema_cols[schema_cols_stripped.index(candidate)]
        break
if LABEL_COL is None:
    print("⚠️  Không tìm thấy cột Label tự động!")
    print(f"Các cột có sẵn: {schema_cols_stripped[:20]}")
    raise ValueError("Hãy đặt LABEL_COL thủ công ở đầu script.")
print(f"✅ Cột label: '{LABEL_COL}'")

# ── Detect features từ FEATS_CANDIDATE ──
available = set(schema_cols_stripped)
FEATS_RAW = [f for f in FEATS_CANDIDATE if f in available]
if len(FEATS_RAW) < 5:
    # fallback: lấy tất cả cột số trừ label
    print("⚠️  Ít feature khớp — dùng tất cả cột số")
    FEATS_RAW = [c for c in schema_cols_stripped
                 if c != LABEL_COL and c not in ('Timestamp', 'timestamp')]
print(f"✅ Sử dụng {len(FEATS_RAW)} features")

LOG_FEATS = [f for f in LOG_FEATS_CANDIDATE if f in set(FEATS_RAW)]
print(f"✅ Log1p transform: {len(LOG_FEATS)} features\n")

# ── Đọc tất cả file Parquet ──
chunks = []
for f in parquet_files:
    try:
        schema = pq.read_schema(f)
        file_cols = [c.strip() for c in schema.names]
        cols_to_read = [schema.names[file_cols.index(c)]
                        for c in FEATS_RAW + [LABEL_COL]
                        if c in file_cols]
        tmp = pd.read_parquet(f, columns=cols_to_read)
        tmp.columns = tmp.columns.str.strip()
        chunks.append(tmp)
        print(f"  ✓ {os.path.basename(f)}: {len(tmp):,} rows")
    except Exception as e:
        print(f"  ✗ {os.path.basename(f)}: {e}")

if not chunks:
    raise RuntimeError("Không đọc được file nào!")

df = pd.concat(chunks, ignore_index=True)
del chunks; gc.collect()

# ── Chuẩn hóa nhãn ──
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.upper()

FEATS = [c for c in df.columns if c != LABEL_COL]
FEATURE_SIZE = len(FEATS)
print(f"\nTổng: {len(df):,} rows | Features: {FEATURE_SIZE}")
print(f"\nPhân phối nhãn:")
print(df[LABEL_COL].value_counts().to_string())

# ── Hiển thị tên attack để kiểm tra ──
unique_labels = df[LABEL_COL].unique().tolist()
benign_labels = [l for l in unique_labels if 'BENIGN' in l or l == 'BENIGN']
attack_labels = [l for l in unique_labels if l not in benign_labels]
print(f"\nBENIGN labels : {benign_labels}")
print(f"Attack labels : {attack_labels}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TIỀN XỬ LÝ
# ─────────────────────────────────────────────────────────────────────────────
# Ép kiểu số — Parquet đôi khi có cột object
for col in FEATS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS)
X_all = df[FEATS].values.astype(np.float32)
finite_mask = np.isfinite(X_all).all(axis=1)
X_all = X_all[finite_mask]
df    = df.iloc[finite_mask]

# Binary label: BENIGN=0, Attack=1
y_all = (~df[LABEL_COL].isin(benign_labels)).astype(np.uint8).values
del df; gc.collect()

# Log1p transform
LOG_IDX = [FEATS.index(f) for f in LOG_FEATS if f in FEATS]
X_all[:, LOG_IDX] = np.log1p(np.abs(X_all[:, LOG_IDX]))

benign_count = (y_all == 0).sum()
attack_count = (y_all == 1).sum()
total        = len(y_all)
print(f"\nClass 0 (BENIGN): {benign_count:,} ({benign_count/total*100:.1f}%)")
print(f"Class 1 (Attack): {attack_count:,} ({attack_count/total*100:.1f}%)")

if attack_count == 0:
    raise ValueError("⛔ Không có sample Attack! Kiểm tra lại tên cột Label.")
if benign_count / total > 0.99:
    print("⚠️  Imbalance cực cao — xem xét tăng FOCAL_ALPHA")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SPLIT INDEX (FIX DATA LEAKAGE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[FIX] Tách Train/Val/Test theo sample index trước khi tạo windows...")
idx_all = np.arange(len(y_all))

idx_tr_vl, idx_ts = train_test_split(
    idx_all, test_size=0.15, random_state=42, stratify=y_all)
idx_tr, idx_vl = train_test_split(
    idx_tr_vl, test_size=0.15/0.85, random_state=42,
    stratify=y_all[idx_tr_vl])

# Sort để giữ temporal order (quan trọng cho BiLSTM)
idx_tr = np.sort(idx_tr)
idx_vl = np.sort(idx_vl)
idx_ts = np.sort(idx_ts)

print(f"Train: {len(idx_tr):,} | Val: {len(idx_vl):,} | Test: {len(idx_ts):,}")
print(f"Attack ratio — Train: {y_all[idx_tr].mean():.3f} | "
      f"Val: {y_all[idx_vl].mean():.3f} | "
      f"Test: {y_all[idx_ts].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SCALER — FIT CHỈ TRÊN TRAIN
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as fsc:
        sc = pickle.load(fsc)
    Xtr = sc.transform(X_all[idx_tr])
    print("\n✅ Load scaler cũ")
else:
    sc  = MinMaxScaler()
    Xtr = sc.fit_transform(X_all[idx_tr])
    with open(SCALER_PATH, 'wb') as fsc:
        pickle.dump(sc, fsc)
    print("\n✅ Scaler mới đã lưu")

Xvl = sc.transform(X_all[idx_vl])
Xts = sc.transform(X_all[idx_ts])
ytr = y_all[idx_tr].astype(np.float32)
yvl = y_all[idx_vl].astype(np.float32)
yts = y_all[idx_ts].astype(np.float32)
del X_all, y_all, idx_tr, idx_vl, idx_ts, idx_tr_vl, idx_all
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 8. SLIDING WINDOWS — TẠO RIÊNG TỪNG SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def make_windows(data, labels, s):
    n, f = data.shape
    if n < s:
        return np.empty((0, s, f), np.float32), np.empty((0,), np.float32)
    shape   = (n - s + 1, s, f)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    W  = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides).copy().astype(np.float32)
    sl        = (n - s + 1, s)
    sl_stride = (labels.strides[0], labels.strides[0])
    Lw = np.lib.stride_tricks.as_strided(
        labels, shape=sl, strides=sl_stride).copy()
    return W, Lw.max(axis=1).astype(np.float32)

print("\nTạo sliding windows (mỗi split độc lập)...")
Wtr, Ytr = make_windows(Xtr, ytr, SEQ); del Xtr, ytr; gc.collect()
Wvl, Yvl = make_windows(Xvl, yvl, SEQ); del Xvl, yvl; gc.collect()
Wts, Yts = make_windows(Xts, yts, SEQ); del Xts, yts; gc.collect()

print(f"  Train : {Wtr.shape} | Attack ratio: {Ytr.mean():.3f}")
print(f"  Val   : {Wvl.shape} | Attack ratio: {Yvl.mean():.3f}")
print(f"  Test  : {Wts.shape} | Attack ratio: {Yts.mean():.3f}")

for name, ratio in [("Train", Ytr.mean()), ("Val", Yvl.mean()), ("Test", Yts.mean())]:
    if ratio > 0.85:
        print(f"  ⚠️  {name} attack ratio={ratio:.3f} cao — kiểm tra lại data!")

Tr_ds = TensorDataset(torch.tensor(Wtr), torch.tensor(Ytr).unsqueeze(1))
Vl_ds = TensorDataset(torch.tensor(Wvl), torch.tensor(Yvl).unsqueeze(1))
Ts_ds = TensorDataset(torch.tensor(Wts), torch.tensor(Yts).unsqueeze(1))
del Wtr, Ytr, Wvl, Yvl, Wts, Yts; gc.collect()
print("✅ Windows xong!\n")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MODEL: CNN-BiLSTM-TRANSFORMER HYBRID v2
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, :d//2]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class HybridDDoSDetector(nn.Module):
    def __init__(self, F, S, hidden=64, heads=4, t_layers=2, dropout=0.3):
        super().__init__()

        # Block 1: CNN + Residual
        self.cnn_proj = nn.Conv1d(F, hidden, kernel_size=1)
        self.cnn = nn.Sequential(
            nn.Conv1d(F, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.cnn_drop = nn.Dropout(dropout * 0.5)

        # Block 2: BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout
        )
        self.lstm_norm = nn.LayerNorm(hidden)
        self.lstm_drop = nn.Dropout(dropout)

        # Block 3: Transformer
        self.pos_enc = PositionalEncoding(hidden, max_len=S + 10)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc_layer, t_layers)

        # Block 4: Head
        self.pre_head_norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        xT = x.transpose(1, 2)
        h  = self.cnn(xT) + self.cnn_proj(xT)
        h  = self.cnn_drop(h).transpose(1, 2)
        h, _ = self.bilstm(h)
        h  = self.lstm_drop(self.lstm_norm(h))
        h  = self.pos_enc(h)
        h  = self.transformer(h)
        h_agg = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        h_agg = self.pre_head_norm(h_agg)
        return self.head(h_agg)


model = HybridDDoSDetector(
    F=FEATURE_SIZE, S=SEQ,
    hidden=HIDDEN, heads=NHEAD,
    t_layers=T_LAYERS, dropout=DROPOUT
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} params | CNN-BiLSTM-Transformer | Features={FEATURE_SIZE}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. FOCAL LOSS + LABEL SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=0.80, gamma=2.5, smoothing=0.05):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        t_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce  = F.binary_cross_entropy_with_logits(
            logits, t_smooth, reduction='none')
        pt   = torch.exp(-bce)
        a_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = a_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


criterion = FocalLossWithSmoothing(FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTH)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)

steps_per_epoch = len(Tr_ds) // BATCH
sch = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=LR, epochs=TOTAL_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1, anneal_strategy='cos'
)

# ─────────────────────────────────────────────────────────────────────────────
# 11. LOAD CHECKPOINT NẾU CÓ
# ─────────────────────────────────────────────────────────────────────────────
start_epoch  = 1
best_val_auc = 0.0
patience_cnt = 0
hist = {'tr': [], 'vl': [], 'vl_auc': []}

if os.path.exists(CKPT_PATH):
    print(f"\n>>> Tìm thấy checkpoint! Đang resume...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    sch.load_state_dict(ckpt['scheduler'])
    start_epoch  = ckpt['epoch'] + 1
    best_val_auc = ckpt.get('best_val_auc', 0.0)
    patience_cnt = ckpt.get('patience_cnt', 0)
    hist         = ckpt.get('history', hist)
    print(f">>> Resume từ epoch {start_epoch}/{TOTAL_EPOCHS} | "
          f"Best AUC={best_val_auc:.4f} | Patience={patience_cnt}/{PATIENCE}")
else:
    print(f"\n>>> Train từ đầu (epoch 1/{TOTAL_EPOCHS})")

# ─────────────────────────────────────────────────────────────────────────────
# 12. TRAINING LOOP VỚI EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────
stopped_early = False

if start_epoch <= TOTAL_EPOCHS:
    ldr  = DataLoader(Tr_ds, batch_size=BATCH, shuffle=True,
                      drop_last=True, pin_memory=True, num_workers=4)
    vldr = DataLoader(Vl_ds, batch_size=BATCH * 2, shuffle=False,
                      pin_memory=True, num_workers=2)

    print(f"\nTraining | {len(ldr)} batches/epoch | Patience={PATIENCE}\n")
    print(f"{'Ep':>4} | {'Train':>8} | {'Val':>8} | "
          f"{'AUC':>6} | {'BestAUC':>8} | {'Pat':>4} | {'LR':>8}")
    print("─" * 72)

    for ep in range(start_epoch, TOTAL_EPOCHS + 1):
        # Train
        model.train(); tr_loss = 0.0
        for bx, by in ldr:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            tr_loss += loss.item()
        atr = tr_loss / len(ldr)
        hist['tr'].append(atr)

        # Validation
        model.eval(); vl_loss = 0.0
        vl_probs, vl_labels = [], []
        with torch.no_grad():
            for bx, by in vldr:
                bx = bx.to(DEVICE); by = by.to(DEVICE)
                logits = model(bx)
                vl_loss += criterion(logits, by).item()
                vl_probs.append(torch.sigmoid(logits).cpu().numpy())
                vl_labels.append(by.cpu().numpy())

        avl    = vl_loss / len(vldr)
        vp     = np.concatenate(vl_probs).ravel()
        vl_arr = np.concatenate(vl_labels).ravel()
        auc    = roc_auc_score(vl_arr, vp)
        hist['vl'].append(avl); hist['vl_auc'].append(auc)

        if auc > best_val_auc:
            best_val_auc = auc; patience_cnt = 0
            torch.save(model.state_dict(), MODEL_BEST); flag = "★"
        else:
            patience_cnt += 1; flag = " "

        torch.save({
            'epoch': ep, 'model': model.state_dict(),
            'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(),
            'best_val_auc': best_val_auc, 'patience_cnt': patience_cnt,
            'history': hist,
        }, CKPT_PATH)

        lr_now = opt.param_groups[0]['lr']
        print(f"{flag}[{ep:3d}/{TOTAL_EPOCHS}] train={atr:.5f}  val={avl:.5f}  "
              f"auc={auc:.4f}  best={best_val_auc:.4f}  "
              f"pat={patience_cnt:2d}/{PATIENCE}  lr={lr_now:.1e}")

        if patience_cnt >= PATIENCE:
            print(f"\n⏹️  Early stopping tại epoch {ep}")
            stopped_early = True; break

    status = "Early stopped" if stopped_early else "Hoàn thành"
    print(f"\n✅ {status} | Best Val AUC: {best_val_auc:.6f}")

    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep_r = range(1, len(hist['tr']) + 1)
    axes[0].plot(ep_r, hist['tr'], label='Train', color='steelblue')
    axes[0].plot(ep_r, hist['vl'], label='Val', color='orange', ls='--')
    axes[0].set_title('Learning Curve — CIC-DDoS2019')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep_r, hist['vl_auc'], color='green', label='Val AUC')
    axes[1].axhline(best_val_auc, color='red', ls=':', alpha=0.5,
                    label=f'Best={best_val_auc:.4f}')
    axes[1].set_title('Validation AUC-ROC')
    axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim([max(0.8, min(hist['vl_auc']) - 0.02), 1.005])
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/learning_curve_cicddos2019.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 13. THRESHOLD TUNING TRÊN VAL SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔍 Threshold Tuning trên Val set...")
model.load_state_dict(torch.load(MODEL_BEST, map_location=DEVICE))
model.eval()

def get_probs(m, ds, bs=1024):
    ldr = DataLoader(ds, batch_size=bs, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for bx, by in ldr:
            probs.append(torch.sigmoid(m(bx.to(DEVICE))).cpu().numpy())
            labels.append(by.numpy())
    return np.concatenate(probs).ravel(), np.concatenate(labels).ravel()

vl_probs, vl_true = get_probs(model, Vl_ds)
precisions, recalls, thresholds = precision_recall_curve(vl_true, vl_probs)
f1_scores  = 2 * precisions * recalls / (precisions + recalls + 1e-8)
fpr_arr, tpr_arr, roc_thresholds = roc_curve(vl_true, vl_probs)

best_f1_idx = np.argmax(f1_scores[:-1])
THR_F1      = float(thresholds[best_f1_idx])

valid_fpr = np.where(fpr_arr <= TARGET_FPR)[0]
if len(valid_fpr):
    best_fpr_idx = valid_fpr[np.argmax(tpr_arr[valid_fpr])]
    THR_FPR      = float(roc_thresholds[best_fpr_idx])
else:
    THR_FPR = THR_F1
    best_fpr_idx = best_f1_idx
    print("  ⚠️  Không tìm được FPR ≤ target, dùng F1-optimal")

print(f"\n  Threshold A (Best F1)      : {THR_F1:.4f}")
print(f"    P={precisions[best_f1_idx]:.4f} R={recalls[best_f1_idx]:.4f} "
      f"F1={f1_scores[best_f1_idx]:.4f}")
print(f"\n  Threshold B (FPR≤{TARGET_FPR:.0%})   : {THR_FPR:.4f}")
if len(valid_fpr):
    print(f"    FPR={fpr_arr[best_fpr_idx]:.4f} TPR={tpr_arr[best_fpr_idx]:.4f}")

if len(valid_fpr) and tpr_arr[best_fpr_idx] >= 0.97:
    BEST_THR  = THR_FPR
    THR_LABEL = f"FPR-controlled(≤{TARGET_FPR:.0%})"
else:
    BEST_THR  = THR_F1
    THR_LABEL = "F1-optimal"

print(f"\n★ Threshold chọn: {BEST_THR:.4f} [{THR_LABEL}]")
with open(f"{RESULT_DIR}/best_threshold_cicddos2019.txt", "w") as ft:
    ft.write(f"threshold={BEST_THR:.6f}\nmethod={THR_LABEL}\n"
             f"thr_f1={THR_F1:.6f}\nthr_fpr={THR_FPR:.6f}\n"
             f"feature_size={FEATURE_SIZE}\nseq={SEQ}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 14. ĐÁNH GIÁ TRÊN TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ĐÁNH GIÁ CUỐI — TẬP TEST")
print("=" * 70)

ts_probs, ts_true = get_probs(model, Ts_ds)
ts_pred_f1  = (ts_probs > THR_F1).astype(int)
ts_pred_fpr = (ts_probs > THR_FPR).astype(int)
ts_auc      = roc_auc_score(ts_true, ts_probs)

print(f"\n── Threshold A: F1-optimal (thr={THR_F1:.4f}) ──")
print(classification_report(ts_true, ts_pred_f1,
      target_names=['BENIGN', 'Attack'], digits=4))
print(f"\n── Threshold B: FPR-controlled (thr={THR_FPR:.4f}) ──")
print(classification_report(ts_true, ts_pred_fpr,
      target_names=['BENIGN', 'Attack'], digits=4))
print(f"AUC-ROC : {ts_auc:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 15. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

cm_f1 = confusion_matrix(ts_true, ts_pred_f1)
sns.heatmap(cm_f1, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Normal', 'Attack'], yticklabels=['BENIGN', 'Attack'])
axes[0, 0].set_title(f'Confusion Matrix\nF1-optimal thr={THR_F1:.3f}')

cm_fpr = confusion_matrix(ts_true, ts_pred_fpr)
sns.heatmap(cm_fpr, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            xticklabels=['Normal', 'Attack'], yticklabels=['BENIGN', 'Attack'])
axes[0, 1].set_title(f'Confusion Matrix\nFPR-controlled thr={THR_FPR:.3f}')

axes[0, 2].hist(ts_probs[ts_true == 0], bins=100, alpha=0.6,
                label='BENIGN', color='steelblue', density=True)
axes[0, 2].hist(ts_probs[ts_true == 1], bins=100, alpha=0.6,
                label='Attack', color='tomato', density=True)
axes[0, 2].axvline(THR_F1, color='blue', ls='--', lw=1.5,
                   label=f'F1={THR_F1:.3f}')
axes[0, 2].axvline(THR_FPR, color='green', ls='--', lw=1.5,
                   label=f'FPR={THR_FPR:.3f}')
axes[0, 2].set_title('Probability Distribution')
axes[0, 2].legend(); axes[0, 2].set_xlabel('P(Attack)')

axes[1, 0].plot(recalls, precisions, color='purple', lw=2)
axes[1, 0].scatter([recalls[best_f1_idx]], [precisions[best_f1_idx]],
                   color='red', s=100, zorder=5,
                   label=f'F1={f1_scores[best_f1_idx]:.4f}')
axes[1, 0].set_title('Precision-Recall Curve (Val)')
axes[1, 0].set_xlabel('Recall'); axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(fpr_arr, tpr_arr, color='darkorange', lw=2,
                label=f'AUC={ts_auc:.4f}')
axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
if len(valid_fpr):
    axes[1, 1].scatter([fpr_arr[best_fpr_idx]], [tpr_arr[best_fpr_idx]],
                       color='green', s=100, zorder=5,
                       label=f'FPR={fpr_arr[best_fpr_idx]:.3f}')
axes[1, 1].axvline(TARGET_FPR, color='red', ls=':', alpha=0.5,
                   label=f'Target={TARGET_FPR:.2f}')
axes[1, 1].set_title(f'ROC Curve (AUC={ts_auc:.4f})')
axes[1, 1].set_xlabel('FPR'); axes[1, 1].set_ylabel('TPR')
axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

ep_r = range(1, len(hist['tr']) + 1)
axes[1, 2].plot(ep_r, hist['vl_auc'], color='green', lw=2)
axes[1, 2].axhline(best_val_auc, color='red', ls=':',
                   label=f'Best={best_val_auc:.4f}')
axes[1, 2].fill_between(ep_r, hist['vl_auc'], alpha=0.15, color='green')
axes[1, 2].set_title('Val AUC over Training')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].legend(); axes[1, 2].grid(alpha=0.3)

plt.suptitle('CNN-BiLSTM-Transformer — CIC-DDoS2019', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/evaluation_cicddos2019.png", dpi=150,
            bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 16. TÓM TẮT
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import precision_score, recall_score

def summarize(yt, yp):
    p  = precision_score(yt, yp, zero_division=0)
    r  = recall_score(yt, yp, zero_division=0)
    f  = f1_score(yt, yp, zero_division=0)
    cm = confusion_matrix(yt, yp)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return p, r, f, tp, fp, fn, tn, fpr, fnr

pA,rA,fA,tpA,fpA,fnA,tnA,fprA,fnrA = summarize(ts_true, ts_pred_f1)
pB,rB,fB,tpB,fpB,fnB,tnB,fprB,fnrB = summarize(ts_true, ts_pred_fpr)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║     KẾT QUẢ — CNN-BiLSTM-Transformer | CIC-DDoS2019                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  AUC-ROC      : {ts_auc:.6f}
║  Best Val AUC : {best_val_auc:.6f}
║  Features     : {FEATURE_SIZE}
╠══════════════════════════════════════════════════════════════════════╣
║                 F1-optimal (A)       FPR-controlled (B)
║  Threshold  :  {THR_F1:.4f}               {THR_FPR:.4f}
║  Precision  :  {pA:.4f}               {pB:.4f}
║  Recall     :  {rA:.4f}               {rB:.4f}
║  F1-Score   :  {fA:.4f}               {fB:.4f}
║  False Alarm:  {fprA:.4f} ({fprA*100:.1f}%)         {fprB:.4f} ({fprB*100:.1f}%)
║  Miss Rate  :  {fnrA:.4f} ({fnrA*100:.1f}%)         {fnrB:.4f} ({fnrB*100:.1f}%)
╠══════════════════════════════════════════════════════════════════════╣
║  TP : {tpA:>12,}          {tpB:>12,}
║  FN : {fnA:>12,}          {fnB:>12,}
║  FP : {fpA:>12,}          {fpB:>12,}
║  TN : {tnA:>12,}          {tnB:>12,}
╠══════════════════════════════════════════════════════════════════════╣
║  model_cicddos2019_best.pth
║  scaler_cicddos2019.pkl
║  best_threshold_cicddos2019.txt
╚══════════════════════════════════════════════════════════════════════╝
""")
print("💡 Gợi ý production:")
print(f"  IDS/IPS (bắt tối đa): Threshold A = {THR_F1:.4f}")
print(f"  SOC/NOC (giảm alert): Threshold B = {THR_FPR:.4f}")
