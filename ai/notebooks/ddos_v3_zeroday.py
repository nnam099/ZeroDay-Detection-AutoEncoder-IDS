# ══════════════════════════════════════════════════════════════════════════════
# DDoS DETECTION v3 — CNN-BiLSTM-Transformer + Zero-Day Detection
# Cải tiến: Semi-Supervised Hybrid (Supervised + Reconstruction Decoder)
# Dataset : CIC-IDS2017  |  Platform: Kaggle GPU T4
# ══════════════════════════════════════════════════════════════════════════════
#
# KIẾN TRÚC MỚI SO VỚI V2:
#   Shared Encoder (CNN-BiLSTM-Transformer) — giữ nguyên
#   ├── Supervised Head  → P(Attack)           [known attacks]
#   └── Decoder mới      → Reconstruction x̂   [zero-day detection]
#
# LOGIC PHÁT HIỆN:
#   P(Attack) > thr_cls  → ATTACK (known)
#   RE > thr_re          → ANOMALY (zero-day)
#   else                 → BENIGN
# ══════════════════════════════════════════════════════════════════════════════

import os, glob, pickle, gc, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        DATA_DIR = "/content/drive/MyDrive/CIC-IDS2017"
        WORK_DIR = "/content/drive/MyDrive/DDoS_v3"
        os.makedirs(WORK_DIR, exist_ok=True)
    except ImportError:
        ENV      = "local"
        DATA_DIR = "./data"
        WORK_DIR = "./working"
        os.makedirs(WORK_DIR, exist_ok=True)

print(f"🌍 Môi trường: {ENV.upper()}")

CKPT_PATH   = f"{WORK_DIR}/ckpt_v3_zeroday.pth"
MODEL_BEST  = f"{WORK_DIR}/model_v3_zeroday_best.pth"
SCALER_PATH = f"{WORK_DIR}/scaler_v3_zeroday.pkl"
THR_PATH    = f"{WORK_DIR}/thresholds_v3.pkl"
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

# Focal Loss
FOCAL_ALPHA  = 0.80
FOCAL_GAMMA  = 2.5
LABEL_SMOOTH = 0.05

# Combined Loss weight — λ điều chỉnh tỷ lệ Reconstruction Loss
# λ nhỏ (0.05-0.1): ưu tiên classification, reconstruction hỗ trợ
# λ lớn (0.3-0.5): encoder học reconstruction nhiều hơn → zero-day tốt hơn nhưng classification giảm
LAMBDA_REC   = 0.1

# Threshold tuning
TARGET_FPR   = 0.03     # FPR ≤ 3% cho classifier
RE_PERCENTILE = 95      # Percentile RE của BENIGN làm threshold zero-day

FEATS_RAW = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s',
    'Fwd IAT Mean', 'Packet Length Mean', 'SYN Flag Count',
    'ACK Flag Count', 'Init_Win_bytes_forward', 'Active Mean',
    'Idle Mean', 'Bwd Packet Length Std'
]
LOG_FEATS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s',
    'Fwd IAT Mean', 'Packet Length Mean', 'Init_Win_bytes_forward',
    'Active Mean', 'Idle Mean', 'Bwd Packet Length Std'
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
# 4. ĐỌC DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────────
csv_files = glob.glob(f"{DATA_DIR}/**/*.csv", recursive=True)
print(f"Tìm thấy {len(csv_files)} file CSV")
if not csv_files:
    raise FileNotFoundError("Không tìm thấy CSV!")

chunks = []
for f in csv_files:
    header = pd.read_csv(f, nrows=0)
    cols_needed = [c for c in header.columns
                   if any(c.strip() == r for r in FEATS_RAW)
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

# ─────────────────────────────────────────────────────────────────────────────
# 5. TIỀN XỬ LÝ
# ─────────────────────────────────────────────────────────────────────────────
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS)
X_all = df[FEATS].values.astype(np.float32)
finite_mask = np.isfinite(X_all).all(axis=1)
X_all = X_all[finite_mask]
df    = df.iloc[finite_mask]

y_all = (df['Label'] != 'BENIGN').astype(np.uint8).values
del df; gc.collect()

LOG_IDX = [FEATS.index(f) for f in LOG_FEATS if f in FEATS]
X_all[:, LOG_IDX] = np.log1p(np.abs(X_all[:, LOG_IDX]))

benign_count = (y_all == 0).sum()
attack_count = (y_all == 1).sum()
total        = len(y_all)
print(f"\nClass 0 (BENIGN): {benign_count:,} ({benign_count/total*100:.1f}%)")
print(f"Class 1 (Attack): {attack_count:,} ({attack_count/total*100:.1f}%)")

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

idx_tr = np.sort(idx_tr)
idx_vl = np.sort(idx_vl)
idx_ts = np.sort(idx_ts)

print(f"Train: {len(idx_tr):,} | Val: {len(idx_vl):,} | Test: {len(idx_ts):,}")
print(f"Attack ratio — Train: {y_all[idx_tr].mean():.3f} | "
      f"Val: {y_all[idx_vl].mean():.3f} | "
      f"Test: {y_all[idx_ts].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SCALER
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
# 8. SLIDING WINDOWS
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

print("\nTạo sliding windows...")
Wtr, Ytr = make_windows(Xtr, ytr, SEQ); del Xtr, ytr; gc.collect()
Wvl, Yvl = make_windows(Xvl, yvl, SEQ); del Xvl, yvl; gc.collect()
Wts, Yts = make_windows(Xts, yts, SEQ); del Xts, yts; gc.collect()

print(f"  Train : {Wtr.shape} | Attack ratio: {Ytr.mean():.3f}")
print(f"  Val   : {Wvl.shape} | Attack ratio: {Yvl.mean():.3f}")
print(f"  Test  : {Wts.shape} | Attack ratio: {Yts.mean():.3f}")

# Dataset giữ nguyên X (để tính reconstruction loss)
Tr_ds = TensorDataset(torch.tensor(Wtr), torch.tensor(Ytr).unsqueeze(1))
Vl_ds = TensorDataset(torch.tensor(Wvl), torch.tensor(Yvl).unsqueeze(1))
Ts_ds = TensorDataset(torch.tensor(Wts), torch.tensor(Yts).unsqueeze(1))
del Wtr, Ytr, Wvl, Yvl, Wts, Yts; gc.collect()
print("✅ Windows xong!\n")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MODEL: CNN-BiLSTM-TRANSFORMER + DECODER (V3)
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


class HybridZeroDayDetector(nn.Module):
    """
    CNN-BiLSTM-Transformer Hybrid V3 — Zero-Day Detection
    
    Kiến trúc:
      Shared Encoder (giống v2)
        ├── Supervised Head  → logit → P(Attack)   [known attacks]
        └── Decoder (MỚI)   → x̂                   [zero-day detection]
    
    Training: L = Focal(logit, y) + λ × MSE(x̂, x)
    Inference:
      P > thr_cls  → ATTACK (known)
      RE > thr_re  → ANOMALY (zero-day)
      else         → BENIGN
    """
    def __init__(self, F, S, hidden=64, heads=4, t_layers=2, dropout=0.3):
        super().__init__()
        self.F = F
        self.S = S

        # ── Block 1: CNN + Residual (giữ nguyên từ v2) ──
        self.cnn_proj = nn.Conv1d(F, hidden, kernel_size=1)
        self.cnn = nn.Sequential(
            nn.Conv1d(F, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.cnn_drop = nn.Dropout(dropout * 0.5)

        # ── Block 2: BiLSTM (giữ nguyên từ v2) ──
        self.bilstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout
        )
        self.lstm_norm = nn.LayerNorm(hidden)
        self.lstm_drop = nn.Dropout(dropout)

        # ── Block 3: Transformer (giữ nguyên từ v2) ──
        self.pos_enc = PositionalEncoding(hidden, max_len=S + 10)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc_layer, t_layers)

        # ── Block 4: Pre-head norm + Aggregation (giữ nguyên) ──
        self.pre_head_norm = nn.LayerNorm(hidden * 2)

        # ── Block 5: Supervised Head (giữ nguyên từ v2) ──
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1)
        )

        # ── Block 6: Decoder (MỚI — cho zero-day detection) ──
        # Input: embedding z (B, hidden*2=128)
        # Output: reconstructed window x̂ (B, S, F)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 4),   # 128 → 256
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden * 4, hidden * 8),   # 256 → 512
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden * 8, F * S),         # 512 → F*S
            nn.Sigmoid()  # output trong [0,1] khớp với MinMaxScaler output
        )

    def encode(self, x):
        """Shared encoder — trích xuất embedding z."""
        xT = x.transpose(1, 2)                             # (B,F,S)
        h  = self.cnn(xT) + self.cnn_proj(xT)             # residual
        h  = self.cnn_drop(h).transpose(1, 2)             # (B,S,hidden)
        h, _ = self.bilstm(h)                             # (B,S,hidden)
        h  = self.lstm_drop(self.lstm_norm(h))
        h  = self.pos_enc(h)
        h  = self.transformer(h)                          # (B,S,hidden)
        h_agg = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.pre_head_norm(h_agg)                  # (B,hidden*2)

    def forward(self, x):
        z = self.encode(x)                                # (B,128)
        logit = self.head(z)                              # (B,1)
        x_hat = self.decoder(z).view(-1, self.S, self.F)  # (B,S,F)
        return logit, x_hat

    def classify_only(self, x):
        """Chỉ lấy classification output — dùng khi inference nhanh."""
        z = self.encode(x)
        return self.head(z)


model = HybridZeroDayDetector(
    F=FEATURE_SIZE, S=SEQ,
    hidden=HIDDEN, heads=NHEAD,
    t_layers=T_LAYERS, dropout=DROPOUT
).to(DEVICE)

enc_params = sum(p.numel() for name, p in model.named_parameters()
                 if 'decoder' not in name)
dec_params = sum(p.numel() for name, p in model.named_parameters()
                 if 'decoder' in name)
print(f"Model V3: {enc_params + dec_params:,} params total")
print(f"  Encoder + Head : {enc_params:,} params")
print(f"  Decoder (new)  : {dec_params:,} params")

# ─────────────────────────────────────────────────────────────────────────────
# 10. COMBINED LOSS: FOCAL + RECONSTRUCTION
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
        return (a_t * (1 - pt) ** self.gamma * bce).mean()


class CombinedLoss(nn.Module):
    """
    L_total = L_focal(logit, y) + λ × L_mse(x̂, x)
    
    L_focal: phạt sai lầm phân loại, ưu tiên attack class
    L_mse  : buộc encoder học biểu diễn tổng quát của normal traffic
             → traffic lạ (zero-day) sẽ có RE cao khi inference
    λ=0.1  : reconstruction loss đóng vai trò regularizer nhẹ
    """
    def __init__(self, lam=0.1, alpha=0.80, gamma=2.5, smoothing=0.05):
        super().__init__()
        self.focal = FocalLossWithSmoothing(alpha, gamma, smoothing)
        self.lam   = lam

    def forward(self, logit, x_hat, targets, x_orig):
        l_cls = self.focal(logit, targets)
        l_rec = F.mse_loss(x_hat, x_orig)
        l_tot = l_cls + self.lam * l_rec
        return l_tot, l_cls.item(), l_rec.item()


criterion = CombinedLoss(
    lam=LAMBDA_REC,
    alpha=FOCAL_ALPHA,
    gamma=FOCAL_GAMMA,
    smoothing=LABEL_SMOOTH
)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)

steps_per_epoch = len(Tr_ds) // BATCH
sch = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=LR, epochs=TOTAL_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1, anneal_strategy='cos'
)

# ─────────────────────────────────────────────────────────────────────────────
# 11. LOAD CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
start_epoch  = 1
best_val_auc = 0.0
patience_cnt = 0
hist = {'tr_total': [], 'tr_cls': [], 'tr_rec': [],
        'vl_total': [], 'vl_cls': [], 'vl_rec': [], 'vl_auc': []}

if os.path.exists(CKPT_PATH):
    print(f"\n>>> Resume từ checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    sch.load_state_dict(ckpt['scheduler'])
    start_epoch  = ckpt['epoch'] + 1
    best_val_auc = ckpt.get('best_val_auc', 0.0)
    patience_cnt = ckpt.get('patience_cnt', 0)
    hist         = ckpt.get('history', hist)
    print(f">>> Epoch {start_epoch}/{TOTAL_EPOCHS} | "
          f"Best AUC={best_val_auc:.4f} | Pat={patience_cnt}/{PATIENCE}")
else:
    print(f"\n>>> Train từ đầu (epoch 1/{TOTAL_EPOCHS})")

# ─────────────────────────────────────────────────────────────────────────────
# 12. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
stopped_early = False

if start_epoch <= TOTAL_EPOCHS:
    ldr  = DataLoader(Tr_ds, batch_size=BATCH, shuffle=True,
                      drop_last=True, pin_memory=True, num_workers=4)
    vldr = DataLoader(Vl_ds, batch_size=BATCH * 2, shuffle=False,
                      pin_memory=True, num_workers=2)

    print(f"\nTraining V3 | {len(ldr)} batches/epoch | "
          f"λ={LAMBDA_REC} | Patience={PATIENCE}\n")
    print(f"{'Ep':>4} | {'Total':>8} | {'Cls':>7} | {'Rec':>7} | "
          f"{'vAUC':>6} | {'Best':>6} | {'Pat':>4}")
    print("─" * 68)

    for ep in range(start_epoch, TOTAL_EPOCHS + 1):
        # ── Train ──
        model.train()
        tr_tot = tr_cls = tr_rec = 0.0
        for bx, by in ldr:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            logit, x_hat = model(bx)
            loss, lc, lr_ = criterion(logit, x_hat, by, bx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            tr_tot += loss.item()
            tr_cls += lc
            tr_rec += lr_
        nb = len(ldr)
        hist['tr_total'].append(tr_tot / nb)
        hist['tr_cls'].append(tr_cls / nb)
        hist['tr_rec'].append(tr_rec / nb)

        # ── Validation ──
        model.eval()
        vl_tot = vl_cls_l = vl_rec_l = 0.0
        vl_probs, vl_labels = [], []
        with torch.no_grad():
            for bx, by in vldr:
                bx = bx.to(DEVICE); by = by.to(DEVICE)
                logit, x_hat = model(bx)
                loss, lc, lr_ = criterion(logit, x_hat, by, bx)
                vl_tot += loss.item()
                vl_cls_l += lc
                vl_rec_l += lr_
                vl_probs.append(torch.sigmoid(logit).cpu().numpy())
                vl_labels.append(by.cpu().numpy())

        nv = len(vldr)
        vp     = np.concatenate(vl_probs).ravel()
        vl_arr = np.concatenate(vl_labels).ravel()
        auc    = roc_auc_score(vl_arr, vp)
        hist['vl_total'].append(vl_tot / nv)
        hist['vl_cls'].append(vl_cls_l / nv)
        hist['vl_rec'].append(vl_rec_l / nv)
        hist['vl_auc'].append(auc)

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
        print(f"{flag}[{ep:3d}/{TOTAL_EPOCHS}] "
              f"tot={tr_tot/nb:.5f}  "
              f"cls={tr_cls/nb:.5f}  "
              f"rec={tr_rec/nb:.5f}  "
              f"auc={auc:.4f}  "
              f"best={best_val_auc:.4f}  "
              f"pat={patience_cnt:2d}/{PATIENCE}  "
              f"lr={lr_now:.1e}")

        if patience_cnt >= PATIENCE:
            print(f"\n⏹️  Early stopping tại epoch {ep}")
            stopped_early = True; break

    print(f"\n✅ {'Early stopped' if stopped_early else 'Hoàn thành'} | "
          f"Best AUC: {best_val_auc:.6f}")

    # Learning curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ep_r = range(1, len(hist['tr_total']) + 1)

    axes[0].plot(ep_r, hist['tr_total'], label='Train Total', color='steelblue')
    axes[0].plot(ep_r, hist['vl_total'], label='Val Total',
                 color='orange', ls='--')
    axes[0].set_title('Total Loss (Focal + λ×Rec)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep_r, hist['tr_cls'], label='Train Cls', color='steelblue')
    axes[1].plot(ep_r, hist['vl_cls'], label='Val Cls',
                 color='orange', ls='--')
    axes[1].plot(ep_r, hist['tr_rec'], label='Train Rec',
                 color='green', ls=':')
    axes[1].plot(ep_r, hist['vl_rec'], label='Val Rec',
                 color='red', ls=':')
    axes[1].set_title('Classification vs Reconstruction Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(ep_r, hist['vl_auc'], color='green', label='Val AUC')
    axes[2].axhline(best_val_auc, color='red', ls=':', alpha=0.5,
                    label=f'Best={best_val_auc:.4f}')
    axes[2].set_title('Validation AUC-ROC')
    axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].set_ylim([0.90, 1.005])

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/v3_learning_curve.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 13. THRESHOLD TUNING — CLASSIFICATION + RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔍 Threshold Tuning trên Val set...")
model.load_state_dict(torch.load(MODEL_BEST, map_location=DEVICE))
model.eval()

def get_probs_and_re(m, ds, bs=1024):
    """Lấy P(Attack), Reconstruction Error và nhãn từ dataset."""
    ldr = DataLoader(ds, batch_size=bs, shuffle=False)
    probs, res, labels = [], [], []
    with torch.no_grad():
        for bx, by in ldr:
            bx_gpu = bx.to(DEVICE)
            logit, x_hat = m(bx_gpu)
            p  = torch.sigmoid(logit).cpu().numpy()
            re = F.mse_loss(x_hat, bx_gpu, reduction='none')
            re = re.mean(dim=[1, 2]).cpu().numpy()
            probs.append(p)
            res.append(re)
            labels.append(by.numpy())
    return (np.concatenate(probs).ravel(),
            np.concatenate(res).ravel(),
            np.concatenate(labels).ravel())

vl_probs, vl_re, vl_true = get_probs_and_re(model, Vl_ds)

# ── Threshold A: Best F1 (Supervised) ──
precisions, recalls, thresholds = precision_recall_curve(vl_true, vl_probs)
f1_scores   = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_f1_idx = np.argmax(f1_scores[:-1])
THR_CLS_F1  = float(thresholds[best_f1_idx])

# ── Threshold B: FPR-controlled (Supervised) ──
fpr_arr, tpr_arr, roc_thr = roc_curve(vl_true, vl_probs)
valid_fpr = np.where(fpr_arr <= TARGET_FPR)[0]
if len(valid_fpr):
    best_fpr_idx = valid_fpr[np.argmax(tpr_arr[valid_fpr])]
    THR_CLS_FPR  = float(roc_thr[best_fpr_idx])
else:
    THR_CLS_FPR  = THR_CLS_F1
    best_fpr_idx = best_f1_idx

# ── Chọn threshold classification chính ──
if len(valid_fpr) and tpr_arr[best_fpr_idx] >= 0.97:
    THR_CLS   = THR_CLS_FPR
    CLS_LABEL = f"FPR-controlled(≤{TARGET_FPR:.0%})"
else:
    THR_CLS   = THR_CLS_F1
    CLS_LABEL = "F1-optimal"

# ── Threshold RE: Zero-Day Detection ──
# Lấy RE của chỉ BENIGN samples, dùng percentile cao
re_benign = vl_re[vl_true == 0]
re_attack = vl_re[vl_true == 1]
THR_RE    = float(np.percentile(re_benign, RE_PERCENTILE))

print(f"\n  Threshold Cls A (F1-opt)   : {THR_CLS_F1:.4f}")
print(f"    P={precisions[best_f1_idx]:.4f} | "
      f"R={recalls[best_f1_idx]:.4f} | "
      f"F1={f1_scores[best_f1_idx]:.4f}")
print(f"\n  Threshold Cls B (FPR≤{TARGET_FPR:.0%}) : {THR_CLS_FPR:.4f}")
if len(valid_fpr):
    print(f"    FPR={fpr_arr[best_fpr_idx]:.4f} | "
          f"TPR={tpr_arr[best_fpr_idx]:.4f}")
print(f"\n★ Threshold Cls chọn: {THR_CLS:.4f} [{CLS_LABEL}]")
print(f"\n  RE BENIGN p50 : {np.percentile(re_benign, 50):.6f}")
print(f"  RE BENIGN p90 : {np.percentile(re_benign, 90):.6f}")
print(f"  RE BENIGN p95 : {np.percentile(re_benign, 95):.6f}  ← dùng làm THR_RE")
print(f"  RE BENIGN p99 : {np.percentile(re_benign, 99):.6f}")
print(f"  RE Attack mean: {re_attack.mean():.6f}")
print(f"\n★ Threshold RE (zero-day): {THR_RE:.6f} [p{RE_PERCENTILE} of BENIGN]")

# Lưu tất cả threshold
thresholds_dict = {
    'thr_cls': THR_CLS, 'thr_cls_f1': THR_CLS_F1,
    'thr_cls_fpr': THR_CLS_FPR, 'thr_re': THR_RE,
    'cls_label': CLS_LABEL, 're_percentile': RE_PERCENTILE,
    'feature_size': FEATURE_SIZE, 'seq': SEQ,
}
with open(THR_PATH, 'wb') as ft:
    pickle.dump(thresholds_dict, ft)
with open(f"{RESULT_DIR}/thresholds_v3.txt", "w") as ft:
    for k, v in thresholds_dict.items():
        ft.write(f"{k}={v}\n")
print(f"\n✅ Đã lưu thresholds vào {THR_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 14. ĐÁNH GIÁ TRÊN TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ĐÁNH GIÁ CUỐI — TẬP TEST")
print("=" * 70)

ts_probs, ts_re, ts_true = get_probs_and_re(model, Ts_ds)
ts_auc = roc_auc_score(ts_true, ts_probs)

# ── Mode 1: Supervised only (như v2) ──
ts_pred_sup = (ts_probs > THR_CLS).astype(int)

# ── Mode 2: Full detection (Supervised + Zero-Day) ──
# 0 = BENIGN, 1 = Attack (known), 2 = ANOMALY (zero-day)
ts_pred_full = np.zeros_like(ts_true, dtype=int)
ts_pred_full[ts_probs > THR_CLS] = 1            # Known attack
# Zero-day: RE cao nhưng P thấp (chưa được classified là known attack)
zeroday_mask = (ts_re > THR_RE) & (ts_probs <= THR_CLS)
ts_pred_full[zeroday_mask] = 2

# Binary view của full mode (coi ANOMALY = attack)
ts_pred_full_bin = (ts_pred_full > 0).astype(int)

print(f"\n── Mode 1: Supervised only (thr_cls={THR_CLS:.4f}) ──")
print(classification_report(ts_true, ts_pred_sup,
      target_names=['BENIGN', 'Attack'], digits=4))

print(f"\n── Mode 2: Full detection (Supervised + Zero-Day thr_re={THR_RE:.6f}) ──")
print(classification_report(ts_true, ts_pred_full_bin,
      target_names=['BENIGN', 'Attack+Anomaly'], digits=4))

print(f"\nAUC-ROC : {ts_auc:.6f}")

# Thống kê zero-day detections
n_known   = (ts_pred_full == 1).sum()
n_zeroday = (ts_pred_full == 2).sum()
n_benign  = (ts_pred_full == 0).sum()
print(f"\nPhân phối prediction (Mode 2):")
print(f"  BENIGN   : {n_benign:,} ({n_benign/len(ts_pred_full)*100:.1f}%)")
print(f"  Attack   : {n_known:,} ({n_known/len(ts_pred_full)*100:.1f}%)")
print(f"  ANOMALY  : {n_zeroday:,} ({n_zeroday/len(ts_pred_full)*100:.1f}%)")

# Trong số ANOMALY, bao nhiêu là thực sự attack?
if n_zeroday > 0:
    true_attacks_as_anomaly = ts_true[zeroday_mask].sum()
    print(f"\n  Trong {n_zeroday:,} ANOMALY predictions:")
    print(f"    Thực sự là Attack: {true_attacks_as_anomaly:,} "
          f"({true_attacks_as_anomaly/n_zeroday*100:.1f}%)")
    print(f"    Thực sự là BENIGN: {n_zeroday - true_attacks_as_anomaly:,} "
          f"({(n_zeroday - true_attacks_as_anomaly)/n_zeroday*100:.1f}%) ← false alarm zero-day")

# ─────────────────────────────────────────────────────────────────────────────
# 15. PHÂN TÍCH RE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 Phân tích Reconstruction Error...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# [0,0] Confusion Matrix — Supervised only
cm_sup = confusion_matrix(ts_true, ts_pred_sup)
sns.heatmap(cm_sup, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Normal', 'Attack'], yticklabels=['BENIGN', 'Attack'])
axes[0, 0].set_title(f'CM — Supervised only\nthr_cls={THR_CLS:.3f}')

# [0,1] Confusion Matrix — Full mode
cm_full = confusion_matrix(ts_true, ts_pred_full_bin)
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            xticklabels=['Normal', 'Attack'], yticklabels=['BENIGN', 'Attack'])
axes[0, 1].set_title(f'CM — Full (Sup + Zero-Day)\nthr_re={THR_RE:.4f}')

# [0,2] P(Attack) distribution
axes[0, 2].hist(ts_probs[ts_true == 0], bins=100, alpha=0.6,
                label='BENIGN', color='steelblue', density=True)
axes[0, 2].hist(ts_probs[ts_true == 1], bins=100, alpha=0.6,
                label='Attack', color='tomato', density=True)
axes[0, 2].axvline(THR_CLS, color='red', ls='--', lw=2,
                   label=f'thr_cls={THR_CLS:.3f}')
axes[0, 2].set_title('P(Attack) Distribution')
axes[0, 2].legend(); axes[0, 2].set_xlabel('P(Attack)')

# [1,0] RE distribution (log scale)
re_b = ts_re[ts_true == 0]
re_a = ts_re[ts_true == 1]
axes[1, 0].hist(re_b, bins=100, alpha=0.6,
                label='BENIGN', color='steelblue', density=True)
axes[1, 0].hist(re_a, bins=100, alpha=0.6,
                label='Attack', color='tomato', density=True)
axes[1, 0].axvline(THR_RE, color='red', ls='--', lw=2,
                   label=f'thr_re={THR_RE:.4f}')
axes[1, 0].set_title('Reconstruction Error Distribution')
axes[1, 0].legend(); axes[1, 0].set_xlabel('RE = MSE(x, x̂)')

# [1,1] Scatter P(Attack) vs RE — 2D decision boundary
sample_idx = np.random.choice(len(ts_true), min(5000, len(ts_true)), replace=False)
colors = np.where(ts_true[sample_idx] == 0, 'steelblue', 'tomato')
axes[1, 1].scatter(ts_probs[sample_idx], ts_re[sample_idx],
                   c=colors, alpha=0.3, s=8)
axes[1, 1].axvline(THR_CLS, color='blue', ls='--', lw=1.5,
                   label=f'thr_cls={THR_CLS:.3f}')
axes[1, 1].axhline(THR_RE, color='red', ls='--', lw=1.5,
                   label=f'thr_re={THR_RE:.4f}')
axes[1, 1].set_xlabel('P(Attack)')
axes[1, 1].set_ylabel('Reconstruction Error')
axes[1, 1].set_title('2D Decision Space\n(blue=BENIGN, red=Attack)')
axes[1, 1].legend()

# Annotate vùng
axes[1, 1].text(0.05, THR_RE * 1.5, 'ANOMALY\n(Zero-Day)',
                fontsize=9, color='darkred', alpha=0.7)
axes[1, 1].text(THR_CLS + 0.02, THR_RE * 0.3, 'ATTACK\n(Known)',
                fontsize=9, color='navy', alpha=0.7)
axes[1, 1].text(0.05, THR_RE * 0.3, 'BENIGN',
                fontsize=9, color='steelblue', alpha=0.7)

# [1,2] ROC Curve
axes[1, 2].plot(fpr_arr, tpr_arr, color='darkorange', lw=2,
                label=f'AUC={ts_auc:.4f}')
axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
if len(valid_fpr):
    axes[1, 2].scatter([fpr_arr[best_fpr_idx]], [tpr_arr[best_fpr_idx]],
                       color='green', s=100, zorder=5,
                       label=f'FPR-thr={fpr_arr[best_fpr_idx]:.3f}')
axes[1, 2].set_title(f'ROC Curve (AUC={ts_auc:.4f})')
axes[1, 2].set_xlabel('FPR'); axes[1, 2].set_ylabel('TPR')
axes[1, 2].legend(); axes[1, 2].grid(alpha=0.3)

plt.suptitle('CNN-BiLSTM-Transformer V3 — Known Attack + Zero-Day Detection',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/v3_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 16. TÓM TẮT KẾT QUẢ
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import precision_score, recall_score

def summarize(yt, yp):
    p  = precision_score(yt, yp, zero_division=0)
    r  = recall_score(yt, yp, zero_division=0)
    f  = f1_score(yt, yp, zero_division=0)
    cm = confusion_matrix(yt, yp)
    tn, fp, fn, tp = cm.ravel()
    return p, r, f, tp, fp, fn, tn, fp/(fp+tn+1e-8), fn/(fn+tp+1e-8)

pS,rS,fS,tpS,fpS,fnS,tnS,fprS,fnrS = summarize(ts_true, ts_pred_sup)
pF,rF,fF,tpF,fpF,fnF,tnF,fprF,fnrF = summarize(ts_true, ts_pred_full_bin)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║     KẾT QUẢ V3 — CNN-BiLSTM-Transformer + Zero-Day Detection        ║
╠══════════════════════════════════════════════════════════════════════╣
║  AUC-ROC      : {ts_auc:.6f}
║  Best Val AUC : {best_val_auc:.6f}
║  Lambda Rec   : {LAMBDA_REC}
╠══════════════════════════════════════════════════════════════════════╣
║              Mode 1 (Supervised)    Mode 2 (+ Zero-Day)
║  Thr Cls   : {THR_CLS:.4f}               {THR_CLS:.4f}
║  Thr RE    : N/A                    {THR_RE:.6f}
║  Precision : {pS:.4f}               {pF:.4f}
║  Recall    : {rS:.4f}               {rF:.4f}
║  F1-Score  : {fS:.4f}               {fF:.4f}
║  FPR       : {fprS:.4f} ({fprS*100:.1f}%)         {fprF:.4f} ({fprF*100:.1f}%)
║  Miss Rate : {fnrS:.4f} ({fnrS*100:.1f}%)         {fnrF:.4f} ({fnrF*100:.1f}%)
╠══════════════════════════════════════════════════════════════════════╣
║  TP : {tpS:>12,}          {tpF:>12,}
║  FN : {fnS:>12,}          {fnF:>12,}
║  FP : {fpS:>12,}          {fpF:>12,}
║  TN : {tnS:>12,}          {tnF:>12,}
╠══════════════════════════════════════════════════════════════════════╣
║  ANOMALY (zero-day) detected: {n_zeroday:,} samples
║  Trong đó thực sự là Attack : {int(true_attacks_as_anomaly):,} samples
╠══════════════════════════════════════════════════════════════════════╣
║  model_v3_zeroday_best.pth
║  scaler_v3_zeroday.pkl
║  thresholds_v3.pkl  (thr_cls={THR_CLS:.4f}, thr_re={THR_RE:.6f})
╚══════════════════════════════════════════════════════════════════════╝
""")

# ─────────────────────────────────────────────────────────────────────────────
# 17. HÀM INFERENCE PRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_production(model, x_window_tensor,
                       thr_cls=THR_CLS, thr_re=THR_RE):
    """
    Inference production — trả về nhãn và scores.
    
    Args:
        model           : HybridZeroDayDetector (đã load weights)
        x_window_tensor : Tensor (1, SEQ, F) — đã qua scaler
        thr_cls         : ngưỡng classification (default từ tuning)
        thr_re          : ngưỡng reconstruction error (default từ tuning)
    
    Returns:
        label    : 'BENIGN' | 'ATTACK' | 'ANOMALY'
        p_attack : float — P(Attack) từ supervised head
        re_score : float — Reconstruction Error
        reason   : str   — lý do quyết định
    """
    model.eval()
    with torch.no_grad():
        x = x_window_tensor.to(DEVICE)
        logit, x_hat = model(x)
        p_attack = torch.sigmoid(logit).item()
        re_score = F.mse_loss(x_hat, x, reduction='none').mean().item()

    if p_attack > thr_cls:
        return 'ATTACK', p_attack, re_score, f'P={p_attack:.4f} > thr={thr_cls:.4f}'
    elif re_score > thr_re:
        return 'ANOMALY', p_attack, re_score, f'RE={re_score:.6f} > thr={thr_re:.6f}'
    else:
        return 'BENIGN', p_attack, re_score, f'P={p_attack:.4f} ≤ {thr_cls:.4f}, RE={re_score:.6f} ≤ {thr_re:.6f}'

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  CÁCH DÙNG INFERENCE PRODUCTION:                                     ║
║                                                                      ║
║  # Load model + thresholds                                           ║
║  model.load_state_dict(torch.load('model_v3_zeroday_best.pth'))      ║
║  with open('thresholds_v3.pkl','rb') as f: thr = pickle.load(f)      ║
║                                                                      ║
║  # Predict 1 window                                                  ║
║  x = torch.tensor(window).unsqueeze(0)  # (1,16,14)                 ║
║  label, p, re, reason = predict_production(                          ║
║      model, x, thr['thr_cls'], thr['thr_re'])                        ║
║                                                                      ║
║  # label: 'BENIGN' | 'ATTACK' | 'ANOMALY'                           ║
╚══════════════════════════════════════════════════════════════════════╝
""")
