# ══════════════════════════════════════════════════════════════════════════════
# DDoS DETECTION — CNN-BiLSTM-Transformer Hybrid + Focal Loss
# Dataset : CIC-IDS2017  |  Target: Recall Attack ≥ 93%, Precision ≥ 90%
# Platform: Kaggle GPU T4  (cũng chạy được trên Colab / Local)
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
                             roc_auc_score, f1_score, precision_recall_curve)
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# 1. MÔI TRƯỜNG & ĐƯỜNG DẪN
# ─────────────────────────────────────────────────────────────────────────────
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or 'KAGGLE_URL_BASE' in os.environ:
    ENV = "kaggle"
    DATA_DIR = "/kaggle/input"
    WORK_DIR = "/kaggle/working"
else:
    try:
        import google.colab
        ENV = "colab"
        from google.colab import drive
        drive.mount('/content/drive')
        DATA_DIR = "/content/drive/MyDrive/CIC-IDS2017"
        WORK_DIR = "/content/drive/MyDrive/DDoS_Hybrid"
        os.makedirs(WORK_DIR, exist_ok=True)
    except ImportError:
        ENV = "local"
        DATA_DIR = "./data"
        WORK_DIR = "./working"
        os.makedirs(WORK_DIR, exist_ok=True)

print(f"🌍 Môi trường: {ENV.upper()}")

CKPT_PATH  = f"{WORK_DIR}/ckpt_hybrid.pth"
MODEL_BEST = f"{WORK_DIR}/model_hybrid_best.pth"
SCALER_PATH= f"{WORK_DIR}/scaler_hybrid.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# 2. HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
SEQ          = 16      # cửa sổ dài hơn → bắt được pattern tấn công kéo dài
HIDDEN       = 64      # kích thước ẩn CNN / LSTM
NHEAD        = 4       # số attention head
T_LAYERS     = 2       # số lớp Transformer
DROPOUT      = 0.2
TOTAL_EPOCHS = 60
BATCH        = 512
LR           = 2e-4

# Focal Loss params — alpha cao → ưu tiên attack class
FOCAL_ALPHA  = 0.80
FOCAL_GAMMA  = 2.5

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
    raise FileNotFoundError(
        "Không tìm thấy CSV! Hãy Add Input → chọn dataset CIC-IDS2017")

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

# Log1p transform
LOG_IDX = [FEATS.index(f) for f in LOG_FEATS if f in FEATS]
X_all[:, LOG_IDX] = np.log1p(np.abs(X_all[:, LOG_IDX]))
print(f"\nLog1p: {len(LOG_IDX)} features | Class 0 (BENIGN): "
      f"{(y_all==0).sum():,} | Class 1 (Attack): {(y_all==1).sum():,}")

# Split 70 / 15 / 15 — stratified
idx_all = np.arange(len(y_all))
idx_tr_vl, idx_ts = train_test_split(
    idx_all, test_size=0.15, random_state=42, stratify=y_all)
idx_tr, idx_vl = train_test_split(
    idx_tr_vl, test_size=0.15/0.85, random_state=42, stratify=y_all[idx_tr_vl])
print(f"Train: {len(idx_tr):,} | Val: {len(idx_vl):,} | Test: {len(idx_ts):,}")

# Scaler
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as fsc: sc = pickle.load(fsc)
    Xtr = sc.transform(X_all[idx_tr])
    print("✅ Load scaler cũ")
else:
    sc = MinMaxScaler()
    Xtr = sc.fit_transform(X_all[idx_tr])
    with open(SCALER_PATH, 'wb') as fsc: pickle.dump(sc, fsc)
    print("✅ Scaler mới đã lưu")

Xvl = sc.transform(X_all[idx_vl])
Xts = sc.transform(X_all[idx_ts])
ytr = y_all[idx_tr].astype(np.float32)
yvl = y_all[idx_vl].astype(np.float32)
yts = y_all[idx_ts].astype(np.float32)
del X_all, y_all, idx_tr, idx_vl, idx_ts, idx_tr_vl, idx_all; gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 6. SLIDING WINDOWS
# ─────────────────────────────────────────────────────────────────────────────
def make_windows(data, labels, s):
    n, f = data.shape
    if n < s:
        return np.empty((0, s, f), np.float32), np.empty((0,), np.float32)
    shape   = (n - s + 1, s, f)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    W = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides).copy().astype(np.float32)
    sl = (n - s + 1, s)
    sl_strides = (labels.strides[0], labels.strides[0])
    Lw = np.lib.stride_tricks.as_strided(
        labels, shape=sl, strides=sl_strides).copy()
    return W, Lw.max(axis=1).astype(np.float32)

print("\nTạo sliding windows...")
Wtr, Ytr = make_windows(Xtr, ytr, SEQ); del Xtr, ytr; gc.collect()
Wvl, Yvl = make_windows(Xvl, yvl, SEQ); del Xvl, yvl; gc.collect()
Wts, Yts = make_windows(Xts, yts, SEQ); del Xts, yts; gc.collect()
print(f"  Train : {Wtr.shape} | Attack ratio: {Ytr.mean():.3f}")
print(f"  Val   : {Wvl.shape} | Attack ratio: {Yvl.mean():.3f}")
print(f"  Test  : {Wts.shape} | Attack ratio: {Yts.mean():.3f}")

Tr_ds = TensorDataset(torch.tensor(Wtr), torch.tensor(Ytr).unsqueeze(1))
Vl_ds = TensorDataset(torch.tensor(Wvl), torch.tensor(Yvl).unsqueeze(1))
Ts_ds = TensorDataset(torch.tensor(Wts), torch.tensor(Yts).unsqueeze(1))
del Wtr, Ytr, Wvl, Yvl, Wts, Yts; gc.collect()
print("✅ Windows xong!\n")

# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL: CNN-BiLSTM-TRANSFORMER HYBRID
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, :d//2]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class HybridDDoSDetector(nn.Module):
    """
    CNN  →  bắt local pattern (burst, spike)
    BiLSTM  →  temporal ordering (flow sequence)
    Transformer  →  global attention (long-range)
    Head  →  binary classifier
    """
    def __init__(self, F, S, hidden=64, heads=4, t_layers=2, dropout=0.2):
        super().__init__()
        self.F = F
        self.S = S

        # ── Block 1: Local CNN Feature Extraction ──
        self.cnn = nn.Sequential(
            nn.Conv1d(F, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )

        # ── Block 2: Bidirectional LSTM ──
        self.bilstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.lstm_norm = nn.LayerNorm(hidden)

        # ── Block 3: Transformer Encoder ──
        self.pos_enc = PositionalEncoding(hidden, max_len=S + 10)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc_layer, t_layers)

        # ── Block 4: Classification Head ──
        # mean pooling + max pooling → concat → MLP
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        # x : (B, S, F)
        # CNN
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # (B,S,hidden)
        # BiLSTM
        h, _ = self.bilstm(h)                              # (B,S,hidden)
        h = self.lstm_norm(h)
        # Transformer
        h = self.pos_enc(h)
        h = self.transformer(h)                            # (B,S,hidden)
        # Aggregation: mean + max
        h_mean = h.mean(dim=1)
        h_max  = h.max(dim=1).values
        h_agg  = torch.cat([h_mean, h_max], dim=-1)       # (B, hidden*2)
        return self.head(h_agg)                            # (B,1)


model = HybridDDoSDetector(
    F=FEATURE_SIZE, S=SEQ,
    hidden=HIDDEN, heads=NHEAD,
    t_layers=T_LAYERS, dropout=DROPOUT
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} params | CNN-BiLSTM-Transformer Hybrid")

# ─────────────────────────────────────────────────────────────────────────────
# 8. FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    FL(p) = -alpha * (1-p)^gamma * log(p)
    alpha=0.80 → ưu tiên penalty attack class
    gamma=2.5  → focus vào hard examples
    """
    def __init__(self, alpha=0.80, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt   = torch.exp(-bce)
        # alpha: attack=alpha, benign=(1-alpha)
        a_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = a_t * (1 - pt) ** self.gamma * bce
        return loss.mean()

criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# OneCycleLR: warmup + cosine decay → hội tụ nhanh & ổn định
steps_per_epoch = len(Tr_ds) // BATCH
sch = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=LR, epochs=TOTAL_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1, anneal_strategy='cos'
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. LOAD CHECKPOINT NẾU CÓ
# ─────────────────────────────────────────────────────────────────────────────
start_epoch = 1
best_val    = float('inf')
hist        = {'tr': [], 'vl': [], 'vl_auc': []}

if os.path.exists(CKPT_PATH):
    print(f">>> Tìm thấy checkpoint! Đang resume...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    sch.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best_val    = ckpt.get('best_val', float('inf'))
    hist        = ckpt.get('history', hist)
    print(f">>> Resume từ epoch {start_epoch}/{TOTAL_EPOCHS}")
else:
    print(f">>> Train từ đầu (epoch 1/{TOTAL_EPOCHS})")

# ─────────────────────────────────────────────────────────────────────────────
# 10. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
if start_epoch <= TOTAL_EPOCHS:
    ldr  = DataLoader(Tr_ds, batch_size=BATCH, shuffle=True,
                      drop_last=True, pin_memory=True, num_workers=4)
    vldr = DataLoader(Vl_ds, batch_size=BATCH * 2, shuffle=False,
                      pin_memory=True, num_workers=2)

    print(f"Training | {len(ldr)} batches/epoch\n")
    print(f"{'Ep':>4} | {'Train':>8} | {'Val':>8} | {'AUC':>6} | {'Best':>8} | {'LR':>8}")
    print("─" * 60)

    for ep in range(start_epoch, TOTAL_EPOCHS + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for bx, by in ldr:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tr_loss += loss.item()
        atr = tr_loss / len(ldr)
        hist['tr'].append(atr)

        # ── Validation ──
        model.eval()
        vl_loss = 0.0
        vl_probs, vl_labels = [], []
        with torch.no_grad():
            for bx, by in vldr:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE)
                logits = model(bx)
                vl_loss += criterion(logits, by).item()
                vl_probs.append(torch.sigmoid(logits).cpu().numpy())
                vl_labels.append(by.cpu().numpy())

        avl = vl_loss / len(vldr)
        vp  = np.concatenate(vl_probs)
        vl_arr = np.concatenate(vl_labels)
        auc = roc_auc_score(vl_arr, vp)
        hist['vl'].append(avl)
        hist['vl_auc'].append(auc)

        if avl < best_val:
            best_val = avl
            torch.save(model.state_dict(), MODEL_BEST)

        torch.save({
            'epoch': ep, 'model': model.state_dict(),
            'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(),
            'best_val': best_val, 'history': hist,
        }, CKPT_PATH)

        lr_now = opt.param_groups[0]['lr']
        print(f"[{ep:3d}/{TOTAL_EPOCHS}] "
              f"train={atr:.5f}  val={avl:.5f}  "
              f"auc={auc:.4f}  best={best_val:.5f}  lr={lr_now:.1e}")

    print(f"\n✅ Train xong! Best val loss: {best_val:.5f}")

    # Plot learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist['tr'], label='Train (Focal Loss)', color='steelblue')
    axes[0].plot(hist['vl'], label='Val (Focal Loss)', color='orange', ls='--')
    axes[0].set_title('Learning Curve — CNN-BiLSTM-Transformer')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(hist['vl_auc'], label='Val AUC-ROC', color='green')
    axes[1].set_title('Validation AUC-ROC')
    axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0.8, 1.0])
    plt.tight_layout(); plt.savefig(f"{WORK_DIR}/learning_curve.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 11. THRESHOLD TUNING TRÊN VAL SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔍 Tìm threshold tối ưu trên Val set...")
model.load_state_dict(torch.load(MODEL_BEST, map_location=DEVICE))
model.eval()

def get_probs(m, ds, bs=1024):
    ldr = DataLoader(ds, batch_size=bs, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for bx, by in ldr:
            bx = bx.to(DEVICE)
            probs.append(torch.sigmoid(m(bx)).cpu().numpy())
            labels.append(by.numpy())
    return np.concatenate(probs).ravel(), np.concatenate(labels).ravel()

vl_probs, vl_true = get_probs(model, Vl_ds)

# Tìm threshold tối ưu theo F1 (có thể đổi sang Recall nếu muốn)
precisions, recalls, thresholds = precision_recall_curve(vl_true, vl_probs)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx   = np.argmax(f1_scores[:-1])
BEST_THR   = thresholds[best_idx]

# ── Thêm bước kiểm tra recall ──
# Nếu recall tại best F1 threshold < 0.90 → hạ threshold để đạt recall ≥ 0.90
recall_at_best = recalls[best_idx]
if recall_at_best < 0.90:
    # Tìm threshold nhỏ nhất để recall ≥ 0.92
    target_recall = 0.92
    valid = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid):
        BEST_THR = thresholds[valid[-1]]   # threshold lớn nhất vẫn đạt recall
        print(f"⚠️  Recall tại F1-optimal < 90%, điều chỉnh threshold → Recall ≥ {target_recall}")

print(f"★ Threshold tối ưu = {BEST_THR:.4f}")
print(f"  Precision@thr : {precisions[best_idx]:.4f}")
print(f"  Recall@thr    : {recalls[best_idx]:.4f}")
print(f"  F1@thr        : {f1_scores[best_idx]:.4f}")

# Lưu threshold vào file
with open(f"{WORK_DIR}/best_threshold.txt", "w") as ft:
    ft.write(f"threshold={BEST_THR:.6f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 12. ĐÁNH GIÁ TRÊN TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("ĐÁNH GIÁ CUỐI — TẬP TEST")
print("="*70)

ts_probs, ts_true = get_probs(model, Ts_ds)
ts_pred = (ts_probs > BEST_THR).astype(int)
ts_auc  = roc_auc_score(ts_true, ts_probs)

print(classification_report(
    ts_true, ts_pred,
    target_names=['BENIGN', 'Attack'], digits=4))
print(f"AUC-ROC : {ts_auc:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Confusion Matrix
cm = confusion_matrix(ts_true, ts_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Attack'],
            yticklabels=['BENIGN', 'Attack'])
axes[0].set_title(f'Confusion Matrix (thr={BEST_THR:.3f})')

# Probability Distribution
axes[1].hist(ts_probs[ts_true == 0], bins=100, alpha=0.6,
             label='BENIGN', color='steelblue', density=True)
axes[1].hist(ts_probs[ts_true == 1], bins=100, alpha=0.6,
             label='Attack', color='tomato', density=True)
axes[1].axvline(BEST_THR, color='red', ls='--', lw=2,
                label=f'threshold={BEST_THR:.3f}')
axes[1].set_title('Probability Distribution: BENIGN vs Attack')
axes[1].legend(); axes[1].set_xlabel('P(Attack)')

# Precision-Recall Curve
axes[2].plot(recalls, precisions, color='purple', lw=2)
axes[2].scatter([recalls[best_idx]], [precisions[best_idx]],
                color='red', zorder=5, s=80,
                label=f'Best F1={f1_scores[best_idx]:.3f}')
axes[2].set_title('Precision-Recall Curve (Val set)')
axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{WORK_DIR}/evaluation.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 14. TÓM TẮT KẾT QUẢ
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import precision_score, recall_score, f1_score as f1

p  = precision_score(ts_true, ts_pred)
r  = recall_score(ts_true, ts_pred)
f  = f1(ts_true, ts_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"""
╔══════════════════════════════════════════════════════════════╗
║       KẾT QUẢ — CNN-BiLSTM-Transformer Hybrid               ║
╠══════════════════════════════════════════════════════════════╣
║  Threshold   : {BEST_THR:.4f}                                  
║  Precision   : {p:.4f}   (độ chính xác khi báo Attack)
║  Recall      : {r:.4f}   (tỷ lệ bắt được Attack)
║  F1-Score    : {f:.4f}
║  AUC-ROC     : {ts_auc:.4f}
║  False Pos Rate: {fpr:.4f}  (báo nhầm BENIGN thành Attack)
╠══════════════════════════════════════════════════════════════╣
║  TP (bắt đúng Attack)  : {tp:>10,}
║  FN (bỏ sót Attack)    : {fn:>10,}
║  FP (báo nhầm BENIGN)  : {fp:>10,}
║  TN (đúng BENIGN)      : {tn:>10,}
╠══════════════════════════════════════════════════════════════╣
║  Model   : {WORK_DIR}/model_hybrid_best.pth
║  Scaler  : {WORK_DIR}/scaler_hybrid.pkl
║  Thr file: {WORK_DIR}/best_threshold.txt
╚══════════════════════════════════════════════════════════════╝
""")
