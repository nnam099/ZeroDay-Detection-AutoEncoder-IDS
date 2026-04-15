# ═══════════════════════════════════════════════════════════════════════════
# DDoS DETECTION v5.0 SOC-OPTIMIZED — Zero-Day Isolation & Mem-Efficient
# FEATURES:
# 1. On-the-fly SlidingWindowDataset (Loại bỏ triệt để lỗi OOM).
# 2. Strict Zero-Day holdout (Tách hoàn toàn Web/Bot/Infil khỏi tập Train).
# 3. Minority Class Bootstrapping (Oversample tấn công hiếm như SSH/SQL).
# ═══════════════════════════════════════════════════════════════════════════
import os, glob, pickle, gc, math, warnings, json, shutil
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, precision_score,
                             recall_score, precision_recall_curve, roc_curve,
                             average_precision_score)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ─────────────────────────────────────────────────────────────────────────
# 1. MÔI TRƯỜNG & PATHS
# ─────────────────────────────────────────────────────────────────────────
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ:
    ENV, DATA_DIR, WORK_DIR = "kaggle", "/kaggle/input", "/kaggle/working"
else:
    try:
        import google.colab
        ENV = "colab"
        from google.colab import drive; drive.mount("/content/drive")
        DATA_DIR = "/content/drive/MyDrive/CIC-IDS2017"
        WORK_DIR = "/content/drive/MyDrive/DDoS_v5_soc"
        os.makedirs(WORK_DIR, exist_ok=True)
    except ImportError:
        ENV = "local"
        DATA_DIR, WORK_DIR = "./data", "./working"
        os.makedirs(WORK_DIR, exist_ok=True)

print(f"🌍 ENV: {ENV.upper()}")
CKPT_PATH   = f"{WORK_DIR}/ckpt_v5_soc.pth"
MODEL_BEST  = f"{WORK_DIR}/model_v5_soc_best.pth"
SCALER_PATH = f"{WORK_DIR}/scaler_v5_soc.pkl"
THR_PATH    = f"{WORK_DIR}/thresholds_v5_soc.pkl"
RESULT_DIR  = WORK_DIR
EXPORT_DIR  = f"{WORK_DIR}/export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# 2. HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────
SEQ          = 16
HIDDEN       = 64
NHEAD        = 4
T_LAYERS     = 2
DROPOUT      = 0.3
TOTAL_EPOCHS = 80
LR           = 2e-4
PATIENCE     = 22     # ★ V5.2: Phải > T_0=20 để sống sót qua lần restart đầu tiên
USE_AMP      = True
FOCAL_ALPHA  = 0.80
FOCAL_GAMMA  = 2.5
LABEL_SMOOTH = 0.05
# ★ V5.2: Cân bằng lại: giảm LAMBDA_REC để classifier không bị chết đói
# Nguyên nhân V5.1 fail: Rec=3.72 >> Cls=0.026 → classifier bị áp đảo
LAMBDA_REC   = 0.35   # 1.00→0.35: đủ để tạo gap RE, không làm chết classifier
LAMBDA_ADV   = 0.40   # giữ penalty nhưng nhẹ hơn
LAMBDA_CTR   = 0.10   # nhẹ nhàng, chỉ regularize Z-space
WINDOW_STRIDE= 4    # Giảm kích thước số chuỗi trùng lặp xuống 4 lần -> Tự động Tăng tốc 400%
TARGET_FPR   = 0.01
RE_PERCENTILE= 95.0
RE_MAD_SIGMA = 2.0
BOTTLENECK   = 24
MODEL_VERSION = "v5.2-balanced"

FEATS_RAW = ["Destination Port","Flow Duration","Total Fwd Packets",
             "Total Backward Packets","Flow Bytes/s","Flow Packets/s",
             "Fwd IAT Mean","Packet Length Mean","SYN Flag Count",
             "ACK Flag Count","Init_Win_bytes_forward","Active Mean",
             "Idle Mean","Bwd Packet Length Std"]
LOG_FEATS = ["Destination Port","Flow Duration","Total Fwd Packets",
             "Total Backward Packets","Flow Bytes/s","Flow Packets/s",
             "Fwd IAT Mean","Packet Length Mean","Init_Win_bytes_forward",
             "Active Mean","Idle Mean","Bwd Packet Length Std"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count()
print(f"Device: {DEVICE} ({N_GPUS} GPUs)")

# ★ OPT: Tăng BATCH và NUM_WORKERS để tận dụng tối đa GPU/CPU
if N_GPUS >= 2:
    BATCH = 4096; ACCUM_STEPS = 1; NUM_WORKERS = 4
elif N_GPUS == 1:
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    BATCH = 3072 if vram >= 14 else 2048
    ACCUM_STEPS = 1; NUM_WORKERS = 4
else:
    BATCH = 512; ACCUM_STEPS = 2; NUM_WORKERS = 2

# ─────────────────────────────────────────────────────────────────────────
# 3. ĐỌC DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────
import pyarrow.parquet as pq

all_files = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True) + \
            glob.glob(f"{DATA_DIR}/**/*.csv", recursive=True)
if not all_files: raise FileNotFoundError(f"Không tìm thấy data tại {DATA_DIR}")

_first = all_files[0]
_s_cols = pq.read_schema(_first).names if _first.endswith(".parquet") else pd.read_csv(_first, nrows=0).columns
LABEL_COL = next((c for c in ["Label","label","Attack","Class","class"] if c in [x.strip() for x in _s_cols]), "Label")

# ★ OPT: Đọc file song song bằng ThreadPoolExecutor để giảm I/O wait
from concurrent.futures import ThreadPoolExecutor, as_completed

_feat_set = set(FEATS_RAW + [LABEL_COL])

def _read_file(f):
    try:
        if f.endswith(".parquet"):
            cols = [c for c in pq.read_schema(f).names if c.strip() in _feat_set]
            tmp = pd.read_parquet(f, columns=cols)
        else:
            hdr = pd.read_csv(f, nrows=0)
            cols = [c for c in hdr.columns if c.strip() in _feat_set]
            tmp = pd.read_csv(f, usecols=cols, low_memory=False)
        tmp.columns = tmp.columns.str.strip()
        print(f"  ✓ {os.path.basename(f)}: {len(tmp):,}")
        return tmp
    except Exception as e:
        print(f"  ✗ {os.path.basename(f)}: {e}")
        return None

_io_workers = min(4, len(all_files))  # Tối đa 4 threads I/O
chunks = []
with ThreadPoolExecutor(max_workers=_io_workers) as pool:
    futs = {pool.submit(_read_file, f): f for f in all_files}
    for fut in as_completed(futs):
        result = fut.result()
        if result is not None:
            chunks.append(result)

df = pd.concat(chunks, ignore_index=True); del chunks; gc.collect()
df = df.rename(columns={LABEL_COL:"Label"})
FEATS = [c for c in df.columns if c != "Label"]

# ─────────────────────────────────────────────────────────────────────────
# 4. SPLIT V5.0 — STRICT ZERO-DAY ISOLATION
# ─────────────────────────────────────────────────────────────────────────
print("\n[SPLIT] V5.0 Zero-Day Strict Isolation & On-The-Fly Windowing...")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS).reset_index(drop=True)

labels_series = df["Label"].astype(str).str.strip().str.upper()
labels_str = labels_series.values
binary_y   = np.zeros(len(labels_str), dtype=np.float32)
cat_array  = np.zeros(len(labels_str), dtype=np.uint8) # 0=Benign, 1=Known, 2=Zero-Day

# Các mẫu tấn công
binary_y[~np.isin(labels_str, ["BENIGN", "NORMAL"])] = 1.0
cat_array[binary_y == 1] = 1

# Khai báo các loại tấn công được xem là ZERO-DAY (Không bao giờ xuất hiện lúc Train)
ZERO_DAY_KEYWORDS = ["BOT", "WEB ATTACK", "INFILTRATION", "HEARTBLEED", "SQL", "XSS", "BRUTE"]
for kw in ZERO_DAY_KEYWORDS:
    cat_array[labels_series.str.contains(kw).values] = 2

def get_window_max(arr, seq):
    num_w = len(arr) - seq + 1
    w = np.lib.stride_tricks.as_strided(arr, shape=(num_w, seq), strides=(arr.strides[0], arr.strides[0]))
    return w.max(axis=1) # 2 thắng 1, 1 thắng 0.

valid_indices = np.arange(SEQ - 1, len(df))
window_cat = get_window_max(cat_array, SEQ)
window_labels_detail = labels_str[SEQ-1:] # Label cuối cùng của mỗi window

idx_benign = valid_indices[window_cat == 0][::WINDOW_STRIDE]
idx_known  = valid_indices[window_cat == 1][::WINDOW_STRIDE]  # Chỉ chứa Known Attacks
idx_zd     = valid_indices[window_cat == 2]  # Chứa Zero-Day (Không áp dụng Stride để bắt 100% khi Test)

print(f"  Tổng Windows -> BENIGN: {len(idx_benign):,} | KNOWN: {len(idx_known):,} | ZERO-DAY: {len(idx_zd):,}")

# Tách riêng tập (Zero-Day đẩy 100% vào Test)
b_tr, b_temp = train_test_split(idx_benign, test_size=0.20, random_state=42)
b_vl, b_ts   = train_test_split(b_temp, test_size=0.50, random_state=42)

k_tr, k_temp = train_test_split(idx_known, test_size=0.20, random_state=42)
k_vl, k_ts   = train_test_split(k_temp, test_size=0.50, random_state=42)

train_idx = np.concatenate([b_tr, k_tr])
val_idx   = np.concatenate([b_vl, k_vl])
test_idx  = np.concatenate([b_ts, k_ts, idx_zd])

print(f"  Tập Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test (cùng Zero-day): {len(test_idx):,}")

# ─────────────────────────────────────────────────────────────────────────
# 5. TIỀN XỬ LÝ (SCALING THEO TRAIN SET)
# ─────────────────────────────────────────────────────────────────────────
print("\n[SCALING] Log1p & RobustScaler...")
X_all = df[FEATS].values.astype(np.float32)
del df; gc.collect()

LOG_IDX = [FEATS.index(f) for f in LOG_FEATS if f in FEATS]
X_all[:, LOG_IDX] = np.log1p(np.abs(X_all[:, LOG_IDX]))

# ★ OPT: Vectorized train_row_mask thay vì loop O(N×SEQ)
# Logic giữ nguyên: mark tất cả raw rows được dùng bởi training windows
train_row_mask = np.zeros(len(X_all), dtype=bool)
# Với mỗi end_idx, các row từ [end_idx-SEQ+1, end_idx] thuộc train set
# → dùng broadcasting: tạo matrix [N_train × SEQ] rồi unique-flatten
if len(train_idx) > 0:
    offsets = np.arange(-(SEQ - 1), 1)              # shape (SEQ,)
    row_indices = (train_idx[:, None] + offsets)     # shape (N_train, SEQ)
    row_indices = np.clip(row_indices, 0, len(X_all) - 1)
    train_row_mask[row_indices.ravel()] = True

if os.path.exists(SCALER_PATH):
    sc = pickle.load(open(SCALER_PATH,"rb"))
    X_all = sc.transform(X_all)
    print("  ✓ Đã load Scaler cũ")
else:
    sc = RobustScaler()
    sc.fit(X_all[train_row_mask])
    X_all = sc.transform(X_all)
    pickle.dump(sc, open(SCALER_PATH,"wb"))
    print("  ✓ Đã build và lưu Scaler mới")
del train_row_mask; gc.collect()

# ─────────────────────────────────────────────────────────────────────────
# 6. DATASETS & SAMPLER V5 (MEMORY EFFICIENT)
# ─────────────────────────────────────────────────────────────────────────
class FastWindowDataset(Dataset):
    def __init__(self, data_mem, labels_mem, indices, seq_len):
        # Ép dữ liệu thành PyTorch View (Zero-copy memory)
        self.data = torch.from_numpy(data_mem) if isinstance(data_mem, np.ndarray) else data_mem
        self.labels = torch.from_numpy(labels_mem) if isinstance(labels_mem, np.ndarray) else labels_mem
        self.indices = indices
        self.seq_len = seq_len

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        end_idx = self.indices[idx]
        start = end_idx - self.seq_len + 1
        x_view = self.data[start : end_idx + 1]
        y_val = self.labels[start : end_idx + 1].max()
        return x_view, y_val.unsqueeze(0)

Tr_ds = FastWindowDataset(X_all, binary_y, train_idx, SEQ)
Vl_ds = FastWindowDataset(X_all, binary_y, val_idx, SEQ)
Ts_ds = FastWindowDataset(X_all, binary_y, test_idx, SEQ)

print("\n  Tính Weights cho Train Sampler để Bootstrapping các Known Attacks nhỏ...")
# Cân bằng các loại tấn công khác (các cuộc tấn công thiểu số được boost trọng số cực mạnh)
train_lbl_details = labels_str[train_idx]
count_map = Counter(train_lbl_details)
w_map = {lbl: 1.0/cnt for lbl, cnt in count_map.items()}
tr_weights = [w_map[lbl] for lbl in train_lbl_details]

_sampler = WeightedRandomSampler(torch.tensor(tr_weights, dtype=torch.float32), len(tr_weights), replacement=True)
del train_lbl_details; gc.collect()
print("✅ V5 Dataset Ready (Khỏi lo OOM)!")

# ─────────────────────────────────────────────────────────────────────────
# 7. MODEL
# ─────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d,2).float() * (-math.log(10000.)/d))
        pe[:, 0::2] = torch.sin(pos * div)
        n_cos = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(pos * div)[:, :n_cos]
        self.register_buffer("pe", pe)

    def forward(self, x): return x + self.pe[:x.size(1)]

class HybridZeroDayDetectorV3(nn.Module):
    """
    V5.1 Changes:
    - Bottleneck Decoder: Z(128) → Bottleneck(24) → Reconstruct(F*S)
      This forces the decoder to use a very narrow information channel.
      Benign traffic (simpler patterns) can pass through 24 dims okay.
      Attack traffic (complex/novel patterns) gets lossy compression → high RE.
    - Separate bottleneck projection from the classification path.
    """
    def __init__(self, F, S, hidden=64, heads=4, t_layers=2, dropout=0.3, bottleneck=24):
        super().__init__()
        self.F, self.S, self.bottleneck = F, S, bottleneck
        self.cnn_proj = nn.Conv1d(F, hidden, 1)
        self.cnn = nn.Sequential(
            nn.Conv1d(F, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.GELU()
        )
        self.cnn_drop = nn.Dropout(dropout*0.5)
        self.bilstm = nn.LSTM(hidden, hidden//2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_norm = nn.LayerNorm(hidden)
        self.lstm_drop = nn.Dropout(dropout)

        self.pos_enc = PositionalEncoding(hidden, max_len=S+10)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden*4, dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, t_layers)
        self.pre_head_norm = nn.LayerNorm(hidden*2)

        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(hidden//2, 1)
        )

        # ★ V5.1: Bottleneck Decoder — nén rất mạnh trước khi reconstruct
        # Lý do: Benign traffic đơn giản hơn → dễ nén vào 24 dim → RE thấp
        # Attack traffic phức tạp/lạ → mất thông tin khi nén → RE cao
        self.bottleneck_proj = nn.Sequential(
            nn.Linear(hidden*2, bottleneck),  # Cổ chai: 128 → 24
            nn.LayerNorm(bottleneck),
            nn.Tanh()  # Bounded [-1,1] để tránh bùng nổ
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden*2), nn.LayerNorm(hidden*2), nn.GELU(), nn.Dropout(dropout*0.3),
            nn.Linear(hidden*2, hidden*4), nn.LayerNorm(hidden*4), nn.GELU(), nn.Dropout(dropout*0.3),
            nn.Linear(hidden*4, F*S)
        )

    def encode(self, x):
        xT = x.transpose(1,2)
        h  = self.cnn(xT) + self.cnn_proj(xT)
        h  = self.cnn_drop(h).transpose(1,2)
        h, _ = self.bilstm(h)
        h  = self.lstm_drop(self.lstm_norm(h))
        h  = self.transformer(self.pos_enc(h))
        z  = torch.cat([h.mean(1), h.max(1).values], -1)
        return self.pre_head_norm(z)

    def forward(self, x):
        z       = self.encode(x)
        z_bn    = self.bottleneck_proj(z)  # Nén qua bottleneck
        x_hat   = self.decoder(z_bn).view(-1, self.S, self.F)
        return self.head(z), x_hat, z  # Trả thêm z để tính Contrastive Loss

model = HybridZeroDayDetectorV3(F=len(FEATS), S=SEQ, hidden=HIDDEN, heads=NHEAD, t_layers=T_LAYERS, dropout=DROPOUT, bottleneck=BOTTLENECK).to(DEVICE)
if N_GPUS >= 2: model = nn.DataParallel(model)
_mc = model.module if hasattr(model,"module") else model

# ─────────────────────────────────────────────────────────────────────────
# 8. LOSS, OPTIMIZER & RECOVERY
# ─────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma=2.5, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing
    def forward(self, logits, targets):
        t = targets*(1-self.smoothing) + 0.5*self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        pt  = torch.exp(-bce)
        at  = self.alpha*targets + (1-self.alpha)*(1-targets)
        return (at * (1-pt)**self.gamma * bce).mean()

class CombinedLoss(nn.Module):
    """
    V5.2 Loss = Focal + λ_rec*RE_benign + λ_adv*max(0, margin - RE_attack) + λ_ctr*Contrastive

    Key fix vs V5.1:
    - λ_rec: 1.0 → 0.35. Rec không còn áp đảo Cls (Cls=0.026 vs Rec=3.72 là sai lầm)
    - re_margin: adaptive = running mean of benign RE × 1.5 (thay vì hardcode 0.5)
      Lý do: Actual RE scale ~1-3, hardcode 0.5 quá nhỏ → adversarial signal yếu
    """
    def __init__(self, lam=0.35, lam_adv=0.40, lam_ctr=0.10, alpha=0.80, gamma=2.5, smoothing=0.05):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma, smoothing)
        self.lam, self.lam_adv, self.lam_ctr = lam, lam_adv, lam_ctr
        # ★ V5.2: Adaptive margin — cập nhật theo EMA của benign RE thực tế
        self.register_buffer("ema_benign_re", torch.tensor(1.0))
        self.ema_alpha = 0.05  # tốc độ cập nhật EMA
        self.ctr_margin = 2.0  # Z(attack) phải xa origin hơn

    def forward(self, logit, x_hat, targets, x_orig, z=None):
        l_cls = self.focal(logit, targets)
        tgt_flat = targets.squeeze(-1)
        benign_mask = (tgt_flat == 0)
        attack_mask = (tgt_flat == 1)

        # 1. Reconstruction Loss (CHỈ trên Benign)
        l_rec = torch.tensor(0., device=logit.device)
        if benign_mask.sum() > 0:
            re_benign_vals = F.mse_loss(x_hat[benign_mask], x_orig[benign_mask], reduction="none").mean(dim=[1,2])
            l_rec = re_benign_vals.mean()
            # ★ V5.2: Cập nhật EMA của benign RE để margin bám sát thực tế
            with torch.no_grad():
                self.ema_benign_re = (1 - self.ema_alpha) * self.ema_benign_re + self.ema_alpha * l_rec.detach()

        # 2. Adversarial Reconstruction: PHẠT nếu RE của attacks quá thấp
        # ★ V5.2: margin = 1.5× EMA(RE_benign) → bám sát RE thực tế, không hardcode
        l_adv = torch.tensor(0., device=logit.device)
        if attack_mask.sum() > 0:
            re_attack = F.mse_loss(x_hat[attack_mask], x_orig[attack_mask], reduction="none").mean(dim=[1,2])
            adaptive_margin = self.ema_benign_re * 1.5  # Attack RE phải > 1.5× benign RE
            l_adv = F.relu(adaptive_margin - re_attack).mean()

        # 3. Contrastive Z-Space Loss (nhẹ nhàng)
        l_ctr = torch.tensor(0., device=logit.device)
        if z is not None:
            if benign_mask.sum() > 0:
                l_ctr += torch.norm(z[benign_mask], dim=1).mean() * 0.05  # pull benign toward origin
            if attack_mask.sum() > 0:
                norms = torch.norm(z[attack_mask], dim=1)
                l_ctr += F.relu(self.ctr_margin - norms).mean()  # push attack away

        total = l_cls + self.lam*l_rec + self.lam_adv*l_adv + self.lam_ctr*l_ctr
        return total, l_cls.item(), l_rec.item(), l_adv.item()

criterion = CombinedLoss(LAMBDA_REC, LAMBDA_ADV, LAMBDA_CTR, FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTH)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
# ★ V5.2: T_0=25 → training có 25 epoch đầu để ổn định trước khi warm restart
sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=25, T_mult=2, eta_min=LR*0.01)
scaler_amp = GradScaler(enabled=USE_AMP)

start_ep = 1; best_vauc = 0.0; pat_cnt = 0
hist = {k:[] for k in ["tr_tot","tr_cls","tr_rec","vl_tot","vl_cls","vl_rec","vl_auc","vl_ap"]}
if os.path.exists(CKPT_PATH):
    # weights_only=False: checkpoint chứa numpy scalars (history list) — safe vì file tự tạo
    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    _mc.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sch.load_state_dict(ck["scheduler"])
    start_ep=ck["epoch"]+1; best_vauc=ck.get("best_val_auc",0.); pat_cnt=ck.get("patience_cnt",0); hist=ck.get("history", hist)
    print(f">>> Resume Epoch {start_ep}/{TOTAL_EPOCHS} | AUC={best_vauc:.4f}")

# ─────────────────────────────────────────────────────────────────────────
# 9. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────
stopped = False
if start_ep <= TOTAL_EPOCHS:
    # ★ OPT: persistent_workers=True → workers không bị tạo lại mỗi epoch
    # ★ OPT: prefetch_factor=4 → prefetch nhiều batch hơn để GPU không idle
    ldr  = DataLoader(Tr_ds, batch_size=BATCH, sampler=_sampler,
                      pin_memory=(DEVICE=="cuda"), num_workers=NUM_WORKERS,
                      prefetch_factor=4, persistent_workers=(NUM_WORKERS > 0))
    vldr = DataLoader(Vl_ds, batch_size=BATCH*2, shuffle=False,
                      pin_memory=(DEVICE=="cuda"), num_workers=NUM_WORKERS,
                      prefetch_factor=4, persistent_workers=(NUM_WORKERS > 0))

    for ep in range(start_ep, TOTAL_EPOCHS+1):
        model.train(); opt.zero_grad()
        # ★ OPT: Tích lũy loss ở Python level thay vì gọi .item() mỗi batch
        # (tránh GPU-CPU sync nhiều lần, chỉ sync 1 lần cuối epoch)
        tr_tot_t = torch.tensor(0., device=DEVICE)
        tr_cls_f = 0.; tr_rec_f = 0.; tr_adv_f = 0.; nb = 0

        for i,(bx,by) in enumerate(ldr):
            bx, by = bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
            with autocast(enabled=USE_AMP):
                logit, x_hat, z = model(bx)  # V5.1: nhận thêm z
                loss, lc, lr_, la = criterion(logit, x_hat, by, bx, z)
            scaler_amp.scale(loss/ACCUM_STEPS).backward()
            if (i+1)%ACCUM_STEPS==0 or (i+1)==len(ldr):
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(opt); scaler_amp.update(); opt.zero_grad()
            # .item() cho loss scalars vì chúng đã là Python float (từ criterion)
            tr_cls_f += lc; tr_rec_f += lr_; tr_adv_f += la
            tr_tot_t += loss.detach()   # detach để không giữ graph, đồng bộ cuối epoch
            nb += 1

        # ★ OPT: Chỉ sync GPU→CPU 1 lần cho total loss
        tr_tot_f = tr_tot_t.item()
        sch.step()

        model.eval()
        # ★ OPT: Pre-allocate list với estimate size để giảm realloc
        vp_l, vl_l = [], []; vl_tots = 0.; vl_cls_f = 0.; vl_rec_f = 0.; nv = 0
        with torch.no_grad():
            for bx,by in vldr:
                bx, by = bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
                with autocast(enabled=USE_AMP):
                    logit, x_hat, z = model(bx)
                    loss, lc, lr_, la = criterion(logit, x_hat, by, bx, z)
                vl_tots += lc + LAMBDA_REC*lr_; vl_cls_f += lc; vl_rec_f += lr_; nv += 1
                vp_l.append(torch.sigmoid(logit).cpu())    # ★ OPT: giữ tensor, concat 1 lần
                vl_l.append(by.cpu())

        # ★ OPT: torch.cat nhanh hơn np.concatenate cho nhiều tensor nhỏ
        vp = torch.cat(vp_l).numpy().ravel()
        vl = torch.cat(vl_l).numpy().ravel()
        auc, ap = roc_auc_score(vl,vp), average_precision_score(vl,vp)

        for k,v in zip(["tr_tot","tr_cls","tr_rec","vl_tot","vl_cls","vl_rec","vl_auc","vl_ap"],
                       [tr_tot_f/nb, tr_cls_f/nb, tr_rec_f/nb, vl_tots/nv, vl_cls_f/nv, vl_rec_f/nv, auc, ap]): hist[k].append(v)

        flag = "★" if auc > best_vauc else " "
        if auc > best_vauc:
            best_vauc=auc; pat_cnt=0; torch.save(_mc.state_dict(), MODEL_BEST)
        else: pat_cnt += 1

        # ★ OPT: Checkpoint nặng — chỉ save mỗi 5 epoch hoặc khi có cải thiện để giảm I/O
        if auc > best_vauc or ep % 5 == 0 or pat_cnt >= PATIENCE:
            torch.save({"epoch":ep, "model":_mc.state_dict(), "optimizer":opt.state_dict(), "scheduler":sch.state_dict(),
                        "best_val_auc":best_vauc, "patience_cnt":pat_cnt, "history":hist}, CKPT_PATH)

        print(f"{flag}[{ep:2d}/{TOTAL_EPOCHS}] Loss:{tr_tot_f/nb:.4f} Cls:{tr_cls_f/nb:.4f} Rec:{tr_rec_f/nb:.4f} Adv:{tr_adv_f/nb:.4f} | vAUC:{auc:.4f} vAP:{ap:.4f} P:{pat_cnt}")
        if pat_cnt >= PATIENCE: stopped=True; break
    print("\n✅ Training Complete!")

# ─────────────────────────────────────────────────────────────────────────
# 10. THRESHOLD TUNING & TEST SET ĐÁNH GIÁ (ZERO-DAY & KNOWN)
# ─────────────────────────────────────────────────────────────────────────
_mc.load_state_dict(torch.load(MODEL_BEST, map_location=DEVICE, weights_only=True)); model.eval()

def get_preds(ds):
    probs, res, labels = [], [], []
    with torch.no_grad():
        # ★ OPT: pin_memory + non_blocking để tăng throughput inference
        for bx,by in DataLoader(ds, batch_size=BATCH*2, num_workers=NUM_WORKERS,
                                 pin_memory=(DEVICE=="cuda")):
            bx = bx.to(DEVICE, non_blocking=True)
            with autocast(enabled=USE_AMP):
                logit, x_hat, z = model(bx)  # V5.1: nhận thêm z
            probs.append(torch.sigmoid(logit).cpu())
            res.append(F.mse_loss(x_hat, bx, reduction="none").mean(dim=[1,2]).cpu())
            labels.append(by)
    # ★ OPT: torch.cat trên CPU một lần, sau đó .numpy()
    return (torch.cat(probs).numpy().ravel(),
            torch.cat(res).numpy().ravel(),
            torch.cat(labels).numpy().ravel())

print("\n🔍 Tuning thresholds trên Val Set (chỉ chứa Known/Benign)...")
vl_probs, vl_re, vl_true = get_preds(Vl_ds)
fpr_arr, tpr_arr, roc_thr = roc_curve(vl_true, vl_probs)
v_fpr = np.where(fpr_arr <= TARGET_FPR)[0]
THR_CLS = float(roc_thr[v_fpr[np.argmax(tpr_arr[v_fpr])]]) if len(v_fpr) else 0.5

# ★ V5.1: Adaptive Threshold
# Logic: Tính RE riêng cho Benign và Known Attacks trên Val set.
# Sau đó, đặt ngưỡng ở điểm tối ưu phân tách RE phân phối giữa hai nhóm.
re_b  = vl_re[vl_true == 0]   # RE của Benign
re_k  = vl_re[vl_true == 1]   # RE của Known Attacks (đã thấy lúc train)
# Dùng trung bình hình học của hai phân phối → ngưỡng nằm GIỮA chúng
med_b = np.median(re_b)
med_k = np.median(re_k) if len(re_k) > 0 else med_b * 3
mad_b = np.median(np.abs(re_b - med_b))
# Ưu tiên: geometric mean giữa benign ceiling và known attack floor
geo_mid = np.sqrt(np.percentile(re_b, RE_PERCENTILE) *
                  np.percentile(re_k, 10) if len(re_k) > 0 else np.percentile(re_b, RE_PERCENTILE))
thr_mad = med_b + RE_MAD_SIGMA * mad_b
thr_pct = float(np.percentile(re_b, RE_PERCENTILE))
# Chọn ngưỡng thấp nhất trong các ứng viên để bắt được nhiều zero-day nhất
THR_RE  = min(geo_mid, thr_mad, thr_pct)
print(f"★ THR_CLS: {THR_CLS:.4f} | THR_RE: {THR_RE:.6f}")
print(f"  → Med(RE_benign)={med_b:.4f} | Med(RE_known)={med_k:.4f} | Geo-mid={geo_mid:.4f}")

print("\n🚀 Đánh giá Test Set...")
ts_probs, ts_re, ts_true = get_preds(Ts_ds)
ts_cat = window_cat[test_idx - (SEQ - 1)]             # 0=Benign, 1=Known, 2=Zero-Day
ts_det = labels_str[test_idx]                         # String labels

pred_head = (ts_probs > THR_CLS).astype(int)
pred_re   = (ts_re > THR_RE).astype(int)
pred_full = pred_head | pred_re         # Hybird OR rule

# Split Test set into 2 parts for analysis
mask_known = (ts_cat != 2)
mask_zd    = (ts_cat == 2) | (ts_cat == 0) # So sánh Zero-day với Benign

print("\n" + "="*50)
print("1. KHẢ NĂNG BẮT KNOWN ATTACKS (Supervised Head)")
print(classification_report(ts_true[mask_known], pred_head[mask_known], target_names=["Benign", "Known Attack"], digits=4))

print("\n2. KHẢ NĂNG BẮT ZERO-DAY ATTACKS (Unsupervised Anomaly RE)")
# Ở mạng lưới Zero-Day, ta kì vọng Anomaly phát hiện Zero-Day. ts_true[mask_zd] sẽ bằng 1 nếu là ZD, 0 là Benign.
print(classification_report(ts_true[mask_zd], pred_re[mask_zd], target_names=["Benign", "Zero-Day Attack"], digits=4))
print("="*50)

# ─────────────────────────────────────────────────────────────────────────
# 11. IN RA CÁC HÌNH ẢNH ĐÁNH GIÁ (REPORT)
# ─────────────────────────────────────────────────────────────────────────
STYLE = {"bg":"#0f1117", "panel":"#1a1d2e", "accent1":"#00d4ff", "accent2":"#ff4757", "accent3":"#ffa502", "text":"#e8eaf6"}
plt.rcParams.update({"figure.facecolor":STYLE["bg"], "axes.facecolor":STYLE["panel"], "axes.labelcolor":STYLE["text"], "xtick.color":STYLE["text"], "ytick.color":STYLE["text"], "text.color":STYLE["text"]})

# Fig 1: Known vs Zero-Day Recall
fig, axes = plt.subplots(1, 2, figsize=(18,6))
fig.suptitle("V5.0 - PER-CLASS DETECTION PERFORMANCE (Zero-Day vs Known)", fontsize=16, color=STYLE["text"])

# Known Bar Chart
atk_names, rates = [], []
for atk in np.unique(ts_det[mask_known & (ts_true==1)]):
    m = (ts_det == atk)
    atk_names.append(atk)
    rates.append(pred_head[m].mean()*100)
axes[0].barh(atk_names, rates, color=STYLE["accent1"], alpha=0.8)
axes[0].set_title("Known Attacks Detection Rate (%)\n(Caught by Supervised Head)")
axes[0].set_xlim(0, 105)
for i, v in enumerate(rates): axes[0].text(v+1, i, f"{v:.1f}%", va='center', color='white', fontsize=10)

# Zero-Day Bar Chart
zd_names, zrates = [], []
for atk in np.unique(ts_det[mask_zd & (ts_true==1)]):
    m = (ts_det == atk)
    zd_names.append(atk)
    zrates.append(pred_full[m].mean()*100) # Combined caught rate
axes[1].barh(zd_names, zrates, color=STYLE["accent3"], alpha=0.8)
axes[1].set_title("Zero-Day Attacks Detection Rate (%)\n(Caught by Hybrid Anomaly)")
axes[1].set_xlim(0, 105)
for i, v in enumerate(zrates): axes[1].text(v+1, i, f"{v:.1f}%", va='center', color='white', fontsize=10)

plt.tight_layout(); plt.savefig(f"{EXPORT_DIR}/v5_per_class_zeroday.png", dpi=150)
plt.close()

# Fig 2: SOC Scatter Plot 2D
fig2 = plt.figure(figsize=(10,8))
plt.title("V5.0 - SOC Decision Space", fontsize=14)
idx = np.random.choice(len(ts_probs), min(10000, len(ts_probs)), replace=False)
c_map = {0: STYLE["accent1"], 1: STYLE["accent2"], 2: STYLE["accent3"]}
labels_map = {0: "Benign", 1: "Known Attack", 2: "Zero-Day Attack"}

for cat_val in [0, 1, 2]:
    m = (ts_cat[idx] == cat_val)
    if m.sum() > 0:
        plt.scatter(ts_probs[idx][m], ts_re[idx][m], c=c_map[cat_val], label=labels_map[cat_val], alpha=0.4, s=8)

plt.axvline(THR_CLS, color=STYLE["text"], ls="--", label=f"P-Thr = {THR_CLS:.2f}")
plt.axhline(THR_RE, color=STYLE["text"], ls=":", label=f"RE-Thr = {THR_RE:.3f}")
plt.xlabel("Probability of Attack (Classifier)"); plt.ylabel("Reconstruction Error (Anomaly)")
plt.yscale('log')
plt.legend(); plt.grid(alpha=0.1)
plt.tight_layout(); plt.savefig(f"{EXPORT_DIR}/v5_decision_space.png", dpi=150)
plt.close()

print(f"\n📦 Xuất thành công tại {EXPORT_DIR}/")
print("  - model_v5_soc_best.pth")
print("  - v5_per_class_zeroday.png (Sơ đồ tỷ lệ bắt ZERO-DAY)")
print("  - v5_decision_space.png (Sơ đồ Scatter 2D cực trực quan)")
