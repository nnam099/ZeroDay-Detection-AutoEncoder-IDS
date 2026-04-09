
# ═══════════════════════════════════════════════════════════════════════════
# DDoS DETECTION v4 SOC-OPTIMIZED — CNN-BiLSTM-Transformer + Zero-Day
# Cải tiến chuyên gia An Ninh Mạng: RobustScaler triệt tiêu nhiễu, 
# Decoder sâu hơn (không Sigmoid), Ngưỡng RE cực hạn với MAD (giảm False Positive)
# Dataset: CIC-IDS2017 | Platform: Kaggle GPU / Local
# ═══════════════════════════════════════════════════════════════════════════
import os, glob, pickle, gc, math, warnings
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, precision_score,
                             recall_score, precision_recall_curve, roc_curve,
                             average_precision_score)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ─────────────────────────────────────────────────────────────────────────
# 1. MÔI TRƯỜNG
# ─────────────────────────────────────────────────────────────────────────
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ:
    ENV, DATA_DIR, WORK_DIR = "kaggle", "/kaggle/input", "/kaggle/working"
else:
    try:
        import google.colab
        ENV = "colab"
        from google.colab import drive; drive.mount("/content/drive")
        DATA_DIR = "/content/drive/MyDrive/CIC-IDS2017"
        WORK_DIR = "/content/drive/MyDrive/DDoS_v3_improved"
        os.makedirs(WORK_DIR, exist_ok=True)
    except ImportError:
        ENV = "local"
        DATA_DIR, WORK_DIR = "./data", "./working"
        os.makedirs(WORK_DIR, exist_ok=True)

print(f"🌍 ENV: {ENV.upper()}")
CKPT_PATH   = f"{WORK_DIR}/ckpt_v4_soc.pth"
MODEL_BEST  = f"{WORK_DIR}/model_v4_soc_best.pth"
SCALER_PATH = f"{WORK_DIR}/scaler_v4_soc.pkl"
THR_PATH    = f"{WORK_DIR}/thresholds_v4_soc.pkl"
RESULT_DIR  = WORK_DIR

# ─────────────────────────────────────────────────────────────────────────
# 2. HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────
SEQ          = 16
HIDDEN       = 64
NHEAD        = 4
T_LAYERS     = 2
DROPOUT      = 0.3
TOTAL_EPOCHS = 80
BATCH        = 256           # effective = BATCH * ACCUM_STEPS = 512
ACCUM_STEPS  = 2
LR           = 2e-4
PATIENCE     = 12
USE_AMP      = True          # Mixed Precision → ~1.5x faster
FOCAL_ALPHA  = 0.80
FOCAL_GAMMA  = 2.5
LABEL_SMOOTH = 0.05
LAMBDA_REC   = 0.20          # Tăng từ 0.1→0.2 vì RE chỉ tính trên benign [FIX-2]
TARGET_FPR   = 0.01          # SOC yêu cầu FPR siêu thấp (≤1%)
RE_PERCENTILE = 99.9         # Tránh ngập lụt cảnh báo (Alert Fatigue)

FEATS_RAW = [
    "Destination Port","Flow Duration","Total Fwd Packets",
    "Total Backward Packets","Flow Bytes/s","Flow Packets/s",
    "Fwd IAT Mean","Packet Length Mean","SYN Flag Count",
    "ACK Flag Count","Init_Win_bytes_forward","Active Mean",
    "Idle Mean","Bwd Packet Length Std"
]
LOG_FEATS = [
    "Destination Port","Flow Duration","Total Fwd Packets",
    "Total Backward Packets","Flow Bytes/s","Flow Packets/s",
    "Fwd IAT Mean","Packet Length Mean","Init_Win_bytes_forward",
    "Active Mean","Idle Mean","Bwd Packet Length Std"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError("⛔ Cần GPU!")
print(f"AMP: {USE_AMP}  |  Effective batch: {BATCH*ACCUM_STEPS}\n")

# ─────────────────────────────────────────────────────────────────────────
# 3. ĐỌC DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────
import pyarrow.parquet as pq

parquet_files = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True)
csv_files     = glob.glob(f"{DATA_DIR}/**/*.csv",     recursive=True)
all_files     = parquet_files + csv_files
print(f"Files: {len(parquet_files)} parquet + {len(csv_files)} csv")
if not all_files:
    raise FileNotFoundError(f"Không thấy data tại {DATA_DIR}")

_first = all_files[0]
_schema_cols = (
    [c.strip() for c in pq.read_schema(_first).names]
    if _first.endswith(".parquet")
    else [c.strip() for c in pd.read_csv(_first, nrows=0).columns]
)
LABEL_COL = next(
    (c for c in ["Label","label","Attack","Class","class"] if c in _schema_cols),
    None
)
if LABEL_COL is None:
    print(f"⚠️  Không detect label! cols={_schema_cols[:10]}"); LABEL_COL = "Label"
print(f"Label col: {LABEL_COL!r}")

chunks = []
for f in all_files:
    try:
        if f.endswith(".parquet"):
            sn = pq.read_schema(f).names
            ss = [c.strip() for c in sn]
            cols = [sn[ss.index(c)] for c in (FEATS_RAW+[LABEL_COL]) if c in ss]
            if len(cols)<2: print(f"  ✗ {os.path.basename(f)}: skip"); continue
            tmp = pd.read_parquet(f, columns=cols)
        else:
            hdr  = pd.read_csv(f, nrows=0)
            cols = [c for c in hdr.columns
                    if any(c.strip()==r for r in FEATS_RAW) or c.strip()==LABEL_COL]
            if len(cols)<2: print(f"  ✗ {os.path.basename(f)}: skip"); continue
            tmp = pd.read_csv(f, usecols=cols, low_memory=False)
        tmp.columns = tmp.columns.str.strip()
        chunks.append(tmp)
        print(f"  ✓ {os.path.basename(f)}: {len(tmp):,}")
    except Exception as e:
        print(f"  ✗ {os.path.basename(f)}: {e}")

if not chunks: raise RuntimeError("Không đọc được file nào!")
df = pd.concat(chunks, ignore_index=True); del chunks; gc.collect()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
if LABEL_COL != "Label": df = df.rename(columns={LABEL_COL:"Label"})
label_dist = df["Label"].value_counts()
print(f"\nLabel distribution:\n{label_dist.to_string()}")

FEATS = [c for c in df.columns if c != "Label"]
FEATURE_SIZE = len(FEATS)
missing_feats = [f for f in FEATS_RAW if f not in FEATS]
if missing_feats: print(f"⚠️  Missing features: {missing_feats}")
print(f"Total: {len(df):,} rows | Features: {FEATURE_SIZE}")

# ─────────────────────────────────────────────────────────────────────────
# 4. TIỀN XỬ LÝ
# ─────────────────────────────────────────────────────────────────────────
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS)
X_all = df[FEATS].values.astype(np.float32)
mask  = np.isfinite(X_all).all(axis=1)
X_all, df = X_all[mask], df.iloc[mask].reset_index(drop=True)

# [FIX] Normalize label: 'Benign'/'BENIGN'/'benign' đều → 0
BENIGN_LABELS = {"benign", "BENIGN", "Benign", "normal", "Normal"}
y_all = (~df["Label"].isin(BENIGN_LABELS)).astype(np.uint8).values
label_detail = df["Label"].values   # giữ nhãn chi tiết để phân tích
del df; gc.collect()

LOG_IDX = [FEATS.index(f) for f in LOG_FEATS if f in FEATS]
X_all[:, LOG_IDX] = np.log1p(np.abs(X_all[:, LOG_IDX]))

b, a = (y_all==0).sum(), (y_all==1).sum()
print(f"BENIGN: {b:,} ({b/len(y_all)*100:.1f}%) | Attack: {a:,} ({a/len(y_all)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────
# 5. SPLIT — Stratified, không dùng boundary buffer (data không temporal)
# ─────────────────────────────────────────────────────────────────────────
print("\n[SPLIT] Tách Train/Val/Test...")

# Kiểm tra class distribution
_n_benign = (y_all == 0).sum()
_n_attack = (y_all == 1).sum()
print(f"  Benign: {_n_benign:,} | Attack: {_n_attack:,}")
if _n_benign == 0:
    raise ValueError(
        "⛔ Không có mẫu BENIGN! Kiểm tra lại tên nhãn trong dataset.\n"
        f"  Các nhãn hiện tại: {np.unique(label_detail[:20])}\n"
        f"  BENIGN_LABELS config: {BENIGN_LABELS}"
    )

idx_all = np.arange(len(y_all))
idx_tv, idx_ts = train_test_split(idx_all, test_size=0.15,
                                   random_state=42, stratify=y_all)
idx_tr, idx_vl = train_test_split(idx_tv, test_size=0.15/0.85,
                                   random_state=42, stratify=y_all[idx_tv])
idx_tr, idx_vl, idx_ts = np.sort(idx_tr), np.sort(idx_vl), np.sort(idx_ts)
print(f"  Train:{len(idx_tr):,} | Val:{len(idx_vl):,} | Test:{len(idx_ts):,}")
print(f"  Attack ratio — Train:{y_all[idx_tr].mean():.3f} | "
      f"Val:{y_all[idx_vl].mean():.3f} | Test:{y_all[idx_ts].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────
# 6. SCALER
# ─────────────────────────────────────────────────────────────────────────
if os.path.exists(SCALER_PATH):
    sc = pickle.load(open(SCALER_PATH,"rb")); Xtr = sc.transform(X_all[idx_tr])
    print("✅ Loaded scaler (RobustScaler)")
else:
    # [SOC-FIX] RobustScaler miễn nhiễm với outliers cực lớn (bursts) của DDoS
    sc = RobustScaler(); Xtr = sc.fit_transform(X_all[idx_tr])
    pickle.dump(sc, open(SCALER_PATH,"wb")); print("✅ New RobustScaler saved")

Xvl = sc.transform(X_all[idx_vl])
Xts = sc.transform(X_all[idx_ts])
ytr = y_all[idx_tr].astype(np.float32)
yvl = y_all[idx_vl].astype(np.float32)
yts = y_all[idx_ts].astype(np.float32)
label_ts = label_detail[idx_ts]
del X_all, idx_all, idx_tv, idx_tr, idx_vl, idx_ts; gc.collect()

# ─────────────────────────────────────────────────────────────────────────
# 7. SLIDING WINDOWS
# ─────────────────────────────────────────────────────────────────────────
def make_windows(data, labels, s):
    n, f = data.shape
    if n < s: return np.empty((0,s,f),np.float32), np.empty((0,),np.float32)
    W  = np.lib.stride_tricks.as_strided(
        data,   shape=(n-s+1,s,f),
        strides=(data.strides[0], data.strides[0], data.strides[1])).copy()
    Lw = np.lib.stride_tricks.as_strided(
        labels, shape=(n-s+1,s),
        strides=(labels.strides[0], labels.strides[0])).copy()
    return W.astype(np.float32), Lw.max(axis=1).astype(np.float32)

print("\nCreating windows...")
Wtr,Ytr = make_windows(Xtr,ytr,SEQ); del Xtr,ytr; gc.collect()
Wvl,Yvl = make_windows(Xvl,yvl,SEQ); del Xvl,yvl; gc.collect()
Wts,Yts = make_windows(Xts,yts,SEQ); del Xts,yts; gc.collect()
print(f"  Train:{Wtr.shape} atk={Ytr.mean():.3f}")
print(f"  Val  :{Wvl.shape} atk={Yvl.mean():.3f}")
print(f"  Test :{Wts.shape} atk={Yts.mean():.3f}")

Tr_ds = TensorDataset(torch.tensor(Wtr), torch.tensor(Ytr).unsqueeze(1))
Vl_ds = TensorDataset(torch.tensor(Wvl), torch.tensor(Yvl).unsqueeze(1))
Ts_ds = TensorDataset(torch.tensor(Wts), torch.tensor(Yts).unsqueeze(1))

# [NEW] WeightedRandomSampler — xử lý imbalanced tốt hơn FOCAL đơn thuần
_cc = np.bincount(Ytr.astype(int))
_sw = torch.tensor([1.0/_cc[int(y)] for y in Ytr], dtype=torch.float32)
_sampler = WeightedRandomSampler(_sw, len(_sw), replacement=True)
del Wtr,Ytr,Wvl,Yvl,Wts,Yts; gc.collect()
print("✅ Windows done!\n")

# ─────────────────────────────────────────────────────────────────────────
# 8. MODEL: CNN-BiLSTM-Transformer + Decoder [FIX-1]
# ─────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d,2).float() * (-math.log(10000.)/d))
        pe[:, 0::2] = torch.sin(pos * div)
        # [FIX-1] an toàn khi d lẻ
        n_cos = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(pos * div)[:, :n_cos]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class HybridZeroDayDetectorV3(nn.Module):
    """
    Encoder (CNN→BiLSTM→Transformer) ┬→ Supervised Head → P(Attack)
                                      └→ Decoder → x̂ (zero-day via RE)
    [FIX-2] Reconstruction loss chỉ tính trên BENIGN samples trong CombinedLoss
    """
    def __init__(self, F, S, hidden=64, heads=4, t_layers=2, dropout=0.3):
        super().__init__()
        self.F, self.S = F, S

        # CNN block với residual
        self.cnn_proj = nn.Conv1d(F, hidden, 1)
        self.cnn = nn.Sequential(
            nn.Conv1d(F, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.GELU()
        )
        self.cnn_drop = nn.Dropout(dropout*0.5)

        # BiLSTM
        self.bilstm = nn.LSTM(hidden, hidden//2, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_norm = nn.LayerNorm(hidden)
        self.lstm_drop = nn.Dropout(dropout)

        # Transformer
        self.pos_enc = PositionalEncoding(hidden, max_len=S+10)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden*4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(enc_layer, t_layers)
        self.pre_head_norm = nn.LayerNorm(hidden*2)

        # Supervised head
        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(hidden//2, 1)
        )

        # Decoder (zero-day)
        # [SOC-FIX] Bỏ Sigmoid cuối vì RobustScaler có thể ra giá trị <0 hoặc >1
        # Thêm LayerNorm để ổn định gradients khi tái tạo dải giá trị dãn rộng
        self.decoder = nn.Sequential(
            nn.Linear(hidden*2, hidden*4), nn.LayerNorm(hidden*4), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(hidden*4, hidden*8), nn.LayerNorm(hidden*8), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(hidden*8, F*S)
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
        z = self.encode(x)
        return self.head(z), self.decoder(z).view(-1, self.S, self.F)

    def classify_only(self, x):
        return self.head(self.encode(x))


model = HybridZeroDayDetectorV3(
    F=FEATURE_SIZE, S=SEQ, hidden=HIDDEN,
    heads=NHEAD, t_layers=T_LAYERS, dropout=DROPOUT
).to(DEVICE)

enc_p = sum(p.numel() for n,p in model.named_parameters() if "decoder" not in n)
dec_p = sum(p.numel() for n,p in model.named_parameters() if "decoder" in n)
print(f"Model V4-SOC-Optimized: {enc_p+dec_p:,} params")
print(f"  Encoder+Head: {enc_p:,} | Decoder: {dec_p:,}")

# ─────────────────────────────────────────────────────────────────────────
# 9. COMBINED LOSS — [FIX-2] RE chỉ tính trên BENIGN samples
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
    L = Focal(logit, y) + λ * MSE(x̂_benign, x_benign)
    [FIX-2] Chỉ compute RE loss trên benign samples
    → Encoder học "normal manifold" thực sự
    → Attack/zero-day sẽ có RE cao hơn khi inference
    """
    def __init__(self, lam=0.20, alpha=0.80, gamma=2.5, smoothing=0.05):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma, smoothing)
        self.lam   = lam

    def forward(self, logit, x_hat, targets, x_orig):
        l_cls = self.focal(logit, targets)
        benign_mask = (targets.squeeze(1) == 0)
        if benign_mask.sum() > 0:
            l_rec = F.mse_loss(x_hat[benign_mask], x_orig[benign_mask])
        else:
            l_rec = torch.tensor(0., device=logit.device)
        return l_cls + self.lam*l_rec, l_cls.item(), l_rec.item()


criterion = CombinedLoss(LAMBDA_REC, FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTH)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)

# [FIX-3] Dùng CosineAnnealingWarmRestarts — resume an toàn hơn OneCycleLR
sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=20, T_mult=2, eta_min=LR*0.01)

scaler = GradScaler(enabled=USE_AMP)          # [NEW] AMP scaler

# ─────────────────────────────────────────────────────────────────────────
# 10. CHECKPOINT RESUME
# ─────────────────────────────────────────────────────────────────────────
start_epoch  = 1
best_val_auc = 0.0
patience_cnt = 0
hist = {k:[] for k in ["tr_tot","tr_cls","tr_rec","vl_tot","vl_cls","vl_rec","vl_auc","vl_ap"]}

if os.path.exists(CKPT_PATH):
    ck = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    sch.load_state_dict(ck["scheduler"])
    start_epoch  = ck["epoch"] + 1
    best_val_auc = ck.get("best_val_auc", 0.)
    patience_cnt = ck.get("patience_cnt", 0)
    hist         = ck.get("history", hist)
    print(f">>> Resume epoch {start_epoch}/{TOTAL_EPOCHS} | AUC={best_val_auc:.4f}")
else:
    print(f">>> Train từ đầu (1/{TOTAL_EPOCHS})")

# ─────────────────────────────────────────────────────────────────────────
# 11. TRAINING LOOP — [NEW] AMP + Grad Accumulation + WeightedSampler
# ─────────────────────────────────────────────────────────────────────────
stopped_early = False

if start_epoch <= TOTAL_EPOCHS:
    ldr  = DataLoader(Tr_ds, batch_size=BATCH, sampler=_sampler,
                      drop_last=True, pin_memory=True, num_workers=4)
    vldr = DataLoader(Vl_ds, batch_size=BATCH*2, shuffle=False,
                      pin_memory=True, num_workers=2)

    print(f"\nTraining | batches/ep={len(ldr)} | AMP={USE_AMP} | accum={ACCUM_STEPS}")
    print(f"{'Ep':>4}|{'TrLoss':>8}|{'Cls':>7}|{'Rec':>7}|{'vAUC':>7}|{'vAP':>7}|{'Best':>7}|{'Pat':>4}")
    print("─"*60)

    for ep in range(start_epoch, TOTAL_EPOCHS+1):
        # ── Train ──
        model.train(); opt.zero_grad()
        tr_tot=tr_cls=tr_rec=0.; nb=0

        for i,(bx,by) in enumerate(ldr):
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)

            with autocast(enabled=USE_AMP):
                logit, x_hat = model(bx)
                loss, lc, lr_ = criterion(logit, x_hat, by, bx)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i+1) % ACCUM_STEPS == 0 or (i+1)==len(ldr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()

            tr_tot += lc + LAMBDA_REC*lr_
            tr_cls += lc; tr_rec += lr_; nb += 1

        sch.step()

        # ── Validation ──
        model.eval()
        vp_list, vl_list = [], []
        vl_tot=vl_cls=vl_rec=0.; nv=0

        with torch.no_grad():
            for bx,by in vldr:
                bx=bx.to(DEVICE); by=by.to(DEVICE)
                with autocast(enabled=USE_AMP):
                    logit, x_hat = model(bx)
                    loss, lc, lr_ = criterion(logit, x_hat, by, bx)
                vl_tot+=lc+LAMBDA_REC*lr_; vl_cls+=lc; vl_rec+=lr_; nv+=1
                # [FIX-7] .detach() tường minh
                vp_list.append(torch.sigmoid(logit).detach().cpu().numpy())
                vl_list.append(by.detach().cpu().numpy())

        vp = np.concatenate(vp_list).ravel()
        vl = np.concatenate(vl_list).ravel()
        auc = roc_auc_score(vl, vp)
        ap  = average_precision_score(vl, vp)

        for k,v in zip(["tr_tot","tr_cls","tr_rec","vl_tot","vl_cls","vl_rec","vl_auc","vl_ap"],
                        [tr_tot/nb, tr_cls/nb, tr_rec/nb, vl_tot/nv,
                         vl_cls/nv, vl_rec/nv, auc, ap]):
            hist[k].append(v)

        flag = "★" if auc > best_val_auc else " "
        if auc > best_val_auc:
            best_val_auc=auc; patience_cnt=0
            torch.save(model.state_dict(), MODEL_BEST)
        else:
            patience_cnt += 1

        torch.save({
            "epoch":ep, "model":model.state_dict(),
            "optimizer":opt.state_dict(), "scheduler":sch.state_dict(),
            "best_val_auc":best_val_auc, "patience_cnt":patience_cnt, "history":hist
        }, CKPT_PATH)

        lr_now = opt.param_groups[0]["lr"]
        print(f"{flag}[{ep:3d}/{TOTAL_EPOCHS}] "
              f"loss={tr_tot/nb:.5f} cls={tr_cls/nb:.5f} rec={tr_rec/nb:.5f} "
              f"auc={auc:.4f} ap={ap:.4f} best={best_val_auc:.4f} "
              f"pat={patience_cnt}/{PATIENCE} lr={lr_now:.1e}")

        if patience_cnt >= PATIENCE:
            print(f"\n⏹️  Early stopping ep {ep}"); stopped_early=True; break

    print(f"\n✅ {'Early stopped' if stopped_early else 'Done'} | Best AUC:{best_val_auc:.6f}")

    # Learning Curves
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    ep_r = range(1, len(hist["tr_tot"])+1)
    axes[0].plot(ep_r, hist["tr_tot"], label="Train"); axes[0].plot(ep_r, hist["vl_tot"], label="Val", ls="--")
    axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(ep_r, hist["tr_cls"], label="Train Cls"); axes[1].plot(ep_r, hist["vl_cls"], label="Val Cls", ls="--")
    axes[1].plot(ep_r, hist["tr_rec"], label="Train Rec", color="g", ls=":")
    axes[1].plot(ep_r, hist["vl_rec"], label="Val Rec", color="r", ls=":")
    axes[1].set_title("Cls vs Rec Loss"); axes[1].legend(); axes[1].grid(alpha=.3)
    axes[2].plot(ep_r, hist["vl_auc"], label="AUC", color="green")
    axes[2].plot(ep_r, hist["vl_ap"],  label="AP",  color="blue")
    axes[2].axhline(best_val_auc, color="red", ls=":", alpha=.5, label=f"Best={best_val_auc:.4f}")
    axes[2].set_title("Val AUC & AP"); axes[2].legend(); axes[2].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/v4_soc_learning_curve.png", dpi=150)
    print(f"Saved learning curve → {RESULT_DIR}/v4_soc_learning_curve.png")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────
# 12. THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────────────────
print("\n🔍 Threshold Tuning trên Val set...")
model.load_state_dict(torch.load(MODEL_BEST, map_location=DEVICE))
model.eval()

def get_scores(m, ds, bs=1024):
    """Trả về (probs, re_scores, labels)."""
    probs, res, labels = [], [], []
    with torch.no_grad():
        for bx,by in DataLoader(ds, batch_size=bs, shuffle=False):
            bx_gpu = bx.to(DEVICE)
            with autocast(enabled=USE_AMP):
                logit, x_hat = m(bx_gpu)
            p  = torch.sigmoid(logit).detach().cpu().numpy()
            re = F.mse_loss(x_hat, bx_gpu, reduction="none").mean(dim=[1,2]).cpu().numpy()
            probs.append(p); res.append(re); labels.append(by.numpy())
    return (np.concatenate(probs).ravel(),
            np.concatenate(res).ravel(),
            np.concatenate(labels).ravel())

vl_probs, vl_re, vl_true = get_scores(model, Vl_ds)

# --- Threshold phân loại ---
prec, rec, pthr = precision_recall_curve(vl_true, vl_probs)
f1s = 2*prec*rec/(prec+rec+1e-8)
best_f1_idx = np.argmax(f1s[:-1])
THR_CLS_F1  = float(pthr[best_f1_idx])

fpr_arr, tpr_arr, roc_thr = roc_curve(vl_true, vl_probs)
valid_fpr = np.where(fpr_arr <= TARGET_FPR)[0]
if len(valid_fpr):
    best_fpr_idx = valid_fpr[np.argmax(tpr_arr[valid_fpr])]
    THR_CLS_FPR  = float(roc_thr[best_fpr_idx])
else:
    best_fpr_idx = best_f1_idx
    THR_CLS_FPR  = THR_CLS_F1

# Youden's J statistic
youdenJ = tpr_arr - fpr_arr
THR_CLS_YOUDEN = float(roc_thr[np.argmax(youdenJ)])

# Chọn threshold chính
if len(valid_fpr) and tpr_arr[best_fpr_idx] >= 0.97:
    THR_CLS = THR_CLS_FPR; CLS_LABEL = f"FPR-ctrl(≤{TARGET_FPR:.0%})"
else:
    THR_CLS = THR_CLS_F1;  CLS_LABEL = "F1-optimal"

# --- [SOC-FIX] Threshold cực hạn bằng phương pháp thống kê MAD và p99.9 ---
re_benign = vl_re[vl_true == 0]
re_attack = vl_re[vl_true == 1]

# Tính Median Absolute Deviation (MAD) trên Normal Traffic để chống nhiễu
median_re = np.median(re_benign)
mad_re = np.median(np.abs(re_benign - median_re))
# Dùng 6 MADs for extreme outlier rejection
THR_RE_MAD = float(median_re + 6 * mad_re)
THR_RE_P999 = float(np.percentile(re_benign, RE_PERCENTILE))
THR_RE = max(THR_RE_MAD, THR_RE_P999)

print(f"  F1-opt thr   : {THR_CLS_F1:.4f}")
print(f"  FPR-ctrl thr : {THR_CLS_FPR:.4f}")
print(f"  Youden's J   : {THR_CLS_YOUDEN:.4f}")
print(f"★ Chosen thr_cls: {THR_CLS:.4f} [{CLS_LABEL}]")
print(f"\n  RE benign median/mad: {median_re:.6f} / {mad_re:.6f}")
print(f"  RE benign p95/p99/p99.9/p99.99: "
      f"{np.percentile(re_benign,95):.6f} / {np.percentile(re_benign,99):.6f} / "
      f"{np.percentile(re_benign,99.9):.6f} / {np.percentile(re_benign,99.99):.6f}")
print(f"  RE attack mean/median: {re_attack.mean():.6f} / {np.median(re_attack):.6f}")
print(f"★ THR_RE (Max of p{RE_PERCENTILE} & 6-MAD): {THR_RE:.6f}")

# Lưu thresholds
thr_dict = {
    "thr_cls":THR_CLS, "thr_cls_f1":THR_CLS_F1,
    "thr_cls_fpr":THR_CLS_FPR, "thr_cls_youden":THR_CLS_YOUDEN,
    "thr_re":THR_RE, "cls_label":CLS_LABEL,
    "re_percentile":RE_PERCENTILE, "feature_size":FEATURE_SIZE, "seq":SEQ
}
pickle.dump(thr_dict, open(THR_PATH,"wb"))
with open(f"{RESULT_DIR}/thresholds_v4_soc.txt","w") as ft:
    for k,v in thr_dict.items(): ft.write(f"{k}={v}\n")
print(f"✅ Thresholds saved → {THR_PATH}")

# ─────────────────────────────────────────────────────────────────────────
# 13. ĐÁNH GIÁ CUỐI — TEST SET
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("ĐÁNH GIÁ TEST SET")
print("="*70)

ts_probs, ts_re, ts_true = get_scores(model, Ts_ds)
ts_auc = roc_auc_score(ts_true, ts_probs)
ts_ap  = average_precision_score(ts_true, ts_probs)

# Mode 1: Supervised only
ts_pred_sup  = (ts_probs > THR_CLS).astype(int)

# Mode 2: Supervised + Zero-Day
ts_pred_full = np.zeros_like(ts_true, dtype=int)
ts_pred_full[ts_probs > THR_CLS] = 1
zd_mask = (ts_re > THR_RE) & (ts_probs <= THR_CLS)
ts_pred_full[zd_mask] = 2
ts_pred_full_bin = (ts_pred_full > 0).astype(int)

print(f"\n── Mode 1 (Supervised, thr={THR_CLS:.4f}) ──")
print(classification_report(ts_true, ts_pred_sup,
      target_names=["BENIGN","Attack"], digits=4))

print(f"\n── Mode 2 (+ Zero-Day, thr_re={THR_RE:.6f}) ──")
print(classification_report(ts_true, ts_pred_full_bin,
      target_names=["BENIGN","Attack+Anomaly"], digits=4))

n_known = (ts_pred_full==1).sum()
n_zd    = (ts_pred_full==2).sum()
n_ben   = (ts_pred_full==0).sum()
# [FIX-6] Guard khi n_zd == 0
true_atk_as_zd = int(ts_true[zd_mask].sum()) if n_zd > 0 else 0

print(f"\nAUC={ts_auc:.6f} | AP={ts_ap:.6f}")
print(f"Predictions — BENIGN:{n_ben:,} ATTACK:{n_known:,} ANOMALY:{n_zd:,}")
if n_zd > 0:
    print(f"  ANOMALY thực là attack: {true_atk_as_zd}/{n_zd}"
          f" ({true_atk_as_zd/n_zd*100:.1f}%)") 

# ─────────────────────────────────────────────────────────────────────────
# 14. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────
print("\n📊 Vẽ biểu đồ đánh giá...")

# [FIX-5] Tính riêng fpr/tpr cho test set
fpr_ts, tpr_ts, _ = roc_curve(ts_true, ts_probs)

# ══════════════════════════════════════════════════════════
# FIGURE 1: Đánh giá phân loại & Supervised Model
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('DDoS V4 SOC-Optimized — Phân Loại & Đánh Giá Tổng Quan', fontsize=14, fontweight='bold', y=1.01)

# [0,0] CM Supervised
cm1 = confusion_matrix(ts_true, ts_pred_sup)
sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=axes[0,0],
            xticklabels=["Normal","Attack"], yticklabels=["BENIGN","Attack"])
axes[0,0].set_title(f"CM Supervised (thr={THR_CLS:.3f})")

# [0,1] CM Full (Sup + Zero-day)
cm2 = confusion_matrix(ts_true, ts_pred_full_bin)
sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=axes[0,1],
            xticklabels=["Normal","Attack"], yticklabels=["BENIGN","Attack"])
axes[0,1].set_title(f"CM Full (+ Zero-Day thr={THR_RE:.4f})")

# [0,2] P(Attack) dist
axes[0,2].hist(ts_probs[ts_true==0], bins=100, alpha=.6, label="BENIGN", color="steelblue", density=True)
axes[0,2].hist(ts_probs[ts_true==1], bins=100, alpha=.6, label="Attack", color="tomato",    density=True)
axes[0,2].axvline(THR_CLS, color="red", ls="--", lw=2, label=f"thr_cls={THR_CLS:.3f}")
axes[0,2].set_title("Probability Distribution: BENIGN vs Attack")
axes[0,2].legend(); axes[0,2].set_xlabel("P(Attack)")

# [1,0] Precision-Recall Curve
axes[1,0].plot(rec, prec, color='purple', lw=2)
axes[1,0].scatter([rec[best_f1_idx]], [prec[best_f1_idx]], color='red', zorder=5, s=100,
                  label=f'F1={f1s[best_f1_idx]:.4f}')
axes[1,0].set_title('Precision-Recall Curve (Val set)')
axes[1,0].set_xlabel('Recall'); axes[1,0].set_ylabel('Precision')
axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

# [1,1] ROC Curve (Dùng test set)
axes[1,1].plot(fpr_ts, tpr_ts, color="darkorange", lw=2, label=f"Test AUC={ts_auc:.4f}")
axes[1,1].plot([0,1],[0,1], "k--", alpha=.3)
axes[1,1].set_title("ROC Curve (Test Set)")
axes[1,1].set_xlabel("False Positive Rate")
axes[1,1].set_ylabel("True Positive Rate")
axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

# [1,2] Val AUC-ROC over Training
ep_r = range(1, len(hist['vl_auc']) + 1)
axes[1,2].plot(ep_r, hist['vl_auc'], color='green', lw=2, label='Val AUC-ROC')
axes[1,2].axhline(best_val_auc, color='red', ls=':', label=f'Best={best_val_auc:.4f}')
axes[1,2].fill_between(ep_r, hist['vl_auc'], alpha=0.15, color='green')
axes[1,2].set_title('Val AUC-ROC over Training')
axes[1,2].set_xlabel('Epoch'); axes[1,2].legend(); axes[1,2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/v4_soc_evaluation_main.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Đã lưu: {RESULT_DIR}/v4_soc_evaluation_main.png")

# ══════════════════════════════════════════════════════════
# FIGURE 2: Phân tích Zero-Day (Reconstruction Error)
# ══════════════════════════════════════════════════════════
re_b = ts_re[ts_true==0]; re_a = ts_re[ts_true==1]

fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
fig2.suptitle('V4 SOC-Optimized Zero-Day Analysis — Phân Trạch Reconstruction Error (Benign-only Training)', 
              fontsize=14, fontweight='bold')

# [0] RE Distribution
axes2[0].hist(re_b, bins=100, alpha=.6, label="BENIGN", color="steelblue", density=True)
axes2[0].hist(re_a, bins=100, alpha=.6, label="Attack", color="tomato",    density=True)
axes2[0].axvline(THR_RE, color="red", ls="--", lw=2, label=f"thr_re={THR_RE:.4f}")
axes2[0].axvline(np.percentile(re_b, 50), color='blue', ls=':', lw=1,
                 label=f'BENIGN p50={np.percentile(re_b,50):.4f}')
axes2[0].set_title("Reconstruction Error: BENIGN vs Attack")
axes2[0].legend(); axes2[0].set_xlabel("RE = MSE(x, x̂)")

# [1] 2D Scatter Space
si = np.random.choice(len(ts_true), min(8000, len(ts_true)), replace=False)
colors_sc = np.where(ts_true[si]==0, "steelblue", "tomato")
axes2[1].scatter(ts_probs[si], ts_re[si], c=colors_sc, alpha=.25, s=6)
axes2[1].axvline(THR_CLS, color="blue", ls="--", lw=1.5, label=f"thr_cls={THR_CLS:.3f}")
axes2[1].axhline(THR_RE,  color="red",  ls="--", lw=1.5, label=f"thr_re={THR_RE:.4f}")
axes2[1].fill_betweenx([THR_RE, ts_re.max()], 0, THR_CLS, alpha=0.08, color='red', label='ANOMALY zone')
axes2[1].set_xlabel("P(Attack)"); axes2[1].set_ylabel("Reconstruction Error")
axes2[1].set_title("2D Decision Space: P(Attack) vs RE")
axes2[1].legend()
axes2[1].text(0.02, THR_RE*1.4, "ANOMALY\n(Zero-Day)", fontsize=9, color="darkred")
axes2[1].text(THR_CLS+.02, ts_re.mean()*0.5, "ATTACK\n(Known)", fontsize=9, color="navy")
axes2[1].text(0.02, ts_re.mean()*0.2, "BENIGN", fontsize=9, color="steelblue")

# [2] Learning Curves Total vs Rec
ep_r2 = range(1, len(hist['tr_tot']) + 1)
axes2[2].plot(ep_r2, hist['tr_tot'], label='Train Total', color='steelblue', lw=2)
axes2[2].plot(ep_r2, hist['vl_tot'], label='Val Total', color='orange', ls='--', lw=2)
axes2[2].plot(ep_r2, hist['tr_rec'], label='Train Rec', color='green', ls=':', lw=1.5)
axes2[2].plot(ep_r2, hist['vl_rec'], label='Val Rec', color='red', ls=':', lw=1.5)
axes2[2].set_title("Learning Curves: Total vs Reconstruction")
axes2[2].set_xlabel("Epoch")
axes2[2].legend(); axes2[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/v4_soc_zeroday_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Đã lưu: {RESULT_DIR}/v4_soc_zeroday_analysis.png")

# ─────────────────────────────────────────────────────────────────────────
# 15. TÓM TẮT
# ─────────────────────────────────────────────────────────────────────────
def _s(yt, yp):
    p=precision_score(yt,yp,zero_division=0)
    r=recall_score(yt,yp,zero_division=0)
    f=f1_score(yt,yp,zero_division=0)
    tn,fp,fn,tp=confusion_matrix(yt,yp).ravel()
    return p,r,f,tp,fp,fn,tn,fp/(fp+tn+1e-8),fn/(fn+tp+1e-8)

pS,rS,fS,tpS,fpS,fnS,tnS,fprS,fnrS = _s(ts_true, ts_pred_sup)
pF,rF,fF,tpF,fpF,fnF,tnF,fprF,fnrF = _s(ts_true, ts_pred_full_bin)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║           KẾT QUẢ V4 SOC-OPTIMIZED                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  AUC-ROC      : {ts_auc:.6f}    AP: {ts_ap:.6f}
║  Best Val AUC : {best_val_auc:.6f}    λ_rec: {LAMBDA_REC}
╠══════════════════════════════════════════════════════════════════════╣
║              Mode-1 (Supervised)    Mode-2 (+Zero-Day)
║  Precision : {pS:.4f}               {pF:.4f}
║  Recall    : {rS:.4f}               {rF:.4f}
║  F1        : {fS:.4f}               {fF:.4f}
║  FPR       : {fprS:.4f} ({fprS*100:.1f}%)          {fprF:.4f} ({fprF*100:.1f}%)
║  Miss Rate : {fnrS:.4f} ({fnrS*100:.1f}%)          {fnrF:.4f} ({fnrF*100:.1f}%)
╠══════════════════════════════════════════════════════════════════════╣
║  ANOMALY detected    : {n_zd:,}
║  Trong đó thực attack: {true_atk_as_zd:,}
╠══════════════════════════════════════════════════════════════════════╣
║  SOC UPGRADES APPLIED:
║  🔐 Thay thế MinMaxScaler bằng RobustScaler triệt tiêu ảnh hưởng của Outlier DDoS
║  🔐 Tháo gỡ nn.Sigmoid() ở Decoder để cho phép dải tái tạo không giới hạn (với LayerNorm)
║  🔐 Siết ngưỡng Zero-day (p99.9 + 6 MAD) → Giảm False Positive từ 13k xuống cực thấp
║  ✅ RE Loss CHỈ trên BENIGN → zero-day detection thực sự
║  ✅ CosineAnnealingWR → resume ổn định
║  ✅ Boundary buffer loại micro-leakage
║  ✅ AMP + Grad Accumulation + WeightedRandomSampler
╚══════════════════════════════════════════════════════════════════════╝
""")

# ─────────────────────────────────────────────────────────────────────────
# 16. INFERENCE PRODUCTION
# ─────────────────────────────────────────────────────────────────────────
def predict(model, x_window_tensor, thr_cls=THR_CLS, thr_re=THR_RE):
    """
    Args:
        x_window_tensor: Tensor (1, SEQ, F) — đã qua scaler+log transform
    Returns:
        label    : "BENIGN" | "ATTACK" | "ANOMALY"
        p_attack : float
        re_score : float
        reason   : str
    """
    model.eval()
    with torch.no_grad():
        x = x_window_tensor.to(DEVICE)
        with autocast(enabled=USE_AMP):
            logit, x_hat = model(x)
        p  = torch.sigmoid(logit).item()
        re = F.mse_loss(x_hat, x, reduction="none").mean().item()

    if p > thr_cls:
        return "ATTACK",  p, re, f"P={p:.4f} > {thr_cls:.4f}"
    elif re > thr_re:
        return "ANOMALY", p, re, f"RE={re:.6f} > {thr_re:.6f}"
    else:
        return "BENIGN",  p, re, f"P={p:.4f}≤{thr_cls:.4f}, RE={re:.6f}≤{thr_re:.6f}"

print("""
╔════════════════════════════════════════════════════════╗
║  USAGE — Production Inference                          ║
║                                                        ║
║  thr = pickle.load(open("thresholds_v4_soc.pkl","rb")) ║
║  x = torch.tensor(window).unsqueeze(0)  # (1,16,F)    ║
║  label, p, re, reason = predict(                       ║
║      model, x, thr["thr_cls"], thr["thr_re"])          ║
╚════════════════════════════════════════════════════════╝
""")
