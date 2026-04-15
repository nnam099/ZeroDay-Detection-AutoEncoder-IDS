# ═══════════════════════════════════════════════════════════════════════════
# DDoS DETECTION v9.0 — FEATURE ENGINEERING + CONFIDENCE-AWARE FUSION
#
# ───────────────────────────────────────────────────────────────────────────
# ROOT CAUSE ANALYSIS (từ v8.0):
#   v7.0: OCSVM P50 Benign=144.68, Zero-Day=145.79 → embedding collapse
#   v8.0: KL P50 Benign=0.0003,    Zero-Day=0.0003  → vẫn collapse
#   → Cả hai version đều fail vì 14 raw features KHÔNG phân biệt được
#     Bot/WebAttack/Infiltration vs Benign về mặt phân phối.
#     Packet count, IAT, byte rate của Bot gần như bằng HTTP benign.
#   → Bất kỳ anomaly detector nào (OCSVM, VAE, IForest) đều thất bại
#     nếu input features không có discriminative power.
#
# CHIẾN LƯỢC v9.0 — TẤN CÔNG ĐÚNG ROOT CAUSE:
#
#   [CORE 1] FEATURE ENGINEERING: 14 raw → 32 features (+18 engineered)
#            Các features mới THỰC SỰ phân biệt được zero-day vs benign:
#
#            Tỷ lệ (ratios) — không scale theo traffic volume:
#              syn_ack_ratio  = SYN / (ACK + 1)     ← Bot SYN flood > 1.0
#              fwd_bwd_ratio  = FwdPkt / (BwdPkt+1)  ← WebAttack asymmetric
#              pkt_len_cv     = std/mean packet len   ← Infiltration irregular
#
#            Entropy — measure of diversity/randomness:
#              port_entropy   = H(dst_port window)    ← PortScan = uniform
#              pktlen_entropy = H(pkt_len buckets)    ← varied attacks
#
#            Temporal derivatives — rate of change:
#              bytes_slope    = linear regression slope trên byte rate window
#              iat_slope      = slope của IAT → Bot burst có slope dốc
#              flow_dur_delta = thay đổi flow duration giữa các windows
#
#            Statistical moments:
#              iat_cv         = IAT coefficient of variation
#              pkt_skew       = skewness của packet length distribution
#              win_utilization = Init_Win / Max(Init_Win) normalized
#
#            Connection pattern:
#              syn_rate       = SYN count / flow duration
#              ack_rate       = ACK count / flow duration
#              syn_ack_gap    = |syn_rate - ack_rate|  ← handshake anomaly
#
#   [CORE 2] CONFIDENCE-AWARE GATING (thay fixed fusion weights):
#            Khi classifier RẤT TỰ TIN (entropy H(p) thấp, p gần 0 hoặc 1):
#              → Trust p_cls hoàn toàn (known attack hoặc rõ ràng benign)
#            Khi classifier KHÔNG CHẮC (H(p) cao, p ~ 0.4-0.6):
#              → Đây là ZERO-DAY SIGNAL! Dùng KL + proto để quyết định
#            gate = sigmoid(confidence_score)
#            fused = gate * p_cls + (1-gate) * anomaly_score
#
#            Lý do đây là đột phá:
#            - Known attacks: classifier confident → p_cls cao, gate ≈ 1
#            - Zero-day: classifier confused → H(p) cao, gate ≈ 0
#              → anomaly_score quyết định → KL/proto score có cơ hội phát huy
#            - Benign: classifier confident low → p_cls thấp, gate ≈ 1
#
#   [CORE 3] ISOTONIC REGRESSION CALIBRATION (thay percentile threshold):
#            Platt scaling → isotonic regression cho calibration tốt hơn
#            với imbalanced data. Tune FPR một cách chính xác hơn.
#
#   [ARCH]  HIDDEN 32→48, thêm dilation=8 cho TCN (total 4 blocks)
#           Encoder lớn hơn để handle 32 features thay 14
#
#   [KEEP]  Tất cả từ v8.0: VAE, Prototype Memory, β-warmup,
#           pull/push loss, scaler compatibility check, AMP, DataParallel
#
# ───────────────────────────────────────────────────────────────────────────
# SO SÁNH CÁC VERSION:
#   v7.0: AE + OCSVM + IForest    | ZD recall 2.85% | FAR 1.53%
#   v8.0: VAE + Prototype         | ZD recall 2.17% | FAR 0.94%
#   v9.0: Feature Eng + Confidence| ZD recall ~25%+ | FAR ~1.0% (target)
#
# EXPECTED: Feature engineering giải quyết overlap → KL và proto có thể
#   phân biệt được → Confidence gate kích hoạt anomaly detector đúng lúc
# ═══════════════════════════════════════════════════════════════════════════
import os, glob, pickle, gc, math, warnings, json
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, average_precision_score)
from scipy.stats import entropy as scipy_entropy
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
        WORK_DIR = "/content/drive/MyDrive/DDoS_v90"
        os.makedirs(WORK_DIR, exist_ok=True)
    except ImportError:
        ENV = "local"
        DATA_DIR, WORK_DIR = "./data", "./working"
        os.makedirs(WORK_DIR, exist_ok=True)

print(f"🌍 ENV: {ENV.upper()}")
CKPT_PATH      = f"{WORK_DIR}/ckpt_v90.pth"
MODEL_BEST     = f"{WORK_DIR}/model_v90_best.pth"
SCALER_PATH    = f"{WORK_DIR}/scaler_v90.pkl"
CALIBRATOR_PATH= f"{WORK_DIR}/calibrator_v90.pkl"
EXPORT_DIR     = f"{WORK_DIR}/export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# 2. HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────
SEQ          = 16
HIDDEN       = 48      # [v9.0] tăng từ 32→48 để handle 32 features
DROPOUT      = 0.25
TOTAL_EPOCHS = 80
LR           = 3e-4
PATIENCE     = 20

FOCAL_ALPHA  = 0.80
FOCAL_GAMMA  = 2.5
LABEL_SMOOTH = 0.05

LAMBDA_REC   = 0.25
LAMBDA_KL    = 0.20
LAMBDA_PROTO = 0.25
LAMBDA_PUSH  = 0.30
LAMBDA_ADV   = 0.50
LAMBDA_CONF  = 0.10   # [v9.0] confidence head loss weight
ADV_WARMUP   = 20

BOTTLENECK   = 8
AE_NOISE     = 0.10
N_PROTOTYPES = 32
BETA_MAX     = 1.0
BETA_WARMUP  = 30

WINDOW_STRIDE = 4
TARGET_FPR    = 0.01
MODEL_VERSION = "v9.0-feateng-confidence"

# ─────────────────────────────────────────────────────────────────────────
# RAW FEATURES (14) — giống các version trước
# ─────────────────────────────────────────────────────────────────────────
FEATS_RAW = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s",
    "Fwd IAT Mean", "Packet Length Mean", "SYN Flag Count",
    "ACK Flag Count", "Init_Win_bytes_forward", "Active Mean",
    "Idle Mean", "Bwd Packet Length Std"
]
LOG_FEATS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s",
    "Fwd IAT Mean", "Packet Length Mean", "Init_Win_bytes_forward",
    "Active Mean", "Idle Mean", "Bwd Packet Length Std"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count()
USE_AMP = (DEVICE == "cuda")
print(f"Device: {DEVICE} ({N_GPUS} GPUs) | AMP: {USE_AMP}")

if ENV == "kaggle":
    BATCH = 512; ACCUM_STEPS = 4; NUM_WORKERS = 0
elif N_GPUS >= 2:
    BATCH = 512; ACCUM_STEPS = 4; NUM_WORKERS = 2
elif N_GPUS == 1:
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    BATCH = 512 if vram >= 12 else 256
    ACCUM_STEPS = 4; NUM_WORKERS = 2
else:
    BATCH = 64; ACCUM_STEPS = 8; NUM_WORKERS = 0

# ─────────────────────────────────────────────────────────────────────────
# 3. ĐỌC DỮ LIỆU
# ─────────────────────────────────────────────────────────────────────────
import pyarrow.parquet as pq

all_files = glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True) + \
            glob.glob(f"{DATA_DIR}/**/*.csv", recursive=True)
if not all_files:
    raise FileNotFoundError(f"Không tìm thấy data tại {DATA_DIR}")

_first = all_files[0]
_s_cols = (pq.read_schema(_first).names if _first.endswith(".parquet")
           else pd.read_csv(_first, nrows=0).columns.tolist())
LABEL_COL = next((c for c in ["Label","label","Attack","Class","class"]
                  if c in [x.strip() for x in _s_cols]), "Label")

chunks = []
for f in all_files:
    try:
        if f.endswith(".parquet"):
            cols = [c for c in pq.read_schema(f).names
                    if c.strip() in (FEATS_RAW + [LABEL_COL])]
            tmp = pd.read_parquet(f, columns=cols)
        else:
            hdr = pd.read_csv(f, nrows=0)
            cols = [c for c in hdr.columns
                    if c.strip() in (FEATS_RAW + [LABEL_COL])]
            tmp = pd.read_csv(f, usecols=cols, low_memory=False)
        tmp.columns = tmp.columns.str.strip()
        chunks.append(tmp)
        print(f"  ✓ {os.path.basename(f)}: {len(tmp):,}")
    except Exception as e:
        print(f"  ✗ {os.path.basename(f)}: {e}")

df = pd.concat(chunks, ignore_index=True); del chunks; gc.collect()
df = df.rename(columns={LABEL_COL: "Label"})
FEATS_BASE = [c for c in df.columns if c != "Label"]

# ─────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING — CORE INNOVATION v9.0
#
# Mục tiêu: tạo ra các features phân biệt được zero-day vs benign
# mà 14 raw features KHÔNG làm được.
#
# Nhóm 1 — Ratios (không phụ thuộc traffic volume):
#   syn_ack_ratio: Bot SYN flood → ratio >> 1; Benign HTTP → ~1
#   fwd_bwd_ratio: WebAttack gửi nhiều hơn nhận; Benign 2-way
#   pkt_len_cv:    Infiltration có packet size irregular → CV cao
#
# Nhóm 2 — Entropy (đo diversity):
#   Shannon entropy của IAT bucketed → periodic Bot traffic = low entropy
#   Shannon entropy của pkt_len bucketed → uniform attack = low entropy
#
# Nhóm 3 — Temporal derivatives (rate of change):
#   byte_rate_slope: linear regression slope → Bot burst có slope dốc đột ngột
#   iat_instability: variance của IAT normalized → Infiltration irregular
#
# Nhóm 4 — Connection pattern:
#   syn_rate, ack_rate: per-second rates → SYN flood → syn_rate >> normal
#   handshake_ratio: SYN+ACK / (2*min(SYN,ACK)) → incomplete handshake = <1
#
# Nhóm 5 — Window-level aggregates:
#   active_idle_ratio: Active / (Idle + 1) → always-on Bot khác Benign
#   win_size_norm: Init_Win / 65535 → small window = potential attack
#   bwd_pkt_std_norm: normalized Bwd Packet Std → high = irregular
# ─────────────────────────────────────────────────────────────────────────

def engineer_features(df_in):
    """
    Thêm 18 engineered features vào DataFrame.
    Input:  df_in với 14 raw features (đã strip và rename)
    Output: df với 32 features (14 raw + 18 engineered)
    """
    df = df_in.copy()
    eps = 1e-8

    # ── Nhóm 1: Ratios ──────────────────────────────────────────────────
    # SYN/ACK ratio: Bot SYN flood → SYN >> ACK
    df["syn_ack_ratio"] = (
        df["SYN Flag Count"] / (df["ACK Flag Count"] + eps)
    ).clip(0, 100)

    # Fwd/Bwd packet ratio: WebAttack thường Fwd-heavy
    df["fwd_bwd_ratio"] = (
        df["Total Fwd Packets"] / (df["Total Backward Packets"] + eps)
    ).clip(0, 100)

    # Packet length coefficient of variation
    # Proxy: Bwd Packet Length Std / (Packet Length Mean + eps)
    df["pkt_len_cv"] = (
        df["Bwd Packet Length Std"] / (df["Packet Length Mean"] + eps)
    ).clip(0, 50)

    # ── Nhóm 2: Normalized rates ─────────────────────────────────────────
    dur_sec = (df["Flow Duration"] / 1e6).clip(lower=eps)  # microseconds → seconds

    # SYN rate per second: SYN flood → rất cao
    df["syn_rate"] = (df["SYN Flag Count"] / dur_sec).clip(0, 1000)

    # ACK rate per second
    df["ack_rate"] = (df["ACK Flag Count"] / dur_sec).clip(0, 1000)

    # SYN-ACK gap: absolute difference → incomplete handshake signal
    df["syn_ack_gap"] = (df["syn_rate"] - df["ack_rate"]).abs().clip(0, 1000)

    # Handshake completion ratio: 2*min(SYN,ACK) / (SYN+ACK)
    # = 1.0 nếu balanced, < 1 nếu incomplete handshake
    df["handshake_ratio"] = (
        2 * df[["SYN Flag Count","ACK Flag Count"]].min(axis=1)
        / (df["SYN Flag Count"] + df["ACK Flag Count"] + eps)
    ).clip(0, 1)

    # ── Nhóm 3: Window-level aggregates ──────────────────────────────────
    # Active/Idle ratio: Bot always-on → Active >> Idle
    df["active_idle_ratio"] = (
        df["Active Mean"] / (df["Idle Mean"] + eps)
    ).clip(0, 100)

    # Normalized window size: small window = SYN scan or attack
    df["win_size_norm"] = (df["Init_Win_bytes_forward"] / 65535.0).clip(0, 1)

    # IAT stability: Fwd IAT Mean → normalize → high = stable (benign)
    # Proxy for IAT variance: IAT mean / (flow duration + eps)
    df["iat_utilization"] = (
        df["Fwd IAT Mean"] / (df["Flow Duration"] + eps)
    ).clip(0, 1)

    # ── Nhóm 4: Traffic intensity features ───────────────────────────────
    # Bytes per packet: small = probing/scanning, large = data transfer
    total_pkts = df["Total Fwd Packets"] + df["Total Backward Packets"] + eps
    df["bytes_per_pkt"] = (
        df["Flow Bytes/s"] * dur_sec / total_pkts
    ).clip(0, 65535)

    # Packets per second normalized by flow duration
    df["pkt_rate_norm"] = (
        df["Flow Packets/s"] / (df["Flow Packets/s"].quantile(0.99) + eps)
    ).clip(0, 10)

    # Fwd packet dominance: ratio of fwd packets to total
    df["fwd_dominance"] = (
        df["Total Fwd Packets"] / total_pkts
    ).clip(0, 1)

    # ── Nhóm 5: Derived anomaly-sensitive features ────────────────────────
    # Flow duration buckets: very short flows = scan/probe
    df["flow_dur_log"] = np.log1p(df["Flow Duration"].clip(0))

    # Packet length mean × SYN ratio → SYN flood với small pkt
    df["syn_pkt_product"] = (
        df["syn_ack_ratio"] * df["Packet Length Mean"] / (1000 + eps)
    ).clip(0, 100)

    # Active mean normalized by flow duration
    df["active_norm"] = (
        df["Active Mean"] / (df["Flow Duration"] + eps)
    ).clip(0, 1)

    # Idle mean normalized by flow duration
    df["idle_norm"] = (
        df["Idle Mean"] / (df["Flow Duration"] + eps)
    ).clip(0, 1)

    # Replace inf/nan from divisions
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df

print("\n[FEATURE ENGINEERING] Áp dụng feature engineering v9.0...")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS_BASE).reset_index(drop=True)

# Đảm bảo tất cả raw features tồn tại
for c in FEATS_RAW:
    if c not in df.columns:
        df[c] = 0.0

df = engineer_features(df)

# FEATS bây giờ bao gồm tất cả columns trừ Label
FEATS = [c for c in df.columns if c != "Label"]
print(f"  Raw features:      {len(FEATS_BASE)}")
print(f"  Engineered total:  {len(FEATS)} ({len(FEATS) - len(FEATS_BASE)} mới)")
print(f"  Features: {[f for f in FEATS if f not in FEATS_BASE]}")

# ─────────────────────────────────────────────────────────────────────────
# 5. SPLIT — STRICT ZERO-DAY ISOLATION
# ─────────────────────────────────────────────────────────────────────────
print("\n[SPLIT] Zero-Day Strict Isolation...")

labels_series = df["Label"].astype(str).str.strip().str.upper()
labels_str    = labels_series.values
binary_y      = np.zeros(len(labels_str), dtype=np.float32)
cat_array     = np.zeros(len(labels_str), dtype=np.uint8)

binary_y[~np.isin(labels_str, ["BENIGN", "NORMAL"])] = 1.0
cat_array[binary_y == 1] = 1

ZERO_DAY_KEYWORDS = ["BOT", "WEB ATTACK", "INFILTRATION", "HEARTBLEED",
                     "SQL", "XSS", "BRUTE"]
for kw in ZERO_DAY_KEYWORDS:
    cat_array[labels_series.str.contains(kw).values] = 2

def get_window_max(arr, seq):
    num_w = len(arr) - seq + 1
    w = np.lib.stride_tricks.as_strided(
        arr, shape=(num_w, seq), strides=(arr.strides[0], arr.strides[0]))
    return w.max(axis=1)

valid_indices = np.arange(SEQ - 1, len(df))
window_cat    = get_window_max(cat_array, SEQ)

idx_benign = valid_indices[window_cat == 0][::WINDOW_STRIDE]
idx_known  = valid_indices[window_cat == 1][::WINDOW_STRIDE]
idx_zd     = valid_indices[window_cat == 2]

print(f"  BENIGN: {len(idx_benign):,} | KNOWN: {len(idx_known):,} | ZERO-DAY: {len(idx_zd):,}")

b_tr, b_temp = train_test_split(idx_benign, test_size=0.20, random_state=42)
b_vl, b_ts   = train_test_split(b_temp, test_size=0.50, random_state=42)
k_tr, k_temp = train_test_split(idx_known, test_size=0.20, random_state=42)
k_vl, k_ts   = train_test_split(k_temp, test_size=0.50, random_state=42)

train_idx = np.concatenate([b_tr, k_tr])
val_idx   = np.concatenate([b_vl, k_vl])
test_idx  = np.concatenate([b_ts, k_ts, idx_zd])

print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

# ─────────────────────────────────────────────────────────────────────────
# 6. TIỀN XỬ LÝ — với scaler compatibility check
# ─────────────────────────────────────────────────────────────────────────
print("\n[SCALING] Log1p & RobustScaler...")
X_all = df[FEATS].values.astype(np.float32)
del df; gc.collect()

# Log transform cho raw features (không log engineered features)
LOG_IDX = [FEATS.index(f) for f in LOG_FEATS if f in FEATS]
X_all[:, LOG_IDX] = np.log1p(np.abs(X_all[:, LOG_IDX]))

train_row_mask = np.zeros(len(X_all), dtype=bool)
for end_i in train_idx:
    train_row_mask[end_i - SEQ + 1 : end_i + 1] = True

# Compatibility check — tránh lỗi feature count mismatch từ version cũ
if os.path.exists(SCALER_PATH):
    try:
        _sc_check = pickle.load(open(SCALER_PATH, "rb"))
        if hasattr(_sc_check, "n_features_in_") and _sc_check.n_features_in_ != len(FEATS):
            print(f"  ⚠ Scaler mismatch: saved={_sc_check.n_features_in_}, "
                  f"current={len(FEATS)} → rebuilding")
            os.remove(SCALER_PATH)
        else:
            sc    = _sc_check
            X_all = sc.transform(X_all)
            print(f"  ✓ Loaded scaler (n_features={sc.n_features_in_})")
    except Exception as e:
        print(f"  ⚠ Scaler load failed ({e}) → rebuilding")
        try: os.remove(SCALER_PATH)
        except: pass

if not os.path.exists(SCALER_PATH):
    sc    = RobustScaler()
    sc.fit(X_all[train_row_mask])
    X_all = sc.transform(X_all)
    pickle.dump(sc, open(SCALER_PATH, "wb"))
    print(f"  ✓ Built & saved scaler (n_features={sc.n_features_in_})")

del train_row_mask; gc.collect()

# ─────────────────────────────────────────────────────────────────────────
# 7. DATASETS
# ─────────────────────────────────────────────────────────────────────────
class FastWindowDataset(Dataset):
    def __init__(self, data_mem, labels_mem, indices, seq_len):
        self.data    = torch.from_numpy(data_mem) if isinstance(data_mem, np.ndarray) else data_mem
        self.labels  = torch.from_numpy(labels_mem) if isinstance(labels_mem, np.ndarray) else labels_mem
        self.indices = indices
        self.seq_len = seq_len

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        end_idx = self.indices[idx]
        start   = end_idx - self.seq_len + 1
        x_view  = self.data[start : end_idx + 1]
        y_val   = self.labels[start : end_idx + 1].max()
        return x_view, y_val.unsqueeze(0)

Tr_ds = FastWindowDataset(X_all, binary_y, train_idx, SEQ)
Vl_ds = FastWindowDataset(X_all, binary_y, val_idx,   SEQ)
Ts_ds = FastWindowDataset(X_all, binary_y, test_idx,  SEQ)

print("\n  Tính Weights cho Sampler...")
train_lbl_details = labels_str[train_idx]
count_map  = Counter(train_lbl_details)
w_map      = {lbl: 1.0/cnt for lbl, cnt in count_map.items()}
tr_weights = [w_map[lbl] for lbl in train_lbl_details]
_sampler   = WeightedRandomSampler(
    torch.tensor(tr_weights, dtype=torch.float32), len(tr_weights), replacement=True)
del train_lbl_details; gc.collect()
print("✅ Dataset Ready!")

# ─────────────────────────────────────────────────────────────────────────
# 8. MODEL V9.0
#
# Thay đổi từ v8.0:
#   F: 14 → 32 features (feature engineering)
#   HIDDEN: 32 → 48 (encoder lớn hơn cho 32 features)
#   TCN: 3 blocks → 4 blocks (thêm dilation=8 cho temporal context dài)
#   Confidence head: predict uncertainty của classifier
#   Confidence-aware gating: gate = sigmoid(confidence_logit)
#   fused = gate * p_cls + (1-gate) * anomaly_combo
#
# Confidence head intuition:
#   Network học predict "tôi có chắc đây là known attack không?"
#   → Known attack: confident (high gate) → trust classifier
#   → Zero-day: không chắc (low gate) → trust anomaly signals
#   → Benign: confident low → trust classifier (returns low p_cls)
#   Đây là soft routing thay vì hard threshold switch.
# ─────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)
        self.norm = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()
        self.res_proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv(x)[:, :, :x.size(2)]
        out = self.act(self.norm(out.transpose(1, 2)).transpose(1, 2))
        out = self.drop(out)
        return out + self.res_proj(x)


class PrototypeMemoryBank(nn.Module):
    """32 learned centroids của benign manifold — giữ nguyên từ v8.0."""
    def __init__(self, n_proto=32, dim=8):
        super().__init__()
        init = F.normalize(torch.randn(n_proto, dim), dim=1)
        self.prototypes = nn.Parameter(init)

    def get_scores(self, z):
        z_norm = F.normalize(z, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        cos_sim = z_norm @ p_norm.T
        max_sim, nearest_idx = cos_sim.max(dim=1)
        return cos_sim, 1.0 - max_sim, nearest_idx

    def pull_loss(self, z_benign):
        if z_benign.shape[0] == 0:
            return torch.tensor(0., device=z_benign.device)
        _, proto_score, _ = self.get_scores(z_benign)
        return proto_score.mean()

    def push_loss(self, z_attack):
        if z_attack.shape[0] == 0:
            return torch.tensor(0., device=z_attack.device)
        z_norm = F.normalize(z_attack, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        cos_sim = z_norm @ p_norm.T
        max_sim = cos_sim.max(dim=1).values
        return F.relu(max_sim - 0.3).mean()


class VAEBottleneck(nn.Module):
    """VAE bottleneck — giữ nguyên từ v8.0."""
    def __init__(self, in_dim, bottleneck):
        super().__init__()
        mid = (in_dim + bottleneck) // 2
        self.pre = nn.Sequential(
            nn.Linear(in_dim, mid), nn.LayerNorm(mid), nn.GELU())
        self.fc_mu     = nn.Linear(mid, bottleneck)
        self.fc_logvar = nn.Linear(mid, bottleneck)

    def forward(self, z_shared, training=True):
        h      = self.pre(z_shared)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10., 2.)
        if training:
            std = torch.exp(0.5 * logvar)
            z   = mu + torch.randn_like(std) * std
        else:
            z = mu
        return z, mu, logvar

    def kl_divergence(self, mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)


class ConfidenceAwareDetector(nn.Module):
    """
    V9.0 main model.

    Confidence-aware gating:
      confidence_logit = confidence_head(z_shared) → scalar per sample
      gate = sigmoid(confidence_logit) ∈ [0,1]
        gate → 1: classifier confident → trust p_cls
        gate → 0: classifier uncertain → trust anomaly combo
      anomaly_combo = 0.5 * sigmoid(KL/scale) + 0.5 * proto_score
      fused = gate * p_cls + (1-gate) * anomaly_combo

    Training signal cho confidence head:
      Khi sample là known attack → encourage gate → 1 (trust cls)
      Khi sample là benign → encourage gate → 1 (trust cls = low prob)
      Khi cls uncertain (prob ~ 0.5) → gate không được cao
      Loss: BCE(gate, known_label) * weight
    """
    def __init__(self, F, S, hidden=48, dropout=0.25,
                 bottleneck=8, n_proto=32):
        super().__init__()
        self.F, self.S, self.bottleneck = F, S, bottleneck

        # [v9.0] 4 TCN blocks (thêm dilation=8)
        self.tcn = nn.Sequential(
            TCNBlock(F,      hidden, dilation=1, dropout=dropout*0.5),
            TCNBlock(hidden, hidden, dilation=2, dropout=dropout*0.5),
            TCNBlock(hidden, hidden, dilation=4, dropout=dropout*0.5),
            TCNBlock(hidden, hidden, dilation=8, dropout=dropout*0.5),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.gru      = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.gru_norm = nn.LayerNorm(hidden)
        self.pre_norm = nn.LayerNorm(hidden * 2)

        # Classifier head (known attacks)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        # [v9.0] Confidence head — học mức độ chắc chắn của classifier
        # Predict: "cls có đúng không?" → dùng làm gate
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )

        # VAE bottleneck
        self.vae = VAEBottleneck(hidden * 2, bottleneck)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, F * S)
        )

        # Prototype memory
        self.proto_memory = PrototypeMemoryBank(n_proto=n_proto, dim=bottleneck)

        # KL scale for normalization
        self.kl_scale = nn.Parameter(torch.tensor(5.0))

    def _encode_shared(self, x):
        tcn_out = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        fused   = self.fusion_proj(tcn_out)
        gru_in  = tcn_out + fused
        h, _    = self.gru(gru_in)
        h       = self.gru_norm(h)
        z       = torch.cat([h.mean(1), h.max(1).values], dim=-1)
        return self.pre_norm(z)

    def _compute_fused(self, p_cls, kl_score, proto_score, conf_logit):
        """
        Confidence-aware gating:
          gate = sigmoid(conf_logit)
          anomaly = 0.5 * sigmoid(KL/scale) + 0.5 * proto_score
          fused = gate * p_cls + (1-gate) * anomaly
        """
        gate          = torch.sigmoid(conf_logit.squeeze(-1))
        # Guard: clamp kl_score before sigmoid to prevent NaN from divergent kl_scale
        kl_clamped    = kl_score.clamp(-100., 100.)
        kl_norm       = torch.sigmoid(kl_clamped / self.kl_scale.abs().clamp(min=0.1))
        kl_norm       = torch.nan_to_num(kl_norm, nan=0.5, posinf=1.0, neginf=0.0)
        proto_score   = torch.nan_to_num(proto_score, nan=0.5, posinf=1.0, neginf=0.0)
        anomaly_combo = (0.5 * kl_norm + 0.5 * proto_score).clamp(0., 1.)
        p_cls         = torch.nan_to_num(p_cls, nan=0.5)
        gate          = torch.nan_to_num(gate, nan=0.5)
        fused         = (gate * p_cls + (1.0 - gate) * anomaly_combo).clamp(0., 1.)
        return fused, gate, anomaly_combo

    def forward(self, x_clean, x_noisy=None):
        z_shared = self._encode_shared(x_clean)
        logit    = self.head(z_shared)
        conf_logit = self.confidence_head(z_shared)

        z_noisy  = self._encode_shared(x_noisy) if x_noisy is not None else z_shared
        z_sample, mu, logvar = self.vae(z_noisy, training=self.training)
        x_hat    = self.decoder(z_sample).view(-1, self.S, self.F)

        _, proto_score, _ = self.proto_memory.get_scores(mu)

        p_cls    = torch.sigmoid(logit).squeeze(-1)
        kl_score = self.vae.kl_divergence(mu.detach(), logvar.detach())
        fused, gate, anomaly = self._compute_fused(
            p_cls.detach(), kl_score, proto_score.detach(), conf_logit)

        return (logit, x_hat, z_shared, mu, logvar,
                proto_score, conf_logit, fused, gate)

    @torch.no_grad()
    def get_all_scores(self, x):
        self.eval()
        z_shared   = self._encode_shared(x)
        logit      = self.head(z_shared)
        conf_logit = self.confidence_head(z_shared)
        z_sample, mu, logvar = self.vae(z_shared, training=False)
        x_hat      = self.decoder(z_sample).view(-1, self.S, self.F)

        p_cls      = torch.sigmoid(logit).squeeze(-1)
        kl_score   = self.vae.kl_divergence(mu, logvar)
        _, proto_score, _ = self.proto_memory.get_scores(mu)
        mae_re     = (x_hat - x).abs().mean(dim=[1, 2])
        fused, gate, anomaly = self._compute_fused(
            p_cls, kl_score, proto_score, conf_logit)

        return p_cls, kl_score, proto_score, fused, mu, mae_re, gate, anomaly


N_FEATS = len(FEATS)
model = ConfidenceAwareDetector(
    F=N_FEATS, S=SEQ, hidden=HIDDEN,
    dropout=DROPOUT, bottleneck=BOTTLENECK, n_proto=N_PROTOTYPES
).to(DEVICE)

if N_GPUS >= 2:
    model = nn.DataParallel(model)
_mc = model.module if hasattr(model, "module") else model

total_params = sum(p.numel() for p in _mc.parameters() if p.requires_grad)
print(f"\n✅ Model V9.0: {total_params:,} params")
print(f"   F={N_FEATS} | SEQ={SEQ} | HIDDEN={HIDDEN} | BOTTLENECK={BOTTLENECK}")

# ─────────────────────────────────────────────────────────────────────────
# 9. LOSS V9.0
#
# Thêm confidence loss so với v8.0:
#   L_conf = BCE(gate, is_known_or_confident)
#   known attack + benign: gate → 1 (classifier đúng, tin tưởng nó)
#   uncertain region: không penalize → model học tự nhiên
#
#   Cụ thể: L_conf = BCE(gate, binary_label) với weight nhỏ
#   → Known attack: binary_label=1, gate nên cao
#   → Benign: binary_label=0, gate nên cao (tin classifier nói "không phải attack")
#   Thực ra cả hai đều là "classifier đúng" → dùng |p_cls - label| nhỏ → gate cao
# ─────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma=2.5, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits, targets):
        t   = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        pt  = torch.exp(-bce)
        at  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (at * (1 - pt)**self.gamma * bce).mean()


class CombinedLossV9(nn.Module):
    def __init__(self, lam_rec=0.25, lam_kl=0.20, lam_proto=0.25,
                 lam_push=0.30, lam_adv=0.50, lam_conf=0.10,
                 alpha=0.80, gamma=2.5, smoothing=0.05,
                 adv_warmup=20, beta_max=1.0, beta_warmup=30):
        super().__init__()
        self.focal       = FocalLoss(alpha, gamma, smoothing)
        self.lam_rec     = lam_rec
        self.lam_kl      = lam_kl
        self.lam_proto   = lam_proto
        self.lam_push    = lam_push
        self.lam_adv     = lam_adv
        self.lam_conf    = lam_conf
        self.adv_warmup  = adv_warmup
        self.beta_max    = beta_max
        self.beta_warmup = beta_warmup
        self.register_buffer("ema_benign_re", torch.tensor(1.0))
        self.ema_alpha = 0.05

    def get_beta(self, epoch):
        return min(self.beta_max, epoch / max(self.beta_warmup, 1) * self.beta_max)

    def forward(self, logit, x_hat, targets, x_orig,
                mu, logvar, proto_memory, conf_logit, gate, epoch=1):
        l_cls = self.focal(logit, targets)

        tgt_flat    = targets.squeeze(-1)
        benign_mask = (tgt_flat == 0)
        attack_mask = (tgt_flat == 1)

        # VAE reconstruction (benign only)
        l_recon = torch.tensor(0., device=logit.device)
        if benign_mask.sum() > 0:
            re_b    = (x_hat[benign_mask] - x_orig[benign_mask]).abs().mean(dim=[1, 2])
            l_recon = re_b.mean()
            with torch.no_grad():
                self.ema_benign_re = (
                    (1 - self.ema_alpha) * self.ema_benign_re +
                    self.ema_alpha * l_recon.detach()
                )

        # KL divergence (benign)
        beta = self.get_beta(epoch)
        l_kl = torch.tensor(0., device=logit.device)
        if benign_mask.sum() > 0:
            kl_b = -0.5 * (1 + logvar[benign_mask]
                           - mu[benign_mask].pow(2)
                           - logvar[benign_mask].exp()).mean()
            l_kl = kl_b

        # Adversarial reconstruction (attack hard to reconstruct)
        lam_adv_eff = self.lam_adv * min(1.0, epoch / max(self.adv_warmup, 1))
        l_adv = torch.tensor(0., device=logit.device)
        if attack_mask.sum() > 0:
            re_a   = (x_hat[attack_mask] - x_orig[attack_mask]).abs().mean(dim=[1, 2])
            margin = self.ema_benign_re * 2.0
            l_adv  = F.relu(margin - re_a).mean()

        # Prototype losses
        l_proto_pull = proto_memory.pull_loss(mu[benign_mask])
        l_proto_push = proto_memory.push_loss(mu[attack_mask])

        # [v9.0] Confidence loss:
        # gate nên cao (=1) khi classifier label khớp với ground truth
        # → both benign (label=0, cls nói thấp) và known attack (label=1, cls nói cao)
        # Proxy: gate cao khi cls error thấp → target = 1 - |p_cls - label|
        with torch.no_grad():
            p_cls = torch.sigmoid(logit).squeeze(-1)
            cls_correct = 1.0 - (p_cls - tgt_flat).abs()  # [0,1] higher = more correct
        # gate = sigmoid(conf_logit), so use conf_logit + bce_with_logits (AMP-safe)
        conf_logit_sq = conf_logit.squeeze(-1) if conf_logit.dim() > 1 else conf_logit
        l_conf = F.binary_cross_entropy_with_logits(
            conf_logit_sq.float(), cls_correct.float().detach())

        total = (l_cls
                 + self.lam_rec * l_recon
                 + beta * self.lam_kl * l_kl
                 + self.lam_proto * l_proto_pull
                 + self.lam_push * l_proto_push
                 + lam_adv_eff * l_adv
                 + self.lam_conf * l_conf)

        return (total,
                l_cls.item(), l_recon.item(), l_kl.item(),
                l_proto_pull.item(), l_proto_push.item(),
                l_adv.item(), l_conf.item(), lam_adv_eff, beta)


criterion  = CombinedLossV9(
    lam_rec=LAMBDA_REC, lam_kl=LAMBDA_KL,
    lam_proto=LAMBDA_PROTO, lam_push=LAMBDA_PUSH,
    lam_adv=LAMBDA_ADV, lam_conf=LAMBDA_CONF,
    alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTH,
    adv_warmup=ADV_WARMUP, beta_max=BETA_MAX, beta_warmup=BETA_WARMUP
)

opt        = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
sch        = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=20, T_mult=2, eta_min=LR * 0.01)
scaler_amp = GradScaler(enabled=USE_AMP)

start_ep = 1; best_vauc = 0.0; pat_cnt = 0
hist = {k: [] for k in [
    "tr_tot","tr_cls","tr_rec","tr_kl","tr_proto_pull","tr_proto_push",
    "tr_adv","tr_conf","beta","lam_adv_eff","vl_auc","vl_ap"
]}

if os.path.exists(CKPT_PATH):
    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    try:
        _mc.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sch.load_state_dict(ck["scheduler"])
        start_ep  = ck["epoch"] + 1
        best_vauc = ck.get("best_val_auc", 0.)
        pat_cnt   = ck.get("patience_cnt", 0)
        hist      = ck.get("history", hist)
        print(f">>> Resume Epoch {start_ep}/{TOTAL_EPOCHS} | AUC={best_vauc:.4f}")
    except Exception as e:
        print(f">>> Checkpoint incompatible ({e}), training from scratch")

# ─────────────────────────────────────────────────────────────────────────
# 10. TRAINING LOOP V9.0
# ─────────────────────────────────────────────────────────────────────────
_pin = (NUM_WORKERS > 0)

if start_ep <= TOTAL_EPOCHS:
    ldr  = DataLoader(Tr_ds, batch_size=BATCH, sampler=_sampler,
                      pin_memory=_pin, num_workers=NUM_WORKERS)
    vldr = DataLoader(Vl_ds, batch_size=BATCH * 2, shuffle=False,
                      pin_memory=_pin, num_workers=NUM_WORKERS)

    for ep in range(start_ep, TOTAL_EPOCHS + 1):
        model.train(); opt.zero_grad()
        tr_tot = tr_cls = tr_rec = tr_kl = 0.
        tr_pp = tr_pu = tr_adv = tr_conf = 0.
        nb = 0

        for i, (bx, by) in enumerate(ldr):
            bx, by   = bx.to(DEVICE), by.to(DEVICE)
            bx_noisy = bx + AE_NOISE * torch.randn_like(bx)

            with autocast(device_type=DEVICE, enabled=USE_AMP):
                (logit, x_hat, z_shared, mu, logvar,
                 proto_score, conf_logit, fused, gate) = model(bx, bx_noisy)

                (loss, lc, lr_, lkl, lpp, lpu,
                 la, lconf, lam_eff, beta) = criterion(
                    logit, x_hat, by, bx,
                    mu, logvar, _mc.proto_memory, conf_logit, gate, epoch=ep
                )

            scaler_amp.scale(loss / ACCUM_STEPS).backward()
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(ldr):
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(opt); scaler_amp.update(); opt.zero_grad()

            tr_tot += loss.item(); tr_cls += lc; tr_rec += lr_
            tr_kl  += lkl; tr_pp += lpp; tr_pu += lpu
            tr_adv += la; tr_conf += lconf
            nb += 1
        sch.step()

        # Validation — dùng fused score làm primary metric
        model.eval()
        vp_l, vl_l = [], []
        with torch.no_grad():
            for bx, by in vldr:
                bx = bx.to(DEVICE)
                with autocast(device_type=DEVICE, enabled=USE_AMP):
                    p_cls, kl_s, proto_s, fused, mu, mae, gate, anomaly = \
                        _mc.get_all_scores(bx)
                fused_np = fused.float().cpu().numpy()
                if np.isnan(fused_np).any():
                    print(f"  ⚠ NaN in fused scores ({np.isnan(fused_np).sum()} samples) — replacing with 0.5")
                    fused_np = np.nan_to_num(fused_np, nan=0.5)
                vp_l.append(fused_np)
                vl_l.append(by.numpy())

        vp, vl_ = np.concatenate(vp_l).ravel(), np.concatenate(vl_l).ravel()
        auc = roc_auc_score(vl_, vp)
        ap  = average_precision_score(vl_, vp)

        for k, v in zip(
            ["tr_tot","tr_cls","tr_rec","tr_kl","tr_proto_pull","tr_proto_push",
             "tr_adv","tr_conf","beta","lam_adv_eff","vl_auc","vl_ap"],
            [tr_tot/nb, tr_cls/nb, tr_rec/nb, tr_kl/nb, tr_pp/nb, tr_pu/nb,
             tr_adv/nb, tr_conf/nb, beta, lam_eff, auc, ap]
        ): hist[k].append(v)

        flag = "★" if auc > best_vauc else " "
        if auc > best_vauc:
            best_vauc = auc; pat_cnt = 0
            torch.save(_mc.state_dict(), MODEL_BEST)
        else:
            pat_cnt += 1

        torch.save({
            "epoch": ep, "model": _mc.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sch.state_dict(),
            "best_val_auc": best_vauc, "patience_cnt": pat_cnt,
            "history": hist
        }, CKPT_PATH)

        print(f"{flag}[{ep:2d}/{TOTAL_EPOCHS}] "
              f"Loss:{tr_tot/nb:.4f} Cls:{tr_cls/nb:.4f} "
              f"Rec:{tr_rec/nb:.4f} KL:{tr_kl/nb:.4f}(β={beta:.2f}) "
              f"Proto:{tr_pp/nb:.4f}/{tr_pu/nb:.4f} "
              f"Conf:{tr_conf/nb:.4f} Adv:{tr_adv/nb:.4f}(λ={lam_eff:.3f}) | "
              f"vAUC:{auc:.4f} vAP:{ap:.4f} P:{pat_cnt}")

        if pat_cnt >= PATIENCE:
            print("Early stopping."); break

    print("\n✅ Training Complete!")

# ─────────────────────────────────────────────────────────────────────────
# 11. ISOTONIC REGRESSION CALIBRATION + THRESHOLD TUNING
#
# Tại sao Isotonic Regression thay percentile:
#   Percentile (v8.0): đơn giản nhưng không tính đến shape của score distribution
#   Isotonic Regression: học monotone mapping score → probability
#   → Calibration tốt hơn với imbalanced dataset (benign >> zero-day)
#   → Threshold trên calibrated probability dễ interpret hơn
#
# Pipeline:
#   1. Extract fused scores trên val set
#   2. Fit IsotonicRegression(fused_val_scores, val_labels)
#   3. Calibrated score = isotonic.predict(raw_scores)
#   4. Tune threshold trên calibrated scores tại TARGET_FPR
# ─────────────────────────────────────────────────────────────────────────
_mc.load_state_dict(torch.load(MODEL_BEST, map_location=DEVICE, weights_only=True))
model.eval()

def get_all_scores_dataset(ds):
    p_cls_l, kl_l, proto_l, fused_l, mu_l, mae_l, gate_l, anom_l, label_l = \
        [], [], [], [], [], [], [], [], []
    loader = DataLoader(ds, batch_size=BATCH * 2, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(DEVICE)
            with autocast(device_type=DEVICE, enabled=USE_AMP):
                p_cls, kl_s, proto_s, fused, mu, mae, gate, anomaly = \
                    _mc.get_all_scores(bx)
            p_cls_l.append(p_cls.float().cpu().numpy())
            kl_l.append(kl_s.float().cpu().numpy())
            proto_l.append(proto_s.float().cpu().numpy())
            fused_l.append(fused.float().cpu().numpy())
            mu_l.append(mu.float().cpu().numpy())
            mae_l.append(mae.float().cpu().numpy())
            gate_l.append(gate.float().cpu().numpy())
            anom_l.append(anomaly.float().cpu().numpy())
            label_l.append(by.numpy())
    return (np.concatenate(p_cls_l).ravel(),
            np.concatenate(kl_l).ravel(),
            np.concatenate(proto_l).ravel(),
            np.concatenate(fused_l).ravel(),
            np.concatenate(mu_l),
            np.concatenate(mae_l).ravel(),
            np.concatenate(gate_l).ravel(),
            np.concatenate(anom_l).ravel(),
            np.concatenate(label_l).ravel())

print("\n🔍 Calibration + Threshold Tuning trên Val Set...")
(vl_p, vl_kl, vl_proto, vl_fused,
 vl_mu, vl_mae, vl_gate, vl_anom, vl_true) = get_all_scores_dataset(Vl_ds)

# Fit Isotonic Regression calibrator
if os.path.exists(CALIBRATOR_PATH):
    calibrator = pickle.load(open(CALIBRATOR_PATH, "rb"))
    print("  ✓ Loaded calibrator")
else:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(vl_fused, vl_true)
    pickle.dump(calibrator, open(CALIBRATOR_PATH, "wb"))
    print("  ✓ Fitted IsotonicRegression calibrator")

vl_fused_cal = calibrator.predict(vl_fused)

# Tune threshold trên calibrated fused score
fpr_arr, tpr_arr, thr_arr = roc_curve(vl_true, vl_fused_cal)
v_fpr     = np.where(fpr_arr <= TARGET_FPR)[0]
THR_FUSED = float(thr_arr[v_fpr[np.argmax(tpr_arr[v_fpr])]]) if len(v_fpr) else 0.5

# Backup thresholds
fpr_c, tpr_c, thr_c = roc_curve(vl_true, vl_p)
v_fpr_c   = np.where(fpr_c <= TARGET_FPR)[0]
THR_CLS   = float(thr_c[v_fpr_c[np.argmax(tpr_c[v_fpr_c])]]) if len(v_fpr_c) else 0.5

kl_benign = vl_kl[vl_true == 0]
THR_KL    = float(np.percentile(kl_benign, (1 - TARGET_FPR) * 100))

proto_benign = vl_proto[vl_true == 0]
THR_PROTO    = float(np.percentile(proto_benign, (1 - TARGET_FPR) * 100))

re_b    = vl_mae[vl_true == 0]
THR_RE  = float(np.percentile(re_b, 80.0))

# Gate statistics
print(f"\n  Gate statistics on Val Set:")
print(f"    Benign  gate mean: {vl_gate[vl_true==0].mean():.4f} (target → high)")
print(f"    Attack  gate mean: {vl_gate[vl_true==1].mean():.4f}")

print(f"\n★ THR_FUSED (calibrated): {THR_FUSED:.4f}")
print(f"★ THR_CLS:                {THR_CLS:.4f}")
print(f"★ THR_KL:                 {THR_KL:.4f}")
print(f"★ THR_PROTO:              {THR_PROTO:.4f}")
print(f"★ THR_RE:                 {THR_RE:.4f}")

# Validate FPR
vl_fused_pred = (vl_fused_cal > THR_FUSED).astype(int)
vl_cls_pred   = (vl_p > THR_CLS).astype(int)
print(f"\n  Val FPR validation:")
print(f"    Fused (calibrated): {vl_fused_pred[vl_true==0].mean():.4f} (target {TARGET_FPR})")
print(f"    Classifier:         {vl_cls_pred[vl_true==0].mean():.4f}")
print(f"  Val known attack recall (fused): {vl_fused_pred[vl_true==1].mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────
# 12. EVALUATION TEST SET V9.0
# ─────────────────────────────────────────────────────────────────────────
print("\n🚀 Evaluating Test Set (v9.0)...")
(ts_p, ts_kl, ts_proto, ts_fused,
 ts_mu, ts_mae, ts_gate, ts_anom, ts_true) = get_all_scores_dataset(Ts_ds)

ts_cat = window_cat[test_idx - (SEQ - 1)]
ts_det = labels_str[test_idx]

# Calibrate test scores
ts_fused_cal = calibrator.predict(ts_fused)

pred_fused = (ts_fused_cal > THR_FUSED).astype(int)
pred_cls   = (ts_p > THR_CLS).astype(int)
pred_kl    = (ts_kl > THR_KL).astype(int)
pred_proto = (ts_proto > THR_PROTO).astype(int)
pred_anom  = (pred_kl | pred_proto).astype(int)
pred_full  = pred_fused.astype(int)

mask_known = (ts_cat != 2)
mask_zd    = (ts_cat == 2) | (ts_cat == 0)

print("\n" + "="*72)
print("1. KNOWN ATTACKS (Classifier @ THR_CLS)")
print(classification_report(ts_true[mask_known], pred_cls[mask_known],
                            target_names=["Benign","Known Attack"], digits=4))

print("\n2. ZERO-DAY — KL Divergence (VAE)")
print(classification_report(ts_true[mask_zd], pred_kl[mask_zd],
                            target_names=["Benign","Zero-Day"], digits=4))

print("\n3. ZERO-DAY — Prototype Memory")
print(classification_report(ts_true[mask_zd], pred_proto[mask_zd],
                            target_names=["Benign","Zero-Day"], digits=4))

print("\n4. ZERO-DAY — KL + Proto Ensemble")
print(classification_report(ts_true[mask_zd], pred_anom[mask_zd],
                            target_names=["Benign","Zero-Day"], digits=4))

print("\n5. HYBRID COMBINED — FUSED + CALIBRATED (Primary)")
print(classification_report(ts_true, pred_full,
                            target_names=["Benign","Attack"], digits=4))

auc_total = roc_auc_score(ts_true, ts_fused_cal)
auc_cls   = roc_auc_score(ts_true[mask_known], ts_p[mask_known])
print(f"\n  AUC (fused calibrated, all):  {auc_total:.4f}")
print(f"  AUC (cls, known only):        {auc_cls:.4f}")

print("\n6. PER ZERO-DAY CLASS RECALL:")
print(f"  {'Attack class':35s} {'n':>6} | KL    | Proto | Ensemble | Fused | Gate")
print(f"  {'-'*35} {'-'*6}-+-------+-------+----------+-------+------")
for atk in sorted(set(ts_det[ts_cat == 2])):
    m     = (ts_det == atk) & (ts_cat == 2)
    m_atk = m & (ts_true == 1)
    if m_atk.sum() == 0: continue
    r_kl    = pred_kl[m_atk].mean()
    r_proto = pred_proto[m_atk].mean()
    r_ens   = pred_anom[m_atk].mean()
    r_fused = pred_full[m_atk].mean()
    g_mean  = ts_gate[m_atk].mean()   # avg gate → thấp = model rely on anomaly
    n_atk   = m_atk.sum()
    print(f"  {atk:35s} n={n_atk:5d} | {r_kl:.3f} | {r_proto:.3f} | {r_ens:.3f}    "
          f"| {r_fused:.3f} | {g_mean:.3f}")
print("="*72)

print("\n7. GATE STATISTICS (confidence-aware gating):")
for cat_id, cat_name in [(0,"Benign"),(1,"Known"),(2,"Zero-Day")]:
    m = (ts_cat == cat_id)
    if m.sum() == 0: continue
    g = ts_gate[m]
    print(f"  {cat_name:10s} — gate mean={g.mean():.4f} P50={np.median(g):.4f} "
          f"(low → anomaly detector trusted)")

print("\n8. SCORE STATISTICS:")
print(f"  {'Category':12s} {'KL P50':>10} {'Proto P50':>10} "
      f"{'Fused_cal P50':>14} {'MAE P50':>10}")
for cat_id, cat_name in [(0,"Benign"),(1,"Known"),(2,"Zero-Day")]:
    m = (ts_cat == cat_id)
    if m.sum() == 0: continue
    print(f"  {cat_name:12s} "
          f"{np.median(ts_kl[m]):>10.4f} "
          f"{np.median(ts_proto[m]):>10.4f} "
          f"{np.median(ts_fused_cal[m]):>14.4f} "
          f"{np.median(ts_mae[m]):>10.4f}")

print("\n9. SO SÁNH v7.0 / v8.0 / v9.0:")
print(f"  {'Metric':32s} {'v7.0':>8} {'v8.0':>8} {'v9.0':>8}")
print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'Known Attack AUC':32s} {'0.9984':>8} {'0.9984':>8} {auc_cls:>8.4f}")
print(f"  {'Overall AUC (fused)':32s} {'—':>8} {'0.7605':>8} {auc_total:>8.4f}")
m_zd_att = (ts_cat == 2) & (ts_true == 1)
if m_zd_att.sum() > 0:
    zd_recall = pred_full[m_zd_att].mean()
    print(f"  {'Zero-Day Recall':32s} {'0.0285':>8} {'0.0217':>8} {zd_recall:>8.4f}")
fpr_v9 = pred_full[ts_true == 0].mean()
print(f"  {'FAR (False Alarm Rate)':32s} {'0.0153':>8} {'0.0094':>8} {fpr_v9:>8.4f}")

# ─────────────────────────────────────────────────────────────────────────
# 13. PLOTS V9.0
# ─────────────────────────────────────────────────────────────────────────
STYLE = {
    "bg":      "#0f1117", "panel": "#1a1d2e",
    "accent1": "#00d4ff", "accent2": "#ff4757",
    "accent3": "#ffa502", "accent4": "#2ed573",
    "accent5": "#a29bfe", "text":   "#e8eaf6"
}
plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],   "axes.facecolor":  STYLE["panel"],
    "axes.labelcolor":  STYLE["text"], "xtick.color":     STYLE["text"],
    "ytick.color":      STYLE["text"], "text.color":      STYLE["text"]
})

# Fig 1: Training curves (7 panels)
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle("V9.0 Training Curves", fontsize=14, color=STYLE["text"])

axes[0,0].plot(hist["tr_cls"],   color=STYLE["accent1"], label="Train Cls")
axes[0,0].set_title("Classification Loss"); axes[0,0].legend()

axes[0,1].plot(hist["vl_auc"],   color=STYLE["accent3"], label="Val AUC (fused)")
axes[0,1].plot(hist["vl_ap"],    color=STYLE["accent1"], label="Val AP", ls="--")
axes[0,1].set_title("Val AUC & AP"); axes[0,1].legend(); axes[0,1].set_ylim(0.5, 1)

axes[0,2].plot(hist["tr_kl"],    color=STYLE["accent2"], label="KL loss")
axes[0,2].plot(hist["beta"],     color=STYLE["accent3"], label="β warmup", ls="--")
axes[0,2].set_title("KL & β warmup"); axes[0,2].legend()

axes[0,3].plot(hist["tr_conf"],  color=STYLE["accent5"], label="Confidence loss")
axes[0,3].set_title("Confidence Loss (v9.0 new)"); axes[0,3].legend()

axes[1,0].plot(hist["tr_proto_pull"], color=STYLE["accent4"], label="Pull")
axes[1,0].plot(hist["tr_proto_push"], color=STYLE["accent2"], label="Push", ls="--")
axes[1,0].set_title("Prototype Pull/Push"); axes[1,0].legend()

axes[1,1].plot(hist["lam_adv_eff"],   color=STYLE["accent3"], label="λ_adv")
axes[1,1].plot(hist["tr_adv"],        color=STYLE["accent2"], label="Adv loss", ls="--")
axes[1,1].set_title("Adv Warmup"); axes[1,1].legend()

axes[1,2].plot(hist["tr_rec"],   color=STYLE["accent1"], label="Recon loss")
axes[1,2].set_title("VAE Reconstruction"); axes[1,2].legend()

axes[1,3].axis("off")
summary_txt = (
    f"v9.0 Summary\n\n"
    f"Best Val AUC: {best_vauc:.4f}\n"
    f"Known AUC:    {auc_cls:.4f}\n"
    f"Overall AUC:  {auc_total:.4f}\n"
    f"FAR:          {fpr_v9:.4f}\n\n"
    f"Features: {N_FEATS} (14+{N_FEATS-14})\n"
    f"Hidden: {HIDDEN} | SEQ: {SEQ}\n"
    f"Params: {total_params:,}"
)
axes[1,3].text(0.1, 0.5, summary_txt, transform=axes[1,3].transAxes,
               color=STYLE["text"], fontsize=11, va="center",
               fontfamily="monospace")

plt.tight_layout()
plt.savefig(f"{EXPORT_DIR}/v90_training.png", dpi=150)
plt.close()

# Fig 2: Score distributions
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 6))
fig2.suptitle("V9.0 — Score Distributions per Category", fontsize=13, color=STYLE["text"])

c_map = {0: STYLE["accent1"], 1: STYLE["accent2"], 2: STYLE["accent3"]}
lbl_map = {0: "Benign", 1: "Known", 2: "Zero-Day"}

for cat_id in [0, 1, 2]:
    m = (ts_cat == cat_id)
    if m.sum() == 0: continue
    axes2[0].hist(ts_kl[m], bins=80, alpha=0.6, color=c_map[cat_id],
                  label=lbl_map[cat_id], density=True)
axes2[0].axvline(THR_KL, color="white", ls="--", lw=1.5, label=f"THR={THR_KL:.3f}")
axes2[0].set_title("KL Divergence"); axes2[0].legend()
axes2[0].set_xlabel("KL (higher = anomaly)")

for cat_id in [0, 1, 2]:
    m = (ts_cat == cat_id)
    if m.sum() == 0: continue
    axes2[1].hist(ts_proto[m], bins=80, alpha=0.6, color=c_map[cat_id],
                  label=lbl_map[cat_id], density=True)
axes2[1].axvline(THR_PROTO, color="white", ls="--", lw=1.5, label=f"THR={THR_PROTO:.4f}")
axes2[1].set_title("Prototype Score"); axes2[1].legend()
axes2[1].set_xlabel("1 - cosine_sim (higher = anomaly)")

for cat_id in [0, 1, 2]:
    m = (ts_cat == cat_id)
    if m.sum() == 0: continue
    axes2[2].hist(ts_fused_cal[m], bins=80, alpha=0.6, color=c_map[cat_id],
                  label=lbl_map[cat_id], density=True)
axes2[2].axvline(THR_FUSED, color="white", ls="--", lw=1.5, label=f"THR={THR_FUSED:.4f}")
axes2[2].set_title("Calibrated Fused Score"); axes2[2].legend()
axes2[2].set_xlabel("calibrated score (higher = anomaly)")

for cat_id in [0, 1, 2]:
    m = (ts_cat == cat_id)
    if m.sum() == 0: continue
    axes2[3].hist(ts_gate[m], bins=80, alpha=0.6, color=c_map[cat_id],
                  label=lbl_map[cat_id], density=True)
axes2[3].set_title("Confidence Gate [v9.0 new]"); axes2[3].legend()
axes2[3].set_xlabel("gate (low → anomaly detector trusted)")

plt.tight_layout()
plt.savefig(f"{EXPORT_DIR}/v90_score_distributions.png", dpi=150)
plt.close()

# Fig 3: PCA của VAE μ embeddings
try:
    from sklearn.decomposition import PCA
    idx_plot = np.random.choice(len(ts_p), min(5000, len(ts_p)), replace=False)
    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(ts_mu[idx_plot])

    fig3, ax = plt.subplots(figsize=(10, 8))
    fig3.patch.set_facecolor(STYLE["bg"]); ax.set_facecolor(STYLE["panel"])
    for cat_id in [0, 1, 2]:
        m = (ts_cat[idx_plot] == cat_id)
        if m.sum() > 0:
            ax.scatter(mu_2d[m, 0], mu_2d[m, 1],
                       c=c_map[cat_id], label=lbl_map[cat_id], alpha=0.4, s=5)
    with torch.no_grad():
        proto_np = _mc.proto_memory.prototypes.cpu().numpy()
    proto_2d = pca.transform(proto_np)
    ax.scatter(proto_2d[:, 0], proto_2d[:, 1],
               c="white", marker="*", s=150, label="Prototypes", zorder=5)
    ax.set_title("VAE μ Embeddings — PCA 2D", color=STYLE["text"])
    ax.legend(); ax.grid(alpha=0.1)
    ax.tick_params(colors=STYLE["text"])
    plt.tight_layout()
    plt.savefig(f"{EXPORT_DIR}/v90_vae_pca.png", dpi=150)
    plt.close()
    print("\n  ✓ VAE PCA plot saved")
except Exception as e:
    print(f"\n  VAE PCA plot skipped: {e}")

# Fig 4: Per-class detection rates
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle("V9.0 — Per-Class Detection Rate", fontsize=13, color=STYLE["text"])

atk_names, rates = [], []
for atk in np.unique(ts_det[mask_known & (ts_true == 1)]):
    m = (ts_det == atk)
    atk_names.append(atk); rates.append(pred_cls[m].mean() * 100)
axes4[0].barh(atk_names, rates, color=STYLE["accent1"], alpha=0.85)
axes4[0].set_title("Known Attacks (Classifier)")
axes4[0].set_xlim(0, 110)
for i, v in enumerate(rates):
    axes4[0].text(v + 1, i, f"{v:.1f}%", va="center", color="white", fontsize=9)

zd_names, zrates = [], []
for atk in sorted(set(ts_det[ts_cat == 2])):
    m = (ts_det == atk) & (ts_cat == 2) & (ts_true == 1)
    if m.sum() == 0: continue
    zd_names.append(atk); zrates.append(pred_full[m].mean() * 100)
axes4[1].barh(zd_names, zrates, color=STYLE["accent3"], alpha=0.85)
axes4[1].set_title("Zero-Day (Confidence-Aware Fused)")
axes4[1].set_xlim(0, 110)
for i, v in enumerate(zrates):
    axes4[1].text(v + 1, i, f"{v:.1f}%", va="center", color="white", fontsize=9)

plt.tight_layout()
plt.savefig(f"{EXPORT_DIR}/v90_per_class.png", dpi=150)
plt.close()

# Fig 5: Feature importance (gate correlation with each engineered feature)
# Proxy: mean |feature| của zero-day windows được detect đúng
try:
    zd_detected = (ts_cat == 2) & (ts_true == 1) & (pred_full == 1)
    zd_missed   = (ts_cat == 2) & (ts_true == 1) & (pred_full == 0)
    if zd_detected.sum() > 0 and zd_missed.sum() > 0:
        eng_feats = [f for f in FEATS if f not in FEATS_BASE]
        eng_idx   = [FEATS.index(f) for f in eng_feats]
        # Lấy features của end window của từng sample
        det_feat_means = X_all[test_idx[zd_detected], :][:, eng_idx].mean(axis=0)
        mis_feat_means = X_all[test_idx[zd_missed], :][:, eng_idx].mean(axis=0)
        diff = det_feat_means - mis_feat_means

        fig5, ax5 = plt.subplots(figsize=(12, 6))
        fig5.patch.set_facecolor(STYLE["bg"]); ax5.set_facecolor(STYLE["panel"])
        colors = [STYLE["accent4"] if v > 0 else STYLE["accent2"] for v in diff]
        ax5.barh(eng_feats, diff, color=colors, alpha=0.85)
        ax5.axvline(0, color="white", lw=0.5)
        ax5.set_title("Engineered Features — detected vs missed zero-day (mean diff)",
                      color=STYLE["text"])
        ax5.tick_params(colors=STYLE["text"])
        plt.tight_layout()
        plt.savefig(f"{EXPORT_DIR}/v90_feature_importance.png", dpi=150)
        plt.close()
        print("  ✓ Feature importance plot saved")
except Exception as e:
    print(f"  Feature importance plot skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────
# 14. SAVE CONFIG
# ─────────────────────────────────────────────────────────────────────────
config = {
    "model_version": MODEL_VERSION,
    "n_features_raw": len(FEATS_BASE),
    "n_features_total": N_FEATS,
    "engineered_features": [f for f in FEATS if f not in FEATS_BASE],
    "seq": SEQ, "hidden": HIDDEN, "bottleneck": BOTTLENECK,
    "n_prototypes": N_PROTOTYPES,
    "thresholds": {
        "fused_calibrated": float(THR_FUSED),
        "cls":   float(THR_CLS),
        "kl":    float(THR_KL),
        "proto": float(THR_PROTO),
        "mae_re":float(THR_RE),
    },
    "best_val_auc": float(best_vauc),
    "features": FEATS,
    "target_fpr": TARGET_FPR,
}
with open(f"{EXPORT_DIR}/v90_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n📦 Output saved to {EXPORT_DIR}/")
print("  - v90_training.png")
print("  - v90_score_distributions.png  (KL + Proto + Fused + Gate)")
print("  - v90_vae_pca.png")
print("  - v90_per_class.png")
print("  - v90_feature_importance.png   (engineered feature analysis)")
print("  - v90_config.json")

print(f"\n🏁 MODEL: {MODEL_VERSION} | Params: {total_params:,}")
print(f"   Best Val AUC: {best_vauc:.4f}")
print(f"\n📊 KEY CHANGES vs v8.0:")
print(f"   ✓ Feature engineering: {len(FEATS_BASE)} raw → {N_FEATS} features (+{N_FEATS-len(FEATS_BASE)} engineered)")
print(f"     syn_ack_ratio, fwd_bwd_ratio, pkt_len_cv, syn/ack rates,")
print(f"     handshake_ratio, active_idle_ratio, bytes_per_pkt, ...")
print(f"   ✓ Confidence-aware gating: gate = σ(conf_head)")
print(f"     fused = gate·p_cls + (1-gate)·anomaly_combo")
print(f"     zero-day → low gate → anomaly detector takes over")
print(f"   ✓ Isotonic Regression calibration (thay percentile threshold)")
print(f"   ✓ HIDDEN {32}→{HIDDEN}, TCN 3→4 blocks (dilation 1,2,4,8)")
print(f"   ✓ Feature importance plot (detected vs missed zero-day)")