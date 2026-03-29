# ══════════════════════════════════════════════════════════════
# TRAIN TransformerVAE — PHIÊN BẢN KAGGLE
# Không cần Google Drive, không cần upload zip
# CSV files đã có sẵn trong /kaggle/input/
# ══════════════════════════════════════════════════════════════

import os, glob, pickle, gc
import torch, torch.nn as nn
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────
WORK_DIR    = "/kaggle/working"
CKPT_PATH   = f"{WORK_DIR}/checkpoint_conv_beta01.pth"
MODEL_BEST  = f"{WORK_DIR}/model_conv_beta01_best.pth"
SCALER_PATH = f"{WORK_DIR}/scaler_conv_beta01.pkl"

SEQ          = 10
LATENT       = 32
NHEAD        = 2
LAYERS       = 2
DROPOUT      = 0.15
BETA         = 0.1
TOTAL_EPOCHS = 80
BATCH        = 512
LR           = 3e-4

FEATS_RAW = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Fwd IAT Mean',
    'Packet Length Mean', 'SYN Flag Count', 'ACK Flag Count',
    'Init_Win_bytes_forward',
]
LOG_FEATS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Fwd IAT Mean',
    'Packet Length Mean', 'Init_Win_bytes_forward',
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
# ĐỌC DỮ LIỆU — CSV có sẵn trong /kaggle/input/
# ─────────────────────────────────────────────────────────────
csv_files = glob.glob("/kaggle/input/**/*.csv", recursive=True)
print(f"Tìm thấy {len(csv_files)} file CSV")
if len(csv_files) == 0:
    raise FileNotFoundError(
        "Không có CSV! Hãy click + Add Input → chọn dataset CIC-IDS2017"
    )

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

# Tách index
idx_b = np.where(y_all == 0)[0]
idx_a = np.where(y_all == 1)[0]
del y_all; gc.collect()
np.random.seed(42); np.random.shuffle(idx_b)
split  = int(len(idx_b) * 0.8)
idx_tr = idx_b[:split]; idx_vl = idx_b[split:]; idx_ts = idx_a
del idx_b, idx_a; gc.collect()
print(f"Train:{len(idx_tr):,} | Val:{len(idx_vl):,} | Test(Attack):{len(idx_ts):,}")

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
del X_all, idx_tr, idx_vl, idx_ts; gc.collect()

# ─────────────────────────────────────────────────────────────
# SLIDING WINDOWS
# ─────────────────────────────────────────────────────────────
def make_windows(data, s):
    n, f = data.shape
    if n < s: return np.empty((0, s, f), dtype=np.float32)
    shape   = (n - s + 1, s, f)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides).copy().astype(np.float32)

print("Tạo sliding windows...")
Wtr = make_windows(Xtr_raw, SEQ); del Xtr_raw; gc.collect(); print(f"  Train: {Wtr.shape}")
Wvl = make_windows(Xvl_raw, SEQ); del Xvl_raw; gc.collect(); print(f"  Val  : {Wvl.shape}")
Wts = make_windows(Xts_raw, SEQ); del Xts_raw; gc.collect(); print(f"  Test : {Wts.shape}")
Ttr = torch.tensor(Wtr); del Wtr; gc.collect()
Tvl = torch.tensor(Wvl); del Wvl; gc.collect()
Tts = torch.tensor(Wts); del Wts; gc.collect()
print("✅ Windows xong!")

# ─────────────────────────────────────────────────────────────
# MODEL TransformerVAE (CẢI TIẾN CHUYÊN GIA)
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
        x = x + self.pe[:x.size(1), :]
        return x

class TransformerVAE(nn.Module):
    def __init__(self, F, S, L, H, N, D):
        super().__init__(); self.F=F; self.S=S
        
        # Tích hợp Positional Encoding và CNN1D Extraction
        self.pos_encoder = PositionalEncoding(F, max_len=S)
        self.local_cnn = nn.Conv1d(in_channels=F, out_channels=F, kernel_size=3, padding=1)
        self.act_cnn = nn.GELU()
        
        enc = nn.TransformerEncoderLayer(F, H, F*8, D, batch_first=True, activation='gelu')
        self.encoder  = nn.TransformerEncoder(enc, N)
        
        self.dropout_z = nn.Dropout(D)
        self.fc_mu    = nn.Linear(F*S, L)
        self.fc_logvar= nn.Linear(F*S, L)
        self.dec_proj = nn.Linear(L, F*S)
        
        dec = nn.TransformerDecoderLayer(F, H, F*8, D, batch_first=True, activation='gelu')
        self.decoder  = nn.TransformerDecoder(dec, N)
        self.fc_out   = nn.Linear(F, F)

    def encode(self, x):
        x_cnn = x.transpose(1, 2)
        x_cnn = self.act_cnn(self.local_cnn(x_cnn))
        x_cnn = x_cnn.transpose(1, 2)
        
        x_emb = self.pos_encoder(x + x_cnn)
        
        h = self.encoder(x_emb).contiguous().view(x.size(0), -1)
        h = self.dropout_z(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def decode(self, z):
        p = self.dec_proj(z).view(-1, self.S, self.F)
        p = self.pos_encoder(p)
        return self.fc_out(self.decoder(p, p))

    def forward(self, x):
        mu, lv = self.encode(x)
        return self.decode(self.reparameterize(mu, lv)), mu, lv


def vae_loss(recon, x, mu, lv, beta=1.0):
    mse = nn.functional.mse_loss(recon, x, reduction='mean')
    kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    return mse + beta * kld, mse, kld


model = TransformerVAE(FEATURE_SIZE, SEQ, LATENT, NHEAD, LAYERS, DROPOUT).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS, eta_min=1e-5)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params | β={BETA}")

# ─────────────────────────────────────────────────────────────
# LOAD CHECKPOINT NẾU CÓ
# ─────────────────────────────────────────────────────────────
start_epoch = 1
best        = float('inf')
hist        = {'tr': [], 'vl': [], 'mse': [], 'kld': []}

if os.path.exists(CKPT_PATH):
    print(f"\n>>> Tìm thấy checkpoint! Đang resume...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    sch.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best        = ckpt['best_val']
    hist        = ckpt.get('history', hist)
    print(f">>> Resume từ epoch {start_epoch}/{TOTAL_EPOCHS}")
else:
    print(f"\n>>> Train từ đầu (β={BETA}, epoch 1/{TOTAL_EPOCHS})")

# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────
if start_epoch <= TOTAL_EPOCHS:
    ldr = DataLoader(TensorDataset(Ttr), batch_size=BATCH, shuffle=True,
                     drop_last=True, pin_memory=True, num_workers=4)
    print(f"Training | {len(ldr)} batches/epoch\n")

    for ep in range(start_epoch, TOTAL_EPOCHS + 1):
        model.train(); tl = tm = tk = 0.0
        for (b,) in ldr:
            b = b.to(DEVICE, non_blocking=True); opt.zero_grad()
            r, mu, lv = model(b)
            loss, mse, kld = vae_loss(r, b, mu, lv, BETA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); tm += mse.item(); tk += kld.item()

        nb = len(ldr)
        atr = tl/nb
        hist['tr'].append(atr); hist['mse'].append(tm/nb); hist['kld'].append(tk/nb)

        model.eval()
        with torch.no_grad():
            vb = Tvl.to(DEVICE); rv, mv, lv2 = model(vb)
            avl = vae_loss(rv, vb, mv, lv2, BETA)[0].item()
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
              f"(MSE={hist['mse'][-1]:.5f} KLD={hist['kld'][-1]:.5f}) "
              f"val={avl:.5f} best={best:.5f} lr={lr_now:.1e}")

    print(f"\n✅ Train xong!")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(hist['tr'], label='Train'); axes[0].plot(hist['vl'], label='Val', ls='--')
    axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(hist['mse'], label='MSE'); axes[1].plot(hist['kld'], label='KLD', ls='--')
    axes[1].set_title(f'MSE vs KLD (β={BETA})'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────
# THRESHOLD + ĐÁNH GIÁ
# ─────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_BEST, map_location=DEVICE))

def mc_eval(model, tensor, passes=15, bs=2048):
    model.train(); all_RE=[]; all_U=[]
    for i in range(0, len(tensor), bs):
        batch = tensor[i:i+bs].to(DEVICE); preds=[]
        with torch.no_grad():
            for _ in range(passes):
                r, _, _ = model(batch); preds.append(r.cpu().numpy())
        P = np.array(preds); D = batch.cpu().numpy()
        all_RE.append(np.mean((P.mean(0)-D)**2, axis=(1,2)))
        all_U.append(np.mean(np.var(P, axis=0),  axis=(1,2)))
        if i % (bs*20) == 0: print(f"  {i}/{len(tensor)}...")
    return np.concatenate(all_RE), np.concatenate(all_U)

print("Tính threshold..."); RE_v, U_v = mc_eval(model, Tvl)
THR_RE = float(np.percentile(RE_v, 90))
THR_U  = float(np.percentile(U_v, 95))
print(f"\n★ threshold_re = {THR_RE:.6f}")
print(f"★ threshold_u  = {THR_U:.10f}")

print("Đánh giá Attack..."); RE_t, U_t = mc_eval(model, Tts)
RE_all = np.concatenate([RE_v, RE_t])
y_true = np.array([0]*len(RE_v)+[1]*len(RE_t))
y_pred = (RE_all > THR_RE).astype(int)
print(f"\n{'='*65}")
print(classification_report(y_true, y_pred,
      target_names=['BENIGN','Attack'], digits=4))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            ax=axes[0], xticklabels=['Normal','Attack'],
            yticklabels=['BENIGN','Attack'])
axes[0].set_title(f'Confusion Matrix (β={BETA})')
axes[1].hist(RE_v[::5], bins=120, alpha=0.6, label='BENIGN', density=True)
axes[1].hist(RE_t[::5], bins=120, alpha=0.6, label='Attack', color='tomato', density=True)
axes[1].axvline(THR_RE, color='red', ls='--', label=f'thr={THR_RE:.4f}')
axes[1].legend(); axes[1].set_title('RE: BENIGN vs Attack')
plt.tight_layout(); plt.show()

print(f"""
╔══════════════════════════════════════════════════════╗
║  Model đã lưu tại /kaggle/working/                  ║
║  Vào Output panel bên phải để tải về                ║
╠══════════════════════════════════════════════════════╣
║  Cập nhật config.yaml:                              ║
║    feature_size: {FEATURE_SIZE}                              ║
║    latent_dim:   {LATENT}                              ║
║    threshold_re: {THR_RE:.6f}                   ║
║    threshold_u:  {THR_U:.8f}               ║
╚══════════════════════════════════════════════════════╝
""")
