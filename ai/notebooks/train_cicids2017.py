# ══════════════════════════════════════════════════════════════
# TRAIN TransformerVAE TRÊN CIC-IDS2017  —  Phiên bản β=0.1
# ✅ Backup model cũ → Train lại từ đầu → Lưu tất cả lên Drive
#
# CÁCH THỰC HIỆN:
#   Bước 1: Mở Google Colab
#   Bước 2: Runtime → Change runtime type → T4 GPU → Save
#   Bước 3: Copy toàn bộ file này vào 1 cell duy nhất
#   Bước 4: Nhấn ▶ Run — khoảng 30 phút là xong
#
# KHI BỊ NGẮT GIỮA CHỪNG:
#   Chạy lại cell → tự mount Drive → tự resume checkpoint → tiếp tục
# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# BƯỚC 0: MOUNT GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────
from google.colab import drive
print("Đang kết nối Google Drive...")
drive.mount('/content/drive', force_remount=True)

import os, shutil

DRIVE_DIR   = "/content/drive/MyDrive/DDoS_AI_Project"
os.makedirs(DRIVE_DIR, exist_ok=True)

# Đường dẫn — TẤT CẢ lưu trên Drive (không bao giờ mất)
DRIVE_ZIP   = f"{DRIVE_DIR}/CIC-IDS2017.zip"
CKPT_PATH   = f"{DRIVE_DIR}/checkpoint_beta01.pth"   # Checkpoint riêng cho β=0.1
MODEL_BEST  = f"{DRIVE_DIR}/model_beta01_best.pth"   # Model β=0.1 tốt nhất
MODEL_OLD   = f"{DRIVE_DIR}/model_best.pth"           # Model cũ β=0.01 (giữ nguyên!)
SCALER_PATH = f"{DRIVE_DIR}/scaler_beta01.pkl"        # Scaler dùng cho β=0.1
LOCAL_ZIP   = "/content/CIC-IDS2017.zip"
LOCAL_DIR   = "/content/CIC-IDS2017"

print(f"✅ Drive kết nối! Thư mục: {DRIVE_DIR}")

# Backup model cũ (β=0.01) nếu có, để không bị mất
if os.path.exists(MODEL_OLD):
    bak = f"{DRIVE_DIR}/model_beta001_backup.pth"
    if not os.path.exists(bak):
        shutil.copy(MODEL_OLD, bak)
        print(f"✅ Backup model cũ (β=0.01) → {bak}")
    else:
        print(f"✅ Backup model cũ đã có: {bak}")

# Lấy file zip từ Drive hoặc upload lần đầu
if os.path.exists(DRIVE_ZIP):
    if not os.path.exists(LOCAL_ZIP):
        print("Copy zip từ Drive về /content/...")
        shutil.copy(DRIVE_ZIP, LOCAL_ZIP)
        print("✅ Copy xong!")
    else:
        print("✅ Zip đã có trong /content/")
else:
    print("\n⚠️  Chưa có CIC-IDS2017.zip trên Drive. Hãy upload...")
    from google.colab import files
    files.upload()
    if os.path.exists(LOCAL_ZIP):
        shutil.copy(LOCAL_ZIP, DRIVE_ZIP)
        print(f"✅ Đã backup zip lên Drive. Từ nay không cần upload lại!")
    else:
        raise FileNotFoundError("Không tìm thấy file zip. Hãy thử lại.")


# ─────────────────────────────────────────────────────────────
# BƯỚC 1: THƯ VIỆN VÀ THAM SỐ
# ─────────────────────────────────────────────────────────────
import zipfile, glob, pickle, gc
import torch, torch.nn as nn
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {DEVICE}")
if DEVICE == "cpu":
    raise RuntimeError(
        "\n" + "="*55 +
        "\n⛔ KHÔNG CÓ GPU! Không thể train trên CPU." +
        "\nLàm theo các bước sau:" +
        "\n  1. Runtime → Change runtime type" +
        "\n  2. Hardware accelerator → T4 GPU" +
        "\n  3. Click Save → Session sẽ restart" +
        "\n  4. Chạy lại cell này" +
        "\n" + "="*55
    )
print("✅ GPU sẵn sàng!")

# ── Tham số model ─────────────────────────────────────────────
SEQ          = 10      # Số flow trong 1 sliding window
LATENT       = 32      # Kích thước không gian tiềm ẩn
NHEAD        = 2       # Attention heads
LAYERS       = 2       # Số lớp Transformer
DROPOUT      = 0.15    # Tỉ lệ dropout (BẮT BUỘC > 0 cho MC Dropout)
BETA         = 0.1     # ★ β=0.1: đủ nhỏ để MSE ưu tiên, đủ lớn để KLD > 0
TOTAL_EPOCHS = 80
BATCH        = 512
LR           = 3e-4

# ── 10 features từ CIC-IDS2017 ───────────────────────────────
FEATS_RAW = [
    'Flow Duration',            # DDoS flows rất ngắn
    'Total Fwd Packets',        # DDoS gửi cực nhiều packet
    'Total Backward Packets',   # Server quá tải → ít packet về
    'Flow Bytes/s',             # Tốc độ byte — DDoS rất cao
    'Flow Packets/s',           # Tốc độ packet — DDoS rất cao
    'Fwd IAT Mean',             # Thời gian giữa packet — DDoS ≈ 0
    'Packet Length Mean',       # Packet nhỏ để tối đa số lượng
    'SYN Flag Count',           # SYN Flood: hàng nghìn SYN
    'ACK Flag Count',           # ACK bất thường
    'Init_Win_bytes_forward',   # Window size — DDoS bot thường = 0
]

# Features có phân phối skewed → cần log1p để normalize tốt
LOG_FEATS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s',  'Flow Packets/s',    'Fwd IAT Mean',
    'Packet Length Mean', 'Init_Win_bytes_forward',
]


# ─────────────────────────────────────────────────────────────
# BƯỚC 2: GIẢI NÉN DATASET
# ─────────────────────────────────────────────────────────────
if not os.path.exists(LOCAL_DIR):
    print("Giải nén CIC-IDS2017.zip...")
    with zipfile.ZipFile(LOCAL_ZIP, 'r') as z:
        z.extractall(LOCAL_DIR)
    print("✅ Giải nén xong!")
else:
    print("✅ Dataset đã giải nén sẵn.")

csv_files = glob.glob(os.path.join(LOCAL_DIR, "**", "*.csv"), recursive=True)
print(f"Tìm thấy {len(csv_files)} file CSV")


# ─────────────────────────────────────────────────────────────
# BƯỚC 3: ĐỌC DỮ LIỆU (chỉ load cột cần thiết → tiết kiệm RAM)
# ─────────────────────────────────────────────────────────────
chunks = []
for f in csv_files:
    header = pd.read_csv(f, nrows=0)
    cols_needed = [c for c in header.columns
                   if any(c.strip() == r.strip() for r in FEATS_RAW)
                   or c.strip() == 'Label']
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
# BƯỚC 4: TIỀN XỬ LÝ + LOG1P TRANSFORM
# ─────────────────────────────────────────────────────────────
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS)
X_all = df[FEATS].values.astype(np.float32)
y_all = (df['Label'] != 'BENIGN').astype(np.uint8).values
del df; gc.collect()

# Log1p transform: kéo đuôi dài về → BENIGN và Attack tách biệt hơn
LOG_IDX = [FEATS.index(f) for f in LOG_FEATS if f in FEATS]
X_all[:, LOG_IDX] = np.log1p(np.abs(X_all[:, LOG_IDX]))
print(f"Log1p transform: {len(LOG_IDX)} features.")

# Tách train/val/test
idx_b = np.where(y_all == 0)[0]
idx_a = np.where(y_all == 1)[0]
del y_all; gc.collect()
np.random.seed(42); np.random.shuffle(idx_b)
split  = int(len(idx_b) * 0.8)
idx_tr = idx_b[:split]; idx_vl = idx_b[split:]; idx_ts = idx_a
del idx_b, idx_a; gc.collect()
print(f"Train:{len(idx_tr):,} | Val:{len(idx_vl):,} | Test(Attack):{len(idx_ts):,}")

# Scaler — load từ Drive nếu đã có (tức là đang resume)
if os.path.exists(SCALER_PATH):
    print("✅ Load scaler từ Drive (resume)...")
    with open(SCALER_PATH, 'rb') as f: sc = pickle.load(f)
    Xtr_raw = sc.transform(X_all[idx_tr])
else:
    print("Fit scaler mới...")
    sc = MinMaxScaler()
    Xtr_raw = sc.fit_transform(X_all[idx_tr])
    with open(SCALER_PATH, 'wb') as f: pickle.dump(sc, f)
    print(f"✅ Scaler lưu: {SCALER_PATH}")

Xvl_raw = sc.transform(X_all[idx_vl])
Xts_raw = sc.transform(X_all[idx_ts])
del X_all, idx_tr, idx_vl, idx_ts; gc.collect()


# ─────────────────────────────────────────────────────────────
# BƯỚC 5: SLIDING WINDOWS (Zero-copy, tiết kiệm RAM)
# ─────────────────────────────────────────────────────────────
def make_windows(data, s):
    n, f = data.shape
    if n < s: return np.empty((0, s, f), dtype=np.float32)
    shape   = (n - s + 1, s, f)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    ).copy().astype(np.float32)

print("Tạo sliding windows...")
Wtr = make_windows(Xtr_raw, SEQ); del Xtr_raw; gc.collect(); print(f"  Train: {Wtr.shape}")
Wvl = make_windows(Xvl_raw, SEQ); del Xvl_raw; gc.collect(); print(f"  Val  : {Wvl.shape}")
Wts = make_windows(Xts_raw, SEQ); del Xts_raw; gc.collect(); print(f"  Test : {Wts.shape}")
Ttr = torch.tensor(Wtr); del Wtr; gc.collect()
Tvl = torch.tensor(Wvl); del Wvl; gc.collect()
Tts = torch.tensor(Wts); del Wts; gc.collect()
print("✅ Windows xong!")


# ─────────────────────────────────────────────────────────────
# BƯỚC 6: MÔ HÌNH TransformerVAE
# ─────────────────────────────────────────────────────────────
class TransformerVAE(nn.Module):
    def __init__(self, F, S, L, H, N, D):
        super().__init__(); self.F=F; self.S=S
        enc = nn.TransformerEncoderLayer(F, H, F*8, D, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc, N)
        self.fc_mu    = nn.Linear(F*S, L)
        self.fc_logvar= nn.Linear(F*S, L)
        self.dec_proj = nn.Linear(L, F*S)
        dec = nn.TransformerDecoderLayer(F, H, F*8, D, batch_first=True)
        self.decoder  = nn.TransformerDecoder(dec, N)
        self.fc_out   = nn.Linear(F, F)

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def decode(self, z):
        p = self.dec_proj(z).view(-1, self.S, self.F)
        return self.fc_out(self.decoder(p, p))

    def forward(self, x):
        mu, lv = self.encode(x)
        return self.decode(self.reparameterize(mu, lv)), mu, lv


def vae_loss(recon, x, mu, lv, beta=1.0):
    """
    Loss = MSE + β × KLD
    β=0.1: KLD đủ lớn để encoder học (KLD > 0),
           MSE vẫn chiếm ưu thế để phân biệt BENIGN/Attack
    """
    mse = nn.functional.mse_loss(recon, x, reduction='mean')
    kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    return mse + beta * kld, mse, kld


model = TransformerVAE(FEATURE_SIZE, SEQ, LATENT, NHEAD, LAYERS, DROPOUT).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS, eta_min=1e-5)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params | β={BETA}")


# ─────────────────────────────────────────────────────────────
# BƯỚC 7: LOAD CHECKPOINT (Resume nếu bị ngắt)
# ─────────────────────────────────────────────────────────────
start_epoch = 1
best        = float('inf')
hist        = {'tr': [], 'vl': [], 'mse': [], 'kld': []}

if os.path.exists(CKPT_PATH):
    print(f"\n{'='*55}")
    print(f">>> Tìm thấy checkpoint β=0.1! Đang resume...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    sch.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best        = ckpt['best_val']
    hist        = ckpt.get('history', hist)
    print(f">>> Resume từ epoch {start_epoch}/{TOTAL_EPOCHS} | best={best:.6f}")
    print(f"{'='*55}\n")
else:
    print(f"\n>>> Bắt đầu train mới (β={BETA}, epoch 1/{TOTAL_EPOCHS})")


# ─────────────────────────────────────────────────────────────
# BƯỚC 8: TRAIN
# ─────────────────────────────────────────────────────────────
if start_epoch <= TOTAL_EPOCHS:
    ldr = DataLoader(TensorDataset(Ttr), batch_size=BATCH, shuffle=True,
                     drop_last=True, pin_memory=(DEVICE=='cuda'), num_workers=2)
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

        nb  = len(ldr)
        atr = tl/nb; hist['tr'].append(atr)
        hist['mse'].append(tm/nb); hist['kld'].append(tk/nb)

        model.eval()
        with torch.no_grad():
            vb = Tvl.to(DEVICE); rv, mv, lv2 = model(vb)
            avl = vae_loss(rv, vb, mv, lv2, BETA)[0].item()
        hist['vl'].append(avl); sch.step()

        if avl < best:
            best = avl
            torch.save(model.state_dict(), MODEL_BEST)

        # Lưu checkpoint lên Drive sau mỗi epoch
        torch.save({
            'epoch': ep, 'model': model.state_dict(),
            'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(),
            'best_val': best, 'history': hist,
        }, CKPT_PATH)

        lr_now = opt.param_groups[0]['lr']
        print(f"[{ep:3d}/{TOTAL_EPOCHS}] "
              f"train={atr:.5f} "
              f"(MSE={hist['mse'][-1]:.5f}  KLD={hist['kld'][-1]:.5f}) "
              f"val={avl:.5f}  best={best:.5f}  lr={lr_now:.1e}")

    print(f"\n✅ Train xong! Model lưu: {MODEL_BEST}")

    # Learning Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(hist['tr'], label='Train', color='steelblue')
    axes[0].plot(hist['vl'], label='Val',   color='tomato', linestyle='--')
    axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(hist['mse'], label='MSE (Recon)', color='green')
    axes[1].plot(hist['kld'], label='KLD',         color='orange', linestyle='--')
    axes[1].set_title(f'MSE vs KLD (β={BETA})')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle(f'Learning Curve — β={BETA}, latent={LATENT}, log-transform')
    plt.tight_layout(); plt.show()
else:
    print("Train đã xong! Tiếp tục đến phần Evaluation.")


# ─────────────────────────────────────────────────────────────
# BƯỚC 9: TÍNH NGƯỠNG (MC Dropout)
# ─────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_BEST, map_location=DEVICE))


def mc_eval(model, tensor, passes=15, bs=2048):
    """
    MC Dropout: Chạy 15 lần với dropout BẬT (model.train())
    → RE  = sai số tái tạo trung bình  (cao = bất thường)
    → U   = phương sai giữa các lần    (cao = chưa học bao giờ)
    """
    model.train()
    all_RE, all_U = [], []
    for i in range(0, len(tensor), bs):
        batch = tensor[i:i+bs].to(DEVICE); preds = []
        with torch.no_grad():
            for _ in range(passes):
                r, _, _ = model(batch)
                preds.append(r.cpu().numpy())
        P = np.array(preds); D = batch.cpu().numpy()
        all_RE.append(np.mean((P.mean(0) - D) ** 2, axis=(1, 2)))
        all_U.append(np.mean(np.var(P, axis=0),      axis=(1, 2)))
        if i % (bs * 20) == 0: print(f"  {i}/{len(tensor)}...")
    return np.concatenate(all_RE), np.concatenate(all_U)


print("Tính threshold trên Val (BENIGN)...")
RE_v, U_v = mc_eval(model, Tvl, passes=15)

THR_RE = float(np.percentile(RE_v, 90))   # 90th percentile BENIGN
THR_U  = float(np.percentile(U_v, 95))    # 95th percentile

print(f"\n{'='*55}")
print(f"  ★ threshold_re = {THR_RE:.6f}")
print(f"  ★ threshold_u  = {THR_U:.10f}")
print(f"{'='*55}")
print(f"RE val — mean={RE_v.mean():.5f}, std={RE_v.std():.5f}")
print(f"U  val — mean={U_v.mean():.2e}, std={U_v.std():.2e}, max={U_v.max():.2e}")


# ─────────────────────────────────────────────────────────────
# BƯỚC 10: ĐÁNH GIÁ TRÊN TẬP ATTACK
# ─────────────────────────────────────────────────────────────
print("\nĐánh giá trên Test (Attack)...")
RE_t, U_t = mc_eval(model, Tts, passes=15)

RE_all = np.concatenate([RE_v,  RE_t])
y_true = np.array([0]*len(RE_v) + [1]*len(RE_t), dtype=int)
y_pred = (RE_all > THR_RE).astype(int)

print(f"\n{'='*65}")
print(f"KẾT QUẢ — CIC-IDS2017 | β={BETA} | log-transform | latent={LATENT}")
print(f"{'='*65}")
print(classification_report(y_true, y_pred,
      target_names=['BENIGN', 'Attack'], digits=4))

# Biểu đồ
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Pred:Normal', 'Pred:Attack'],
            yticklabels=['True:BENIGN', 'True:Attack'])
axes[0].set_title(f'Confusion Matrix (β={BETA}, log-transform)')

axes[1].hist(RE_v[::5], bins=120, alpha=0.6, label='BENIGN',
             color='steelblue', density=True)
axes[1].hist(RE_t[::5], bins=120, alpha=0.6, label='Attack',
             color='tomato', density=True)
axes[1].axvline(THR_RE, color='red', linestyle='--', linewidth=2,
                label=f'threshold_re={THR_RE:.4f}')
axes[1].set_xlabel('Reconstruction Error')
axes[1].legend(); axes[1].set_title('RE: BENIGN vs Attack')
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────
# BƯỚC 11: DETECTION RATE THEO TỪNG LOẠI TẤN CÔNG
# ─────────────────────────────────────────────────────────────
print(f"\n{'Attack Type':<42} {'N':>8}  {'RE_mean':>9}  {'Det%':>7}")
print("─" * 72)

csv_files2 = glob.glob(os.path.join(LOCAL_DIR, "**", "*.csv"), recursive=True)
dfs2 = []
for f in csv_files2:
    header = pd.read_csv(f, nrows=0)
    cols_needed = [c for c in header.columns
                   if any(c.strip() == r.strip() for r in FEATS_RAW)
                   or c.strip() == 'Label']
    tmp = pd.read_csv(f, usecols=cols_needed, low_memory=False)
    tmp.columns = tmp.columns.str.strip()
    dfs2.append(tmp)
df2 = pd.concat(dfs2, ignore_index=True); del dfs2; gc.collect()
df2['Label'] = df2['Label'].str.strip()
df_atk = df2[df2['Label'] != 'BENIGN'].copy(); del df2; gc.collect()

for attack_type in sorted(df_atk['Label'].unique()):
    sub = df_atk[df_atk['Label'] == attack_type]
    if len(sub) < SEQ + 5: continue
    Xsub = sub[[f for f in FEATS if f in sub.columns]].values.astype(np.float32)
    Xsub[:, LOG_IDX] = np.log1p(np.abs(Xsub[:, LOG_IDX]))
    Xsub_sc = sc.transform(Xsub)
    Wsub = make_windows(Xsub_sc, SEQ)
    if len(Wsub) == 0: continue
    RE_sub, _ = mc_eval(model, torch.tensor(Wsub), passes=5, bs=2048)
    det = (RE_sub > THR_RE).mean() * 100
    print(f"  {attack_type:<40} {len(Wsub):>8,}  {RE_sub.mean():>9.5f}  {det:>6.1f}%")


# ─────────────────────────────────────────────────────────────
# BƯỚC 12: TẢI MODEL VỀ MÁY (Drive đã lưu rồi, đây là backup local)
# ─────────────────────────────────────────────────────────────
from google.colab import files as colab_files
colab_files.download(MODEL_BEST)
colab_files.download(SCALER_PATH)

print(f"""
╔══════════════════════════════════════════════════════╗
║  KẾT QUẢ PHIÊN TRAIN β=0.1                         ║
╠══════════════════════════════════════════════════════╣
║  Google Drive: {DRIVE_DIR}
║    ├── model_beta01_best.pth  ← model mới (β=0.1)  ║
║    ├── model_beta001_backup   ← model cũ (β=0.01)  ║
║    ├── checkpoint_beta01.pth  ← resume khi ngắt    ║
║    ├── scaler_beta01.pkl                            ║
║    └── CIC-IDS2017.zip        ← không cần upload   ║
╠══════════════════════════════════════════════════════╣
║  Cập nhật config.yaml:                              ║
║    model:                                           ║
║      feature_size: {FEATURE_SIZE}                              ║
║      latent_dim:   {LATENT}                              ║
║      nhead:        {NHEAD}                               ║
║      dropout:      {DROPOUT}                           ║
║    detection:                                       ║
║      threshold_re: {THR_RE:.6f}                   ║
║      threshold_u:  {THR_U:.8f}               ║
╚══════════════════════════════════════════════════════╝
""")
