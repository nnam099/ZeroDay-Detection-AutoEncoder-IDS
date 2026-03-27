# ══════════════════════════════════════════════════════════════
# TRAIN TransformerVAE TRÊN CIC-IDS2017
# ✅ Lưu tất cả vào Google Drive — không bao giờ mất dữ liệu
#
# LẦN ĐẦU CHẠY:
#   1. Runtime > Change runtime type > T4 GPU
#   2. Chạy cell → sẽ hỏi quyền truy cập Drive → đồng ý
#   3. Upload CIC-IDS2017.zip 1 lần duy nhất khi được hỏi
#   → Code tự lưu zip lên Drive, không cần upload lại bao giờ nữa
#
# LẦN SAU KHI BỊ NGẮT:
#   1. Mở lại Colab, chạy cell
#   → Tự mount Drive, tự resume checkpoint, tiếp tục train
# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# BƯỚC 0: MOUNT GOOGLE DRIVE — LƯU TẤT CẢ VÀO ĐÂY
# ─────────────────────────────────────────────────────────────
from google.colab import drive
print("Đang kết nối Google Drive...")
drive.mount('/content/drive', force_remount=True)

import os, shutil
DRIVE_DIR = "/content/drive/MyDrive/DDoS_AI_Project"
os.makedirs(DRIVE_DIR, exist_ok=True)
print(f"✅ Drive đã kết nối! Thư mục project: {DRIVE_DIR}")

# ── Đường dẫn — TẤT CẢ lưu trên Drive ───────────────────────
DRIVE_ZIP   = f"{DRIVE_DIR}/CIC-IDS2017.zip"
CKPT_PATH   = f"{DRIVE_DIR}/checkpoint_v3.pth"   # Resume khi bị ngắt
MODEL_BEST  = f"{DRIVE_DIR}/model_best.pth"       # Model tốt nhất
SCALER_PATH = f"{DRIVE_DIR}/scaler.pkl"           # Scaler MinMax
LOCAL_ZIP   = "/content/CIC-IDS2017.zip"          # Copy local để giải nén nhanh
LOCAL_DIR   = "/content/CIC-IDS2017"              # Thư mục giải nén

# ── Xử lý file zip ────────────────────────────────────────────
if os.path.exists(DRIVE_ZIP):
    # Zip đã có trên Drive → copy xuống /content/ để giải nén nhanh
    if not os.path.exists(LOCAL_ZIP):
        print("Copy zip từ Drive về /content/ (nhanh hơn giải nén từ Drive)...")
        shutil.copy(DRIVE_ZIP, LOCAL_ZIP)
        print("✅ Copy xong!")
    else:
        print("✅ Zip đã có sẵn trong /content/")
else:
    # Chưa có trên Drive → hỏi upload lần đầu
    print("\n" + "="*55)
    print("⚠️  Chưa có CIC-IDS2017.zip trên Drive!")
    print("    Hãy chọn file zip từ máy tính để upload...")
    print("="*55)
    from google.colab import files
    files.upload()   # Hiện nút "Choose Files"
    # Sau khi upload xong, backup lên Drive ngay
    if os.path.exists(LOCAL_ZIP):
        print("Đang backup zip lên Drive (chỉ làm 1 lần)...")
        shutil.copy(LOCAL_ZIP, DRIVE_ZIP)
        print(f"✅ Đã backup lên Drive: {DRIVE_ZIP}")
        print("→ Từ nay không cần upload lại nữa!")
    else:
        raise FileNotFoundError("Không tìm thấy CIC-IDS2017.zip. Hãy thử lại.")


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
    print("⚠️  Đang dùng CPU! Vào Runtime > Change runtime type > T4 GPU để train nhanh hơn.")

SEQ          = 10
LATENT       = 32
NHEAD        = 2
LAYERS       = 2
DROPOUT      = 0.15
BETA         = 0.01
TOTAL_EPOCHS = 80
BATCH        = 512
LR           = 3e-4

FEATS_RAW = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s',  'Flow Packets/s',    'Fwd IAT Mean',
    'Packet Length Mean', 'SYN Flag Count', 'ACK Flag Count',
    'Init_Win_bytes_forward',
]
LOG_FEATS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s',  'Flow Packets/s',    'Fwd IAT Mean',
    'Packet Length Mean', 'Init_Win_bytes_forward',
]


# ─────────────────────────────────────────────────────────────
# BƯỚC 2: GIẢI NÉN
# ─────────────────────────────────────────────────────────────
if not os.path.exists(LOCAL_DIR):
    print("Giải nén CIC-IDS2017.zip...")
    with zipfile.ZipFile(LOCAL_ZIP, 'r') as z:
        z.extractall(LOCAL_DIR)
    print("Giải nén xong!")
else:
    print("✅ Dataset đã giải nén sẵn.")

csv_files = glob.glob(os.path.join(LOCAL_DIR, "**", "*.csv"), recursive=True)
print(f"Tìm thấy {len(csv_files)} file CSV")


# ─────────────────────────────────────────────────────────────
# BƯỚC 3: ĐỌC DỮ LIỆU
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
# BƯỚC 4: TIỀN XỬ LÝ + LOG TRANSFORM
# ─────────────────────────────────────────────────────────────
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS)
X_all = df[FEATS].values.astype(np.float32)
y_all = (df['Label'] != 'BENIGN').astype(np.uint8).values
del df; gc.collect()
print("DataFrame đã xóa khỏi RAM.")

# Log1p transform cho features skewed
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

# Scaler — load từ Drive nếu đã có
if os.path.exists(SCALER_PATH):
    print("✅ Load scaler từ Drive...")
    with open(SCALER_PATH, 'rb') as f: sc = pickle.load(f)
    Xtr_raw = sc.transform(X_all[idx_tr])
else:
    print("Fit scaler mới...")
    sc = MinMaxScaler()
    Xtr_raw = sc.fit_transform(X_all[idx_tr])
    with open(SCALER_PATH, 'wb') as f: pickle.dump(sc, f)
    print(f"✅ Scaler lưu lên Drive: {SCALER_PATH}")

Xvl_raw = sc.transform(X_all[idx_vl])
Xts_raw = sc.transform(X_all[idx_ts])
del X_all, idx_tr, idx_vl, idx_ts; gc.collect()


# ─────────────────────────────────────────────────────────────
# BƯỚC 5: SLIDING WINDOWS (Zero-copy)
# ─────────────────────────────────────────────────────────────
def make_windows(data, s):
    n, f = data.shape
    if n < s: return np.empty((0, s, f), dtype=np.float32)
    shape   = (n - s + 1, s, f)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(data, shape=shape,
                                           strides=strides).copy().astype(np.float32)

print("Tạo sliding windows...")
Wtr = make_windows(Xtr_raw, SEQ); del Xtr_raw; gc.collect(); print(f"  Train: {Wtr.shape}")
Wvl = make_windows(Xvl_raw, SEQ); del Xvl_raw; gc.collect(); print(f"  Val  : {Wvl.shape}")
Wts = make_windows(Xts_raw, SEQ); del Xts_raw; gc.collect(); print(f"  Test : {Wts.shape}")
Ttr = torch.tensor(Wtr); del Wtr; gc.collect()
Tvl = torch.tensor(Wvl); del Wvl; gc.collect()
Tts = torch.tensor(Wts); del Wts; gc.collect()
print("✅ Windows xong!")


# ─────────────────────────────────────────────────────────────
# BƯỚC 6: MODEL
# ─────────────────────────────────────────────────────────────
class TransformerVAE(nn.Module):
    def __init__(self, F, S, L, H, N, D):
        super().__init__(); self.F=F; self.S=S
        enc=nn.TransformerEncoderLayer(F,H,F*8,D,batch_first=True)
        self.encoder=nn.TransformerEncoder(enc,N)
        self.fc_mu=nn.Linear(F*S,L); self.fc_lv=nn.Linear(F*S,L)
        self.dec_proj=nn.Linear(L,F*S)
        dec=nn.TransformerDecoderLayer(F,H,F*8,D,batch_first=True)
        self.decoder=nn.TransformerDecoder(dec,N); self.fc_out=nn.Linear(F,F)

    def encode(self,x):
        h=self.encoder(x).view(x.size(0),-1); return self.fc_mu(h),self.fc_lv(h)

    def reparameterize(self,mu,lv):
        return mu+torch.randn_like(mu)*torch.exp(.5*lv)

    def decode(self,z):
        p=self.dec_proj(z).view(-1,self.S,self.F); return self.fc_out(self.decoder(p,p))

    def forward(self,x):
        mu,lv=self.encode(x); return self.decode(self.reparameterize(mu,lv)),mu,lv

def vae_loss(r,x,mu,lv,b=1.):
    mse=nn.functional.mse_loss(r,x)
    kld=-0.5*torch.sum(1+lv-mu.pow(2)-lv.exp())/x.size(0)
    return mse+b*kld, mse, kld

model = TransformerVAE(FEATURE_SIZE,SEQ,LATENT,NHEAD,LAYERS,DROPOUT).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS, eta_min=1e-5)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")


# ─────────────────────────────────────────────────────────────
# BƯỚC 7: LOAD CHECKPOINT TỪ DRIVE NẾU CÓ
# ─────────────────────────────────────────────────────────────
start_epoch = 1; best = float('inf'); hist = {'tr':[],'vl':[],'mse':[],'kld':[]}

if os.path.exists(CKPT_PATH):
    print(f"\n{'='*50}")
    print(f">>> Tìm thấy checkpoint trên Drive!")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    sch.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best        = ckpt['best_val']
    hist        = ckpt.get('history', hist)
    print(f">>> Resume từ epoch {start_epoch}/{TOTAL_EPOCHS} | best={best:.6f}")
    print(f"{'='*50}\n")
else:
    print(f"\n>>> Bắt đầu train từ đầu (epoch 1/{TOTAL_EPOCHS})")


# ─────────────────────────────────────────────────────────────
# BƯỚC 8: TRAIN + TỰ ĐỘNG CHECKPOINT LÊN DRIVE
# ─────────────────────────────────────────────────────────────
if start_epoch <= TOTAL_EPOCHS:
    ldr = DataLoader(TensorDataset(Ttr), batch_size=BATCH, shuffle=True,
                     drop_last=True, pin_memory=(DEVICE=='cuda'), num_workers=2)
    print(f"Training | {len(ldr)} batches/epoch\n")

    for ep in range(start_epoch, TOTAL_EPOCHS+1):
        model.train(); tl=tm=tk=0.0
        for (b,) in ldr:
            b=b.to(DEVICE,non_blocking=True); opt.zero_grad()
            r,mu,lv=model(b); loss,mse,kld=vae_loss(r,b,mu,lv,BETA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); tl+=loss.item(); tm+=mse.item(); tk+=kld.item()
        nb=len(ldr)
        atr=tl/nb; hist['tr'].append(atr)
        hist['mse'].append(tm/nb); hist['kld'].append(tk/nb)

        model.eval()
        with torch.no_grad():
            vb=Tvl.to(DEVICE); rv,mv,lv2=model(vb)
            avl=vae_loss(rv,vb,mv,lv2,BETA)[0].item()
        hist['vl'].append(avl); sch.step()

        if avl<best:
            best=avl; torch.save(model.state_dict(), MODEL_BEST)

        # ── Lưu checkpoint lên DRIVE sau mỗi epoch ──────────
        torch.save({
            'epoch':ep, 'model':model.state_dict(),
            'optimizer':opt.state_dict(), 'scheduler':sch.state_dict(),
            'best_val':best, 'history':hist,
        }, CKPT_PATH)

        if ep%5==0 or ep==start_epoch or ep==TOTAL_EPOCHS:
            lr_now=opt.param_groups[0]['lr']
            print(f"[{ep:3d}/{TOTAL_EPOCHS}] train={atr:.5f} "
                  f"(MSE={hist['mse'][-1]:.5f} KLD={hist['kld'][-1]:.5f}) "
                  f"val={avl:.5f} best={best:.5f} lr={lr_now:.1e}")

    print(f"\n✅ Train xong! Tất cả đã lưu trên Drive: {DRIVE_DIR}")

    fig,ax=plt.subplots(1,2,figsize=(14,4))
    ax[0].plot(hist['tr'],label='Train'); ax[0].plot(hist['vl'],label='Val',ls='--')
    ax[0].set_title('Total Loss'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(hist['mse'],label='MSE'); ax[1].plot(hist['kld'],label='KLD',ls='--')
    ax[1].set_title('MSE vs KLD'); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.suptitle(f'β={BETA}, latent={LATENT}, log-transform'); plt.tight_layout(); plt.show()
else:
    print("Train đã xong! Chạy tiếp phần Evaluation.")


# ─────────────────────────────────────────────────────────────
# BƯỚC 9: ĐÁNH GIÁ (MC Dropout)
# ─────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_BEST, map_location=DEVICE))

def mc_eval(model, tensor, passes=15, bs=2048):
    model.train(); all_RE=[]; all_U=[]
    for i in range(0,len(tensor),bs):
        batch=tensor[i:i+bs].to(DEVICE); preds=[]
        with torch.no_grad():
            for _ in range(passes):
                r,_,_=model(batch); preds.append(r.cpu().numpy())
        P=np.array(preds); D=batch.cpu().numpy()
        all_RE.append(np.mean((P.mean(0)-D)**2,axis=(1,2)))
        all_U.append(np.mean(np.var(P,axis=0),axis=(1,2)))
        if i%(bs*20)==0: print(f"  {i}/{len(tensor)}...")
    return np.concatenate(all_RE),np.concatenate(all_U)

print("Tính threshold..."); RE_v,U_v=mc_eval(model,Tvl)
THR_RE=float(np.percentile(RE_v,90)); THR_U=float(np.percentile(U_v,95))
print(f"\n★ threshold_re = {THR_RE:.6f}")
print(f"★ threshold_u  = {THR_U:.10f}")

print("Đánh giá Attack..."); RE_t,U_t=mc_eval(model,Tts)
RE_all=np.concatenate([RE_v,RE_t])
y_true=np.array([0]*len(RE_v)+[1]*len(RE_t))
y_pred=(RE_all>THR_RE).astype(int)
print(f"\n{'='*60}")
print(classification_report(y_true,y_pred,target_names=['BENIGN','Attack'],digits=4))

fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.heatmap(confusion_matrix(y_true,y_pred),annot=True,fmt='d',cmap='Blues',ax=ax[0],
            xticklabels=['Normal','Attack'],yticklabels=['BENIGN','Attack'])
ax[0].set_title(f'Confusion Matrix (β={BETA}, log-transform)')
ax[1].hist(RE_v[::5],bins=120,alpha=0.6,label='BENIGN',density=True)
ax[1].hist(RE_t[::5],bins=120,alpha=0.6,label='Attack',color='tomato',density=True)
ax[1].axvline(THR_RE,color='red',ls='--',label=f'thr={THR_RE:.4f}')
ax[1].legend(); ax[1].set_title('RE: BENIGN vs Attack')
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────
# BƯỚC 10: TẢI MODEL VỀ MÁY (bổ sung — file đã lưu trên Drive rồi)
# ─────────────────────────────────────────────────────────────
from google.colab import files as colab_files
colab_files.download(MODEL_BEST)
colab_files.download(SCALER_PATH)

print(f"""
╔══════════════════════════════════════════════════════╗
║  TẤT CẢ ĐÃ LƯU TRÊN GOOGLE DRIVE:                  ║
║  📁 {DRIVE_DIR}
║    ├── checkpoint_v3.pth  (resume khi bị ngắt)      ║
║    ├── model_best.pth     (weights tốt nhất)         ║
║    ├── scaler.pkl         (MinMax scaler)            ║
║    └── CIC-IDS2017.zip    (không cần upload lại)    ║
║                                                      ║
║  Cập nhật config.yaml:                              ║
║    feature_size: {FEATURE_SIZE}                               ║
║    latent_dim:   {LATENT}                               ║
║    threshold_re: {THR_RE:.6f}                      ║
║    threshold_u:  {THR_U:.8f}                ║
╚══════════════════════════════════════════════════════╝
""")
