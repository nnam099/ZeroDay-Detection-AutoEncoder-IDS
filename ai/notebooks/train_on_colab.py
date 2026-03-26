# =============================================================
# DDoS Detection — Huấn luyện TransformerVAE trên Google Colab
# Chạy FREE với GPU T4. Không ảnh hưởng đến máy local!
# =============================================================
# Hướng dẫn:
# 1. Mở https://colab.research.google.com/
# 2. Upload file này (File > Upload notebook)
# 3. Vào Runtime > Change runtime type > GPU (T4)
# 4. Chạy từng cell theo thứ tự từ trên xuống
# =============================================================

# -------------------------------------------------------
# CELL 1: Cài đặt thư viện
# -------------------------------------------------------
# !pip install pandas scikit-learn matplotlib seaborn torch torchvision

# -------------------------------------------------------
# CELL 2: Kiểm tra GPU
# -------------------------------------------------------
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None — dùng CPU")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------
# CELL 3: TẠO DỮ LIỆU GIẢ LẬP (vì chưa có log thực)
# Mô phỏng Apache access log với 2 loại: Normal & Attack
# -------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_synthetic_logs(n_normal=2000, n_attack=300, seq_length=10):
    """
    Tạo dữ liệu log giả lập cho việc huấn luyện:
    - Normal traffic: status 200, bytes ~5000, request thưa
    - DDoS traffic: status 200/503, bytes nhỏ, request dày đặc
    
    Features: [status, bytes, method_cat, url_length, hour]
    """
    np.random.seed(42)
    
    # --- Traffic bình thường ---
    normal = pd.DataFrame({
        'status':     np.random.choice([200, 301, 404], n_normal, p=[0.85, 0.1, 0.05]),
        'bytes':      np.random.lognormal(mean=8.5, sigma=1.2, size=n_normal).astype(int),
        'method_cat': np.random.choice([0, 1], n_normal, p=[0.9, 0.1]),      # 0=GET, 1=POST
        'url_length': np.random.randint(5, 60, n_normal),
        'hour':       np.random.choice(range(8, 22), n_normal),               # giờ hành chính
        'label':     'Normal'
    })

    # --- Traffic DDoS (HTTP Flood) ---
    attack = pd.DataFrame({
        'status':     np.random.choice([200, 503], n_attack, p=[0.6, 0.4]),
        'bytes':      np.random.randint(50, 500, n_attack),                   # request nhỏ, nhiều
        'method_cat': np.zeros(n_attack, dtype=int),                          # toàn GET
        'url_length': np.random.randint(10, 20, n_attack),                   # URL ngắn, lặp lại
        'hour':       np.random.choice([2, 3, 4], n_attack),                 # tấn công ban đêm
        'label':     'Attack'
    })

    df = pd.concat([normal, attack], ignore_index=True).sample(frac=1, random_state=42)
    print(f"Dataset: {len(df)} rows — Normal: {n_normal}, Attack: {n_attack}")
    return df

df = generate_synthetic_logs()
df.head(10)


# -------------------------------------------------------
# CELL 4: TIỀN XỬ LÝ DỮ LIỆU & TẠO SLIDING WINDOWS
# -------------------------------------------------------
FEATURES     = ['status', 'bytes', 'method_cat', 'url_length', 'hour']
SEQ_LENGTH   = 10
FEATURE_SIZE = len(FEATURES)  # = 5

scaler = MinMaxScaler()

# Chỉ dùng dữ liệu NORMAL để train (bài toán Anomaly Detection không giám sát)
normal_df = df[df['label'] == 'Normal'].copy()
normal_data = scaler.fit_transform(normal_df[FEATURES].fillna(0).values)

def create_windows(data, seq_length):
    windows = []
    for i in range(len(data) - seq_length + 1):
        windows.append(data[i : i + seq_length])
    return np.array(windows)

X_normal = create_windows(normal_data, SEQ_LENGTH)
print(f"Shape dữ liệu train: {X_normal.shape}")  # [N, seq_length, feature_size]

# Chia train/val: 80/20
split = int(len(X_normal) * 0.8)
X_train = torch.tensor(X_normal[:split], dtype=torch.float32)
X_val   = torch.tensor(X_normal[split:], dtype=torch.float32)
print(f"Train: {X_train.shape} | Val: {X_val.shape}")


# -------------------------------------------------------
# CELL 5: ĐỊNH NGHĨA MÔ HÌNH TransformerVAE
# -------------------------------------------------------
import torch.nn as nn

class TransformerVAE(nn.Module):
    def __init__(self, feature_size, seq_length, latent_dim, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.feature_size = feature_size
        self.seq_length   = seq_length

        # ENCODER
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=nhead,
            batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_mu     = nn.Linear(feature_size * seq_length, latent_dim)
        self.fc_logvar = nn.Linear(feature_size * seq_length, latent_dim)

        # DECODER
        self.decoder_input = nn.Linear(latent_dim, feature_size * seq_length)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=feature_size, nhead=nhead,
            batch_first=True, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.fc_final = nn.Linear(feature_size, feature_size)

    def encode(self, x):
        h      = self.transformer_encoder(x)
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z_proj = self.decoder_input(z).view(-1, self.seq_length, self.feature_size)
        out    = self.transformer_decoder(z_proj, z_proj)
        return self.fc_final(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon_x    = self.decode(z)
        return recon_x, mu, logvar

# Khởi tạo
LATENT_DIM = 4
model = TransformerVAE(
    feature_size=FEATURE_SIZE,
    seq_length=SEQ_LENGTH,
    latent_dim=LATENT_DIM,
    dropout=0.1  # dropout cần thiết cho MC Dropout sau này
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Mô hình khởi tạo thành công: {total_params:,} tham số")


# -------------------------------------------------------
# CELL 6: HÀM TÍNH LOSS (MSE + KLD)   [train.py]
# -------------------------------------------------------
def calculate_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld_loss   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kld_loss, recon_loss, kld_loss


# -------------------------------------------------------
# CELL 7: VÒNG LẶP HUẤN LUYỆN CHÍNH
# -------------------------------------------------------
from torch.utils.data import DataLoader, TensorDataset

EPOCHS     = 60
BATCH_SIZE = 64
LR         = 0.001
BETA       = 1.0   # Trọng số KLD — tăng nếu muốn latent space gọn hơn

optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)

train_losses = []
val_losses   = []

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    epoch_loss = 0
    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss, recon_l, kld_l = calculate_loss(recon, batch, mu, logvar, BETA)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # tránh gradient explosion
        optimizer.step()
        epoch_loss += loss.item()

    avg_train = epoch_loss / len(train_loader)
    train_losses.append(avg_train)

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        val_batch = X_val.to(DEVICE)
        recon_val, mu_val, logvar_val = model(val_batch)
        val_loss, _, _ = calculate_loss(recon_val, val_batch, mu_val, logvar_val, BETA)
    val_losses.append(val_loss.item())

    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:3d}/{EPOCHS}] | Train Loss: {avg_train:.5f} | Val Loss: {val_loss.item():.5f}")

print("\n✅ Huấn luyện hoàn tất!")


# -------------------------------------------------------
# CELL 8: VẼ BIỂU ĐỒ LOSS
# -------------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss', color='steelblue')
plt.plot(val_losses,   label='Val Loss',   color='tomato', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Quá trình huấn luyện TransformerVAE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -------------------------------------------------------
# CELL 9: TÍNH NGƯỠNG ĐỘNG (threshold_tuning)
# Chạy trên tập val để tìm ngưỡng phù hợp
# -------------------------------------------------------
model.eval()
with torch.no_grad():
    val_recon, val_mu, _ = model(X_val.to(DEVICE))
    val_errors = nn.functional.mse_loss(
        val_recon, X_val.to(DEVICE), reduction='none'
    ).mean(dim=(1, 2)).cpu().numpy()

# MC Dropout để tính uncertainty trên val set
model.train()
mc_preds = []
with torch.no_grad():
    for _ in range(20):
        recon, _, _ = model(X_val.to(DEVICE))
        mc_preds.append(recon.cpu().numpy())

mc_preds      = np.array(mc_preds)   # [20, N, seq, features]
val_uncert    = np.mean(np.var(mc_preds, axis=0), axis=(1, 2))

# Ngưỡng 3-sigma cho RE, percentile 95 cho Uncertainty
threshold_re = np.mean(val_errors) + 3 * np.std(val_errors)
threshold_u  = np.percentile(val_uncert, 95)

print(f"📊 Ngưỡng phát hiện tính được:")
print(f"   threshold_re = {threshold_re:.6f}")
print(f"   threshold_u  = {threshold_u:.6f}")
print(f"\n→ Cập nhật hai giá trị này vào config/config.yaml!")


# -------------------------------------------------------
# CELL 10: KIỂM TRA TRÊN DỮ LIỆU MỚI (cả Normal & Attack)
# -------------------------------------------------------
all_data   = scaler.transform(df[FEATURES].fillna(0).values)
X_all      = create_windows(all_data, SEQ_LENGTH)
X_all_t    = torch.tensor(X_all, dtype=torch.float32)

# Tính RE
model.eval()
with torch.no_grad():
    recon_all, _, _ = model(X_all_t.to(DEVICE))
    re_all = nn.functional.mse_loss(
        recon_all, X_all_t.to(DEVICE), reduction='none'
    ).mean(dim=(1, 2)).cpu().numpy()

# Tính Uncertainty (MC Dropout)
model.train()
mc_all = []
with torch.no_grad():
    for _ in range(20):
        r, _, _ = model(X_all_t.to(DEVICE))
        mc_all.append(r.cpu().numpy())
u_all = np.mean(np.var(np.array(mc_all), axis=0), axis=(1, 2))

# Gán nhãn dự đoán
def auto_labeler(errors, uncertainties, threshold_re, threshold_u):
    labels = []
    for e, u in zip(errors, uncertainties):
        if e > threshold_re:
            labels.append("Unknown Attack" if u > threshold_u else "Known Anomaly")
        else:
            labels.append("Normal")
    return np.array(labels)

pred_labels = auto_labeler(re_all, u_all, threshold_re, threshold_u)

# Kết quả
from collections import Counter
print("Nhãn dự đoán:", Counter(pred_labels))


# -------------------------------------------------------
# CELL 11: BIỂU ĐỒ KẾT QUẢ
# -------------------------------------------------------
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Reconstruction Error
axes[0].plot(re_all, alpha=0.5, label='RE', color='steelblue')
axes[0].axhline(threshold_re, color='red',    linestyle='--', label=f'RE threshold ({threshold_re:.4f})')
axes[0].set_title('Reconstruction Error')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('MSE')
axes[0].legend()

# Uncertainty phân theo label
sns.boxplot(x=pred_labels, y=u_all, ax=axes[1], palette='Set2')
axes[1].axhline(threshold_u, color='red', linestyle='--', label=f'U threshold ({threshold_u:.4f})')
axes[1].set_title('Uncertainty theo nhãn dự đoán')
axes[1].set_xlabel('Nhãn')
axes[1].set_ylabel('Uncertainty Score')
axes[1].legend()

plt.tight_layout()
plt.show()


# -------------------------------------------------------
# CELL 12: LƯU MÔ HÌNH & TẢI VỀ MÁY CỦA BẠN
# -------------------------------------------------------
import os

os.makedirs("models", exist_ok=True)
SAVE_PATH = "models/transformer_vae_v1.pth"

torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ Đã lưu mô hình tại: {SAVE_PATH}")

# Tải file về máy (chỉ chạy được trong Google Colab)
from google.colab import files
files.download(SAVE_PATH)
print("📥 Đang tải file về máy...")
print("\n→ Sau khi tải về, đặt file vào thư mục:")
print("   DDoS-Mitigation/ai/models/transformer_vae_v1.pth")
