import torch
import torch.nn as nn
import pickle
import numpy as np

# ── KIẾN TRÚC MÔ HÌNH TRANSFORMER VAE CỦA BẠN ────────────────────────────────
class TransformerVAE(nn.Module):
    """
    Kiến trúc TransformerVAE cho phát hiện tấn công DDoS.
    """
    def __init__(self, feature_size, seq_length, latent_dim,
                 nhead=2, num_layers=2, dropout=0.1):
        super(TransformerVAE, self).__init__()

        self.feature_size = feature_size
        self.seq_length = seq_length

        # ── ENCODER ──────────────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=feature_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_mu     = nn.Linear(feature_size * seq_length, latent_dim)
        self.fc_logvar = nn.Linear(feature_size * seq_length, latent_dim)

        # ── DECODER ──────────────────────────────────────────────────────────
        self.decoder_input = nn.Linear(latent_dim, feature_size * seq_length)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=feature_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_final = nn.Linear(feature_size, feature_size)

    def encode(self, x):
        h = self.transformer_encoder(x)
        h_flat = h.view(h.size(0), -1)
        mu     = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z_proj = self.decoder_input(z).view(-1, self.seq_length, self.feature_size)
        out = self.transformer_decoder(z_proj, z_proj)
        return self.fc_final(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon_x    = self.decode(z)
        return recon_x, mu, logvar

# ── LOGIC DỰ ĐOÁN (INFERENCE) DÀNH CHO VAE ──────────────────────────────────
class DDoSPredictor:
    def __init__(self, model_path, scaler_path):
        # 1. Load Scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # 2. Khai báo các thông số kiến trúc (🚨 BẠN CẦN SỬA CHO KHỚP LÚC TRAIN)
        self.feature_size = 3   # Số lượng cột tính năng (ví dụ: pkts, bytes, syn)
        self.seq_length = 1     # Độ dài chuỗi (Nếu train từng log đơn lẻ thì để 1)
        self.latent_dim = 16    # Số chiều không gian nén (Latent space)
        
        # Ngưỡng phát hiện bất thường (Căn cứ vào biểu đồ loss lúc bạn train)
        self.anomaly_threshold = 0.5 
        
        # Khởi tạo khung model
        self.model = TransformerVAE(
            feature_size=self.feature_size,
            seq_length=self.seq_length,
            latent_dim=self.latent_dim
        ) 
        
        # Nạp tủy não (.pth) vào khung và thiết lập chế độ eval
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval() 
        
        print("✅ Đã nạp TransformerVAE và Scaler lên bộ nhớ!")

    def predict(self, feature_list):
        try:
            # 1. Chuẩn hóa dữ liệu 2D
            features = np.array(feature_list).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # 2. Chuyển sang Tensor 3D để đưa vào Transformer [batch, seq_length, feature_size]
            input_tensor = torch.tensor(features_scaled, dtype=torch.float32).view(1, self.seq_length, self.feature_size)
            
            # 3. Chạy qua mạng Neural VAE để giải nén
            with torch.no_grad():
                recon_x, mu, logvar = self.model(input_tensor)
                
                # 4. Tính toán độ lỗi tái tạo (Reconstruction Error) bằng Mean Squared Error (MSE)
                mse_loss = torch.mean((input_tensor - recon_x) ** 2).item()
            
            # 5. Quyết định: Nếu độ lỗi lớn hơn ngưỡng cho phép => Đây là cuộc tấn công
            if mse_loss > self.anomaly_threshold:
                return 1 # Bị DDoS
            else:
                return 0 # Bình thường
                
        except Exception as e:
            print(f"❌ Lỗi tính toán AI: {e}")
            return 0