import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Tiêm thông tin vị trí thời gian của gói tin vào chuỗi.
    DDoS phụ thuộc cực lớn vào thứ tự thời gian của các packet liên tiếp.
    """
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
        # x shape: [batch, seq, feature]
        x = x + self.pe[:x.size(1), :]
        return x


class TransformerVAE(nn.Module):
    """
    Kiến trúc Conv-TransformerVAE (Áp dụng Tư duy Chuyên gia An ninh mạng).
    
    Cải tiến cốt lõi:
    1. POSITIONAL ENCODING: Transformer bắt buộc phải có để hiểu "Packet nào đến trước, packet nào đến sau". Không có PE, model sẽ coi mớ packet là 1 mớ hỗn độn (Bag-of-packets) thay vì 1 chuỗi luồng mạng (Flow stream).
    2. CONV1D LAYER: DDoS là sự "Bùng nổ" (Burstiness) gói tin trong 1 chớp mắt cục bộ. CNN1D trích xuất cực tốt đặc trưng Burst cục bộ này trước khi nhồi qua Transformer để bắt ngữ cảnh toàn cục.
    """

    def __init__(self, feature_size, seq_length, latent_dim,
                 nhead=2, num_layers=2, dropout=0.1):
        super(TransformerVAE, self).__init__()

        self.feature_size = feature_size
        self.seq_length = seq_length

        # 1. POSITIONAL ENCODING
        self.pos_encoder = PositionalEncoding(feature_size, max_len=seq_length)

        # 2. LOCAL BURST EXTRACTOR (CNN1D)
        # Bắt dính các xung lượng (Burst) chớp nhoáng của SYN Flood / UDP Flood
        self.local_cnn = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, padding=1)
        self.act_cnn = nn.GELU()  # GELU chống bão hoà gradient tốt hơn ReLU khi log chứa nhiều số 0

        # ── ENCODER ──────────────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=feature_size * 8,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout_z = nn.Dropout(dropout)
        self.fc_mu     = nn.Linear(feature_size * seq_length, latent_dim)
        self.fc_logvar = nn.Linear(feature_size * seq_length, latent_dim)

        # ── DECODER ──────────────────────────────────────────────────────────
        self.decoder_input = nn.Linear(latent_dim, feature_size * seq_length)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=feature_size * 8,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_final = nn.Linear(feature_size, feature_size)

    def encode(self, x):
        """
        x shape: [batch, seq_length, feature_size]
        """
        # A. Trích xuất đặc trưng Burst cục bộ bằng CNN
        # Phải transpose vì Conv1D của PyTorch nhận [batch, channels, length]
        x_cnn = x.transpose(1, 2)
        x_cnn = self.act_cnn(self.local_cnn(x_cnn))
        x_cnn = x_cnn.transpose(1, 2)

        # B. Cộng Residual và Nhúng Vị trí Thời gian
        x_emb = self.pos_encoder(x + x_cnn)

        # C. Phân tích chuỗi toàn cục (Global Context) bằng Transformer
        h = self.transformer_encoder(x_emb)
        h_flat = h.contiguous().view(h.size(0), -1)
        h_flat = self.dropout_z(h_flat) # Bổ sung dropout cho chống overfit nhiễu
        
        mu     = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z_proj = self.decoder_input(z).view(-1, self.seq_length, self.feature_size)
        # Thêm thứ tự thời gian vào cả decoder để tái tạo chuẩn hơn
        z_proj = self.pos_encoder(z_proj)
        out = self.transformer_decoder(z_proj, z_proj)
        return self.fc_final(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon_x    = self.decode(z)
        return recon_x, mu, logvar
