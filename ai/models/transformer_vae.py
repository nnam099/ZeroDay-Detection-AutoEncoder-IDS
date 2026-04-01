import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MemoryBank(nn.Module):
    """
    PROTOTYPICAL MEMORY BANK: Bộ nhớ lưu trữ "Dấu vân tay" của dữ liệu Benign sạch.
    Khi Hacker tấn công, data lạ sẽ bị ép phải map với các cấu trúc chuẩn mực này.
    Do không khớp, sai số Reconstruction Error (RE) của Attack sẽ bị khuyếch đại văng xa.
    """
    def __init__(self, num_prototypes, latent_dim, shrink_thres=0.0025):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.shrink_thres = shrink_thres

    def forward(self, z):
        # Tính độ tương đồng giữa latent z và các Prototypes
        att = F.linear(z, self.prototypes)
        att = F.softmax(att, dim=-1)
        
        if self.shrink_thres > 0:
            att = F.relu(att - self.shrink_thres)
            att = F.normalize(att, p=1, dim=-1)
            
        # Tổ hợp lại latent z từ các nguyên mẫu chuẩn
        z_mem = F.linear(att, self.prototypes.t())
        return z_mem, att


class MemConvTransformerAE(nn.Module):
    """
    Kiến trúc Deterministic Mem-AE siêu tốc O(1) cho Security Production.
    Cắt bỏ module VAE ngẫu nhiên nguyên gốc. Bổ sung Memory Bank.
    """

    def __init__(self, feature_size, seq_length, latent_dim,
                 nhead=2, num_layers=2, dropout=0.1, num_prototypes=100):
        super(MemConvTransformerAE, self).__init__()

        self.feature_size = feature_size
        self.seq_length = seq_length

        # 1. POSITIONAL ENCODING
        self.pos_encoder = PositionalEncoding(feature_size, max_len=seq_length)

        # 2. LOCAL BURST EXTRACTOR (CNN1D)
        self.local_cnn = nn.Conv1d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, padding=1)
        self.act_cnn = nn.GELU()

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
        
        # Mapping thẳng, không chạy Reparameterize (HẾT VAE)
        self.fc_z = nn.Linear(feature_size * seq_length, latent_dim)

        # ── Lõi Lọc Độc (Memory Bank) ──────────────────────────────────────────
        self.memory_bank = MemoryBank(num_prototypes, latent_dim)

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
        """ x shape: [batch, seq_length, feature_size] """
        # CNN trích đặc trưng cục bộ
        x_cnn = x.transpose(1, 2)
        x_cnn = self.act_cnn(self.local_cnn(x_cnn))
        x_cnn = x_cnn.transpose(1, 2)

        # Ghép Positional Time
        x_emb = self.pos_encoder(x + x_cnn)

        # Biến hình Global Sequence 
        h = self.transformer_encoder(x_emb)
        h_flat = h.contiguous().view(h.size(0), -1)
        h_flat = self.dropout_z(h_flat)
        return self.fc_z(h_flat)

    def decode(self, z):
        z_proj = self.decoder_input(z).view(-1, self.seq_length, self.feature_size)
        z_proj = self.pos_encoder(z_proj)
        out = self.transformer_decoder(z_proj, z_proj)
        return self.fc_final(out)

    def forward(self, x):
        z = self.encode(x)
        # Ép z thành bản clean (Benign Prototype)
        z_mem, att_weights = self.memory_bank(z)
        
        # Vẽ chuỗi mới
        recon_x = self.decode(z_mem)
        return recon_x, att_weights
