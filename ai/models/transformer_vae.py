import torch
import torch.nn as nn


class TransformerVAE(nn.Module):
    """
    Kiến trúc TransformerVAE cho phát hiện tấn công DDoS.

    Luồng hoạt động:
        Input (log sequences) --> Encoder --> (mu, logvar) --> Reparameterize --> z
        z --> Decoder --> Reconstructed sequences

    Phát hiện bất thường:
        - Reconstruction Error cao  => Traffic bất thường
        - Uncertainty cao (MC Dropout) => Kịch bản tấn công mới (Zero-day)

    QUAN TRỌNG: dropout > 0 là BẮT BUỘC để MC Dropout hoạt động đúng.
    Nếu dropout = 0, get_uncertainty() sẽ cho kết quả giống nhau mỗi lần => vô nghĩa.
    """

    def __init__(self, feature_size, seq_length, latent_dim,
                 nhead=2, num_layers=2, dropout=0.1):
        super(TransformerVAE, self).__init__()

        self.feature_size = feature_size
        self.seq_length = seq_length

        # ── ENCODER ──────────────────────────────────────────────────────────
        # dropout=dropout: BẮT BUỘC để MC Dropout hoạt động tại inference
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=feature_size * 4,   # FFN lớn hơn => học tốt hơn
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Latent space projections
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

        # Output projection về đúng kích thước feature
        self.fc_final = nn.Linear(feature_size, feature_size)

    # ── ENCODE ───────────────────────────────────────────────────────────────
    def encode(self, x):
        """
        x shape: [batch, seq_length, feature_size]
        Trả về: (mu, logvar) — các tham số phân phối tiềm ẩn
        """
        h = self.transformer_encoder(x)           # [batch, seq, feat]
        h_flat = h.view(h.size(0), -1)            # [batch, seq*feat]
        mu     = self.fc_mu(h_flat)               # [batch, latent_dim]
        logvar = self.fc_logvar(h_flat)           # [batch, latent_dim]
        return mu, logvar

    # ── REPARAMETERIZE ───────────────────────────────────────────────────────
    def reparameterize(self, mu, logvar):
        """
        Trick tái tham số hoá: z = mu + epsilon * sigma
        Epsilon ~ N(0, I) — nhiễu ngẫu nhiên Gaussian
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── DECODE ───────────────────────────────────────────────────────────────
    def decode(self, z):
        """
        z shape: [batch, latent_dim]
        Tái tạo lại chuỗi log ban đầu.
        """
        # Chiếu z về không gian chuỗi
        z_proj = self.decoder_input(z).view(-1, self.seq_length, self.feature_size)
        # Dùng chính z_proj làm cả tgt lẫn memory (self-reconstruction)
        out = self.transformer_decoder(z_proj, z_proj)
        return self.fc_final(out)

    # ── FORWARD ──────────────────────────────────────────────────────────────
    def forward(self, x):
        """Trả về (recon_x, mu, logvar)"""
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon_x    = self.decode(z)
        return recon_x, mu, logvar
