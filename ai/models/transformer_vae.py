# self-attention layer

# mean và các log-variance layer: các lớp tuyến tính để tính toán
# $\mu$ và $\sigma$ của không gian tìm ẩn (latent space)

# reparameterization function: lấy mẫu (sampling) từ phân phối xác suất

import torch
import torch.nn as nn

class TransformerVAE(nn.Module):
    def __init__(self, feature_size, seq_length, latent_dim, nhead=2, num_layers=2):
        super(TransformerVAE, self).__init__()
        # Khai báo nn.TransformerEncoder và nn.TransformerDecoder
        # Sử dụng nn.Linear để tạo tầng mu và logvar cho Latent Space
        self.feature_size = feature_size
        self.seq_length = seq_length

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_mu = nn.Linear(feature_size * seq_length, latent_dim)
        self.fc_logvar = nn.Linear(feature_size * seq_length, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, feature_size * seq_length)
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_final = nn.Linear(feature_size, feature_size)

    def encode(self, x):
        # Pass x qua TransformerEncoder -> Flatten -> mu & logvar
        # x shape: [batch, seq_length, feature_size]
        h = self.transformer_encoder(x)
        h_flat = h.view(h.size(0), -1)

        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # hàm lấy mẫu ngẫu nhiên (sampling) để tạo biến z
        # z = mu + epsilon * exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # nhiễu ngẫu nhiên gausian
        return mu + eps * std

    def decode(self, z):
        # Pass z qua TransformerDecoder để tái tạo lại chuỗi log ban đầu

        # 1. Chuyển z về lại dạng chuỗi [Batch, Seq_Len, Feature_Size]
        z_projected = self.decoder_input(z).view(-1, self.seq_length, self.feature_size)

        # 2. Sử dụng Transformer Decoder (trong VAE đơn giản có thể dùng chính z làm memory)
        out = self.transformer_decoder(z_projected, z_projected)
        return self.fc_final(out)

    def forward(self, x):
      # encode -> reparameterize -> decode
      mu, logvar = self.encode(x)
      z = self.reparameterize(mu, logvar)
      recon_x = self.decode(z)
      return recon_x, mu, logvar
