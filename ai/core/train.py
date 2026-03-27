import torch
import torch.nn as nn


def calculate_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """
    Hàm mất mát VAE = Reconstruction Loss + β × KL Divergence

    ─── Reconstruction Loss (MSE) ──────────────────────────────────────────
    Đo độ sai lệch giữa chuỗi gốc và chuỗi tái tạo.
    Traffic "bình thường" sau khi huấn luyện sẽ có MSE thấp.
    Traffic tấn công (chưa thấy khi train) sẽ có MSE cao => phát hiện anomaly.

    ─── KL Divergence ──────────────────────────────────────────────────────
    Điều chuẩn hoá không gian tiềm ẩn về phân phối N(0, I).
    Công thức: -0.5 * sum(1 + log(σ²) - μ² - σ²)

    FIX: Chuẩn hoá KLD theo batch size để tránh mất mát bị dominated bởi KLD
    khi batch lớn. Điều này giúp β có ý nghĩa ổn định.

    Args:
        recon_x : output của Decoder, shape [batch, seq, feat]
        x       : input gốc,          shape [batch, seq, feat]
        mu      : vector trung bình tiềm ẩn,  shape [batch, latent_dim]
        logvar  : log phương sai tiềm ẩn,     shape [batch, latent_dim]
        beta    : trọng số cân bằng Recon vs KLD (β-VAE)

    Returns:
        (total_loss, recon_loss, kld_loss)
    """
    batch_size = x.size(0)

    # MSE trung bình trên toàn batch và toàn bộ chiều
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')

    # KLD chuẩn hoá theo batch size (FIX: nguyên bản dùng torch.sum không chia)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss


def self_learning_process(model, optimizer, new_patterns,
                           save_path: str,
                           queue_limit: int = 100,
                           epochs: int = 20,
                           beta: float = 0.5):
    """
    Tự học (Self-learning / Fine-tuning) khi phát hiện đủ mẫu tấn công mới.

    Mục tiêu: Mô hình học cách "bình thường hoá" kịch bản tấn công mới
              để lần sau nhận ra nó là "Known Anomaly" thay vì "Unknown".

    Args:
        model        : TransformerVAE đang dùng
        optimizer    : Adam optimizer
        new_patterns : list hoặc Tensor chứa chuỗi log tấn công mới
        save_path    : đường dẫn lưu model (.pth)
        queue_limit  : số mẫu tối thiểu để kích hoạt tự học (tránh overfit)
        epochs       : số epoch fine-tuning (ít hơn train ban đầu)
        beta         : β cho KLD loss (thấp hơn để ưu tiên reconstruction)

    Returns:
        True nếu tự học thành công, False nếu chưa đủ mẫu.
    """
    if len(new_patterns) < queue_limit:
        print(f"[INFO] Buffer mới: {len(new_patterns)}/{queue_limit} mẫu. Chưa đủ để tự học.")
        return False

    print(f"[SELF-LEARNING] Đang học từ {len(new_patterns)} mẫu kịch bản mới...")
    model.train()

    # Chuyển dữ liệu về Tensor nếu cần
    if not torch.is_tensor(new_patterns):
        import numpy as np
        new_patterns = np.array(new_patterns)
        new_data_tensor = torch.tensor(new_patterns, dtype=torch.float32)
    else:
        new_data_tensor = new_patterns

    # Tự động chọn device theo model
    device = next(model.parameters()).device
    new_data_tensor = new_data_tensor.to(device)

    # Fine-tuning với learning rate thấp hơn (tránh catastrophic forgetting)
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, mu, logvar = model(new_data_tensor)
        loss, recon_l, kld_l = calculate_loss(recon, new_data_tensor, mu, logvar, beta)
        loss.backward()

        # Gradient clipping để ổn định training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch [{epoch+1:2d}/{epochs}] Loss={loss.item():.5f} "
                  f"(Recon={recon_l.item():.5f}, KLD={kld_l.item():.5f})")

    # Lưu model sau khi học xong
    torch.save(model.state_dict(), save_path)
    print(f"[SELF-LEARNING] Đã cập nhật model tại: {save_path}")
    return True