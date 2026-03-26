# mse loss

# knullback-leibler divergence (kld)

import torch
import torch.nn as nn

def calculate_loss(recon_x, x, mu, logvar, beta=1.0):
    # Công thức: Loss = Reconstruction_Loss + KLD_Loss

    # Tính MSE giữa recon_x và x (đo sai số tái tạo)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')

    # Tính KL Divergence (đo mức độ lệch giữa phân phối học được và chuẩn N(0,1))
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss

import os

def self_learning_process(model, optimizer, new_patterns, save_path, queue_limit=100):
    if len(new_patterns) < queue_limit: # Cần ít nhất 100 mẫu mới để tránh overfitting
        return False

    print(f"--- Đang học từ {len(new_patterns)} mẫu kịch bản mới... ---")
    model.train()

    # Chuyển dữ liệu mới thành Tensor Dataset
    # Kiểm tra nếu chưa là tensor thì mới khởi tạo
    if not torch.is_tensor(new_patterns):
        new_data_tensor = torch.tensor(new_patterns, dtype=torch.float32).cuda()
    else:
        new_data_tensor = new_patterns.cuda()

    # Fine-tuning nhanh trong 10-20 epochs
    for epoch in range(20):
        optimizer.zero_grad()
        recon, mu, logvar = model(new_data_tensor)
        loss, _, _ = calculate_loss(recon, new_data_tensor, mu, logvar)
        loss.backward()
        optimizer.step()

    # Lưu lại phiên bản mô hình mới
    torch.save(model.state_dict(), save_path)
    print(f"--- Đã cập nhật mô hình tại {save_path} ---")
    return True