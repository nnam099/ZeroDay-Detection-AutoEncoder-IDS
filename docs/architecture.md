# Architecture — IDS v14.0

## IDSBackbone
Linear(n_feat → 256) → LayerNorm → GELU → ResBlock × 3

## Components
| Component   | Dim | Purpose                    |
|-------------|-----|----------------------------|
| classifier  | n_c | Supervised classification  |
| proj_head   | 64  | SupCon contrastive learning|
| autoencoder | n_f | Anomaly reconstruction     |

## Loss
Total = FocalLoss + 0.3 × SupConLoss + 0.5 × AE_MSE
