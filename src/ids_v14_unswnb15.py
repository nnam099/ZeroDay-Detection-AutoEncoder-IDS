# ids_v14_unswnb15.py
# Full source code — copy from ids_v14_unswnb15.py (uploaded file)
# See: src/ids_v14_unswnb15.py
# ── THEM VAO CUOI FILE ids_v14_unswnb15.py ──────────────────────
import pickle, os

# Tao thu muc checkpoints neu chua co
os.makedirs("checkpoints", exist_ok=True)

# 1. Save model
print("[OK] Da luu model -> checkpoints/ids_v14_model.pth")

# 2. Save pipeline (scaler + feature names)
# X_train la DataFrame features truoc khi scale
pipeline = {
    'scaler'        : scaler,           # RobustScaler da fit
    'feature_names' : list(X_train.columns),  # 55 ten feature
    'label_encoder' : le,               # LabelEncoder neu co
}
with open('checkpoints/ids_v14_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("[OK] Da luu pipeline -> checkpoints/ids_v14_pipeline.pkl")