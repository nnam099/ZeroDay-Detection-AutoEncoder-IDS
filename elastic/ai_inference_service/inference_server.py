"""
╔══════════════════════════════════════════════════════════════════════════╗
║         AI INFERENCE SERVER — DDoS Hybrid Detector                      ║
║  Nhận flow features từ Logstash → Trả kết quả Attack/BENIGN             ║
║  Framework: FastAPI + PyTorch (CPU mode, ~50-80ms/request)              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os, math, pickle, time, logging
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — chỉnh lại cho phù hợp với nơi bạn lưu model
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.getenv("MODEL_PATH",     "./model_hybrid_best.pth")
SCALER_PATH    = os.getenv("SCALER_PATH",    "./scaler_hybrid.pkl")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "./best_threshold.txt")

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS (phải khớp với lúc train)
# ─────────────────────────────────────────────────────────────────────────────
SEQ          = 16
HIDDEN       = 64
NHEAD        = 4
T_LAYERS     = 2
DROPOUT      = 0.2
DEVICE       = "cpu"   # Inference server dùng CPU, đủ nhanh

FEATURE_NAMES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s',
    'Fwd IAT Mean', 'Packet Length Mean', 'SYN Flag Count',
    'ACK Flag Count', 'Init_Win_bytes_forward', 'Active Mean',
    'Idle Mean', 'Bwd Packet Length Std'
]
LOG_FEATS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Flow Bytes/s', 'Flow Packets/s',
    'Fwd IAT Mean', 'Packet Length Mean', 'Init_Win_bytes_forward',
    'Active Mean', 'Idle Mean', 'Bwd Packet Length Std'
]
FEATURE_SIZE = len(FEATURE_NAMES)
LOG_IDX = [FEATURE_NAMES.index(f) for f in LOG_FEATS if f in FEATURE_NAMES]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION (copy từ ddos_hybrid_detector.py)
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, :d//2]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class HybridDDoSDetector(nn.Module):
    def __init__(self, F, S, hidden=64, heads=4, t_layers=2, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(F, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )
        self.bilstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout
        )
        self.lstm_norm = nn.LayerNorm(hidden)
        self.pos_enc = PositionalEncoding(hidden, max_len=S + 10)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc_layer, t_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.bilstm(h)
        h = self.lstm_norm(h)
        h = self.pos_enc(h)
        h = self.transformer(h)
        h_mean = h.mean(dim=1)
        h_max  = h.max(dim=1).values
        return self.head(torch.cat([h_mean, h_max], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# MOCK MODE — Tự động bật khi không có model files
# Dùng để test pipeline Logstash → AI → Elasticsearch mà không cần train model
# ─────────────────────────────────────────────────────────────────────────────
MOCK_MODE = False


def load_artifacts():
    global MOCK_MODE
    logger.info("📦 Loading model artifacts...")

    # Kiểm tra xem các file model có tồn tại không
    missing = [p for p in [MODEL_PATH, SCALER_PATH] if not os.path.exists(p)]
    if missing:
        logger.warning("⚠️  MOCK MODE ACTIVATED — Các file model sau không tìm thấy:")
        for p in missing:
            logger.warning(f"   Missing: {p}")
        logger.warning("   Server sẽ trả kết quả giả lập (random) để test pipeline.")
        logger.warning("   Để dùng model thật: copy model files vào thư mục /app/models/")
        MOCK_MODE = True
        return None, None, 0.5

    # Load threshold
    threshold = 0.5
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            threshold = float(f.read().split("=")[1].strip())
    logger.info(f"  Threshold: {threshold:.4f}")

    # Load scaler
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("  Scaler: loaded")

    # Load model
    model = HybridDDoSDetector(
        F=FEATURE_SIZE, S=SEQ,
        hidden=HIDDEN, heads=NHEAD,
        t_layers=T_LAYERS, dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    logger.info(f"  Model: loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    MOCK_MODE = False
    return model, scaler, threshold


model_global, scaler_global, threshold_global = load_artifacts()

# Buffer để gom đủ SEQ=16 flows trước khi inference
# Key: source_ip (hoặc flow_id), Value: list of feature vectors
flow_buffer: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DDoS AI Inference API",
    description="CNN-BiLSTM-Transformer Hybrid — Real-time DDoS Detection",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class FlowFeatures(BaseModel):
    """Một flow network đơn lẻ từ Logstash"""
    flow_id:                str   = Field(..., description="ID định danh flow/session")
    source_ip:              str   = Field(..., description="IP nguồn")
    destination_port:       float = Field(0)
    flow_duration:          float = Field(0)
    total_fwd_packets:      float = Field(0)
    total_backward_packets: float = Field(0)
    flow_bytes_s:           float = Field(0)
    flow_packets_s:         float = Field(0)
    fwd_iat_mean:           float = Field(0)
    packet_length_mean:     float = Field(0)
    syn_flag_count:         float = Field(0)
    ack_flag_count:         float = Field(0)
    init_win_bytes_forward: float = Field(0)
    active_mean:            float = Field(0)
    idle_mean:              float = Field(0)
    bwd_packet_length_std:  float = Field(0)
    timestamp:              Optional[str] = None


class BatchFlowRequest(BaseModel):
    """Batch inference — gửi đúng 16 flows (1 sequence window)"""
    flows: List[FlowFeatures]
    sequence_id: Optional[str] = "default"


class PredictionResponse(BaseModel):
    sequence_id:   str
    source_ip:     str
    label:         str           # "ATTACK" hoặc "BENIGN"
    probability:   float         # P(attack) ∈ [0, 1]
    confidence:    str           # "HIGH" / "MEDIUM" / "LOW"
    tier:          str           # "BLOCK" / "CAPTCHA" / "ALLOW"
    latency_ms:    float
    threshold_used: float


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Extract features từ FlowFeatures object
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(flow: FlowFeatures) -> np.ndarray:
    return np.array([
        flow.destination_port,
        flow.flow_duration,
        flow.total_fwd_packets,
        flow.total_backward_packets,
        flow.flow_bytes_s,
        flow.flow_packets_s,
        flow.fwd_iat_mean,
        flow.packet_length_mean,
        flow.syn_flag_count,
        flow.ack_flag_count,
        flow.init_win_bytes_forward,
        flow.active_mean,
        flow.idle_mean,
        flow.bwd_packet_length_std,
    ], dtype=np.float32)


def get_tier(prob: float, threshold: float) -> tuple[str, str, str]:
    """
    Tiered Response:
    - prob < threshold * 0.7  → ALLOW
    - prob ∈ [threshold*0.7, threshold*1.2] → CAPTCHA (nghi ngờ)
    - prob > threshold * 1.2  → BLOCK
    """
    low  = threshold * 0.7
    high = threshold * 1.2

    if prob < low:
        return "BENIGN", "HIGH", "ALLOW"
    elif prob < high:
        label = "BENIGN" if prob < threshold else "ATTACK"
        return label, "LOW", "CAPTCHA"
    else:
        return "ATTACK", "HIGH", "BLOCK"


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "CNN-BiLSTM-Transformer Hybrid v3" if not MOCK_MODE else "MOCK_MODE (no model files)",
        "mock_mode": MOCK_MODE,
        "threshold": threshold_global,
        "device": DEVICE,
        "seq_len": SEQ
    }


@app.post("/predict/batch", response_model=PredictionResponse)
def predict_batch(req: BatchFlowRequest):
    """
    Nhận 16 flows (1 sliding window) → Trả kết quả phân loại.
    Logstash nên gom đủ 16 flows cùng source_ip rồi mới gọi endpoint này.
    """
    t0 = time.time()

    if len(req.flows) != SEQ:
        raise HTTPException(
            status_code=400,
            detail=f"Cần đúng {SEQ} flows, nhận được {len(req.flows)}"
        )

    try:
        # ── MOCK MODE: Trả kết quả ngẫu nhiên để test pipeline ───────────────
        if MOCK_MODE:
            import random
            prob = round(random.uniform(0.0, 1.0), 6)
            label, confidence, tier = get_tier(prob, threshold_global)
            latency = round((time.time() - t0) * 1000, 2)
            logger.info(f"[MOCK][{req.sequence_id}] src={req.flows[0].source_ip} → {label} (p={prob:.3f})")
            return PredictionResponse(
                sequence_id=req.sequence_id,
                source_ip=req.flows[0].source_ip,
                label=label,
                probability=prob,
                confidence=confidence,
                tier=tier,
                latency_ms=latency,
                threshold_used=threshold_global,
            )

        # ── REAL MODE ─────────────────────────────────────────────────────────
        # Extract & preprocess
        X = np.array([extract_features(f) for f in req.flows], dtype=np.float32)

        # Replace inf/nan
        X = np.where(np.isfinite(X), X, 0.0)

        # Log1p transform
        X[:, LOG_IDX] = np.log1p(np.abs(X[:, LOG_IDX]))

        # Scale
        X_scaled = scaler_global.transform(X)  # (16, 14)

        # Inference
        inp = torch.tensor(X_scaled[np.newaxis, :, :]).to(DEVICE)  # (1, 16, 14)
        with torch.no_grad():
            logit = model_global(inp)
            prob = torch.sigmoid(logit).item()

        label, confidence, tier = get_tier(prob, threshold_global)
        latency = (time.time() - t0) * 1000

        logger.info(
            f"[{req.sequence_id}] src={req.flows[0].source_ip} "
            f"→ {label} (p={prob:.3f}, tier={tier}, {latency:.1f}ms)"
        )

        return PredictionResponse(
            sequence_id=req.sequence_id,
            source_ip=req.flows[0].source_ip,
            label=label,
            probability=round(prob, 6),
            confidence=confidence,
            tier=tier,
            latency_ms=round(latency, 2),
            threshold_used=threshold_global,
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/stream")
def predict_stream(flow: FlowFeatures):
    """
    Streaming mode: nạp từng flow một.
    Server tự gom đủ SEQ flows theo source_ip rồi inference.
    Trả về kết quả khi đủ window, hoặc {"status": "buffering"} khi chưa đủ.
    """
    t0 = time.time()
    key = flow.source_ip

    feat = extract_features(flow)

    if key not in flow_buffer:
        flow_buffer[key] = []
    flow_buffer[key].append(feat)

    # Sliding window: nếu chưa đủ SEQ, giữ lại
    if len(flow_buffer[key]) < SEQ:
        return {
            "status": "buffering",
            "source_ip": key,
            "buffered": len(flow_buffer[key]),
            "needed": SEQ
        }

    # Lấy cửa sổ cuối cùng, xóa flow cũ nhất (sliding)
    window = np.array(flow_buffer[key][-SEQ:], dtype=np.float32)
    flow_buffer[key] = flow_buffer[key][-(SEQ - 1):]  # giữ SEQ-1 để tạo window tiếp

    # ── MOCK MODE ────────────────────────────────────────────────────────────
    if MOCK_MODE:
        import random
        prob = round(random.uniform(0.0, 1.0), 6)
        label, confidence, tier = get_tier(prob, threshold_global)
        latency = round((time.time() - t0) * 1000, 2)
        logger.info(f"[MOCK][stream] src={key} → {label} (p={prob:.3f})")
        return {
            "status": "predicted",
            "mock_mode": True,
            "source_ip": key,
            "label": label,
            "probability": prob,
            "confidence": confidence,
            "tier": tier,
            "action": {
                "BLOCK":   "drop_connection",
                "CAPTCHA": "rate_limit_and_challenge",
                "ALLOW":   "pass_through",
            }.get(tier, "pass_through"),
            "latency_ms": latency,
        }

    # ── REAL MODE ─────────────────────────────────────────────────────────────
    # Preprocess
    window = np.where(np.isfinite(window), window, 0.0)
    window[:, LOG_IDX] = np.log1p(np.abs(window[:, LOG_IDX]))
    X_scaled = scaler_global.transform(window)

    inp = torch.tensor(X_scaled[np.newaxis]).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model_global(inp)).item()

    label, confidence, tier = get_tier(prob, threshold_global)
    latency = (time.time() - t0) * 1000

    return {
        "status": "predicted",
        "mock_mode": False,
        "source_ip": key,
        "label": label,
        "probability": round(prob, 6),
        "confidence": confidence,
        "tier": tier,
        "action": {
            "BLOCK":   "drop_connection",
            "CAPTCHA": "rate_limit_and_challenge",
            "ALLOW":   "pass_through",
        }.get(tier, "pass_through"),
        "latency_ms": round(latency, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
