# -*- coding: utf-8 -*-
"""
SOC AI Platform v15 - Dashboard
Compatible: Windows, Python 3.9+
Run: streamlit run app.py
"""

import streamlit as st
import sys
import os
import importlib.util
import numpy as np
import pandas as pd
import json
import uuid
import pickle
from datetime import datetime

# ── Page config (phai dat truoc bat ky st.* nao) ─────────────────
st.set_page_config(
    page_title="SOC AI Platform v15",
    page_icon="[SOC]",
    layout="wide"
)

st.markdown("""
<style>
    :root {
        --soc-bg: #090d12;
        --soc-panel: #111821;
        --soc-panel-2: #16202b;
        --soc-line: #263241;
        --soc-text: #d6dde6;
        --soc-muted: #8b98a7;
        --soc-blue: #3aa0ff;
        --soc-green: #37d67a;
        --soc-yellow: #f6c343;
        --soc-red: #ff4d5e;
    }
    .stApp { background: var(--soc-bg); color: var(--soc-text); }
    section[data-testid="stSidebar"] { background: #0b1118; border-right: 1px solid var(--soc-line); }
    div[data-testid="stMetric"] {
        background: var(--soc-panel);
        border: 1px solid var(--soc-line);
        border-radius: 6px;
        padding: 12px 14px;
    }
    div[data-testid="stMetricLabel"] p { color: var(--soc-muted); font-size: 0.78rem; text-transform: uppercase; }
    .soc-header {
        border: 1px solid var(--soc-line);
        background: linear-gradient(90deg, #101a24 0%, #0d141c 100%);
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .soc-title { font-size: 1.35rem; font-weight: 700; margin: 0; }
    .soc-subtitle { color: var(--soc-muted); margin-top: 4px; font-size: 0.88rem; }
    .soc-panel {
        border: 1px solid var(--soc-line);
        background: var(--soc-panel);
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .soc-badge {
        display: inline-block;
        border: 1px solid var(--soc-line);
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 0.75rem;
        color: var(--soc-text);
        background: var(--soc-panel-2);
        margin-right: 6px;
    }
    .soc-detail-title {
        color: var(--soc-muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }
    .soc-detail-value {
        color: var(--soc-text);
        font-size: 1.02rem;
        font-weight: 650;
        margin-bottom: 10px;
    }
    .soc-pill-red { border-color: rgba(255,77,94,0.5); color: #ff8d99; }
    .soc-pill-blue { border-color: rgba(58,160,255,0.5); color: #8bc9ff; }
    .soc-pill-green { border-color: rgba(55,214,122,0.5); color: #8cf0b5; }
    .sidebar-brand {
        border: 1px solid var(--soc-line);
        background: #0f1720;
        border-radius: 8px;
        padding: 14px 12px;
        margin: 6px 0 14px 0;
    }
    .sidebar-brand-title { font-size: 1.05rem; font-weight: 750; color: #f4f8fc; }
    .sidebar-brand-sub { color: var(--soc-muted); font-size: 0.74rem; margin-top: 4px; }
    .health-grid {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 8px 10px;
        align-items: center;
        border: 1px solid var(--soc-line);
        border-radius: 8px;
        padding: 12px;
        background: #0f1720;
    }
    .health-label { color: var(--soc-muted); font-size: 0.75rem; text-transform: uppercase; }
    .health-ok, .health-warn, .health-bad {
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 0.70rem;
        font-weight: 700;
    }
    .health-ok { color: #8cf0b5; background: rgba(55,214,122,0.10); border: 1px solid rgba(55,214,122,0.35); }
    .health-warn { color: #ffd77a; background: rgba(246,195,67,0.10); border: 1px solid rgba(246,195,67,0.35); }
    .health-bad { color: #ff9ca6; background: rgba(255,77,94,0.10); border: 1px solid rgba(255,77,94,0.35); }
    .sev-critical { color: var(--soc-red); font-weight: 700; }
    .sev-high { color: #ff9b55; font-weight: 700; }
    .sev-medium { color: var(--soc-yellow); font-weight: 700; }
    .sev-low { color: var(--soc-green); font-weight: 700; }
    .block-container { padding-top: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Them src/ vao path de import module ──────────────────────────
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC_DIR)

# ── Kiem tra thu vien can thiet ───────────────────────────────────
def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

MISSING_REQUIRED = []
try:
    import torch
except ImportError:
    MISSING_REQUIRED.append("torch")

HAS_SHAP = _has_module("shap")
if not HAS_SHAP:
    st.warning("[!] Thieu shap — SHAP se bi tat")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("[!] Thieu python-dotenv — se bo qua load .env")

LLM_PROVIDER = None
try:
    from llm_agent import LLM_PROVIDER as _LLM_PROVIDER
    LLM_PROVIDER = (_LLM_PROVIDER or "").lower()
except Exception:
    LLM_PROVIDER = None

PROVIDER_DEPS = {
    "groq": "groq",
    "gemini": "google.generativeai",
    "openai": "openai",
    "anthropic": "anthropic",
}
LLM_DEP = PROVIDER_DEPS.get(LLM_PROVIDER or "")
HAS_LLM_DEPS = True
if LLM_DEP and not _has_module(LLM_DEP):
    HAS_LLM_DEPS = False
    st.warning(f"[!] Thieu thu vien cho LLM provider '{LLM_PROVIDER}': {LLM_DEP}")

if MISSING_REQUIRED:
    st.error(f"Thieu thu vien bat buoc: {', '.join(MISSING_REQUIRED)}")
    st.code(f"pip install {' '.join(MISSING_REQUIRED)}")
    st.stop()

PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
LLM_KEY_ENV = PROVIDER_KEYS.get(LLM_PROVIDER or "")
LLM_KEY_OK = bool(os.getenv(LLM_KEY_ENV, "").strip()) if LLM_KEY_ENV else False

# ── Import modules v15 ────────────────────────────────────────────
HAS_EXPLAINER = False
if HAS_SHAP:
    try:
        from explainer import SHAPExplainer
        HAS_EXPLAINER = True
    except ImportError:
        st.warning("[!] Khong tim thay explainer.py trong src/ - SHAP se bi tat")

try:
    from mitre_mapper import MITREMapper
    HAS_MITRE = True
except ImportError:
    HAS_MITRE = False
    st.warning("[!] Khong tim thay mitre_mapper.py trong src/")

HAS_LLM = False
if LLM_DEP and HAS_LLM_DEPS:
    try:
        from llm_agent import SOCTriageAgent
        HAS_LLM = True
    except ImportError:
        st.warning("[!] Khong tim thay llm_agent.py trong src/")

# ── Runtime config ────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CKPT_DIR = os.getenv("IDS_CHECKPOINT_DIR", os.path.join(BASE_DIR, 'checkpoints'))
DATA_DIR = os.getenv("IDS_DATA_DIR", os.path.join(BASE_DIR, 'data'))

# v14 is the operational default because this repo currently ships v14 artifacts.
MODEL_VERSION = os.getenv("IDS_MODEL_VERSION", "v14").strip().lower()
if MODEL_VERSION not in {"v14", "v15"}:
    st.warning(f"IDS_MODEL_VERSION khong hop le: {MODEL_VERSION}. Fallback ve v14.")
    MODEL_VERSION = "v14"

DEFAULT_MODEL = f"ids_{MODEL_VERSION}_model.pth"
DEFAULT_PIPE = f"ids_{MODEL_VERSION}_pipeline.pkl"
MODEL_PATH = os.getenv("IDS_MODEL_PATH", os.path.join(CKPT_DIR, DEFAULT_MODEL))
PIPE_PATH = os.getenv("IDS_PIPELINE_PATH", os.path.join(CKPT_DIR, DEFAULT_PIPE))
DATA_PATH = os.getenv("IDS_SAMPLE_DATA_PATH", os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv'))

# CLASS_NAMES se duoc load tu label_encoder sau khi model load
# De tranh sai thu tu do LabelEncoder sort theo alphabet
CLASS_NAMES  = ["Normal", "DoS", "Exploits", "Reconnaissance", "Generic"]  # fallback
AE_THRESHOLD = 0.5
PIPELINE_THRESHOLDS = None
PIPELINE_META = {}

def _build_categorical_maps_from_sample() -> dict:
    if not os.path.exists(DATA_PATH):
        return {}
    try:
        cols = pd.read_csv(DATA_PATH, nrows=1).columns
        usecols = [c for c in ['proto', 'service', 'state'] if c in cols]
        if not usecols:
            return {}
        df = pd.read_csv(DATA_PATH, usecols=usecols)
        maps = {}
        for col in usecols:
            values = df[col].astype(str).fillna('unk')
            classes = sorted(values.unique().tolist())
            if 'unk' not in classes:
                classes.append('unk')
            maps[col] = {v: i for i, v in enumerate(classes)}
        return maps
    except Exception:
        return {}

# ── Sidebar navigation ────────────────────────────────────────────
NAV_ITEMS = ["[1] Dashboard", "[2] Analyze Alert", "[3] Zero-Day Logs", "[4] Ask AI", "[5] Setup Guide"]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = NAV_ITEMS[0]

st.sidebar.markdown(
    f"""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">SOC AI Platform</div>
        <div class="sidebar-brand-sub">IDS {MODEL_VERSION.upper()} | SHAP | MITRE ATT&CK | LLM</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Navigation", NAV_ITEMS, key="nav_page")

def _health(value, warn=False):
    if value:
        return '<span class="health-ok">OK</span>'
    if warn:
        return '<span class="health-warn">WARN</span>'
    return '<span class="health-bad">OFF</span>'

llm_ok = HAS_LLM and LLM_KEY_OK
st.sidebar.markdown("**System Health**")
st.sidebar.markdown(
    f"""
    <div class="health-grid">
        <div class="health-label">SHAP</div><div>{_health(HAS_EXPLAINER)}</div>
        <div class="health-label">MITRE</div><div>{_health(HAS_MITRE)}</div>
        <div class="health-label">LLM</div><div>{_health(llm_ok, warn=HAS_LLM)}</div>
        <div class="health-label">Model</div><div>{_health(os.path.exists(MODEL_PATH))}</div>
        <div class="health-label">Pipeline</div><div>{_health(os.path.exists(PIPE_PATH))}</div>
        <div class="health-label">Data</div><div>{_health(os.path.exists(DATA_PATH))}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption(f"Model version: {MODEL_VERSION.upper()}")

# ── Pipeline guard/autogen ─────────────────────────────────────────
def ensure_pipeline():
    if os.path.exists(PIPE_PATH):
        try:
            with open(PIPE_PATH, 'rb') as f:
                pipeline = pickle.load(f)
            if not pipeline.get('categorical_maps'):
                maps = _build_categorical_maps_from_sample()
                if maps:
                    pipeline['categorical_maps'] = maps
                    pipeline['categorical_maps_source'] = 'sample_data_fallback'
            return pipeline
        except Exception as e:
            st.warning(f"[!] Loi doc pipeline: {e}")
            return None

    if not os.path.exists(DATA_DIR):
        st.warning("[!] Khong tim thay data/ de tao pipeline")
        return None

    try:
        train_mod = __import__(f"ids_{MODEL_VERSION}_unswnb15", fromlist=["load_unsw_csvs"])
        load_unsw_csvs = train_mod.load_unsw_csvs
        clean_df = train_mod.clean_df
        prepare_splits = train_mod.prepare_splits
        with st.spinner("Dang tao pipeline tu data/ (chi 1 lan)..."):
            df = load_unsw_csvs(DATA_DIR)
            df = clean_df(df)
            splits = prepare_splits(df, seed=42)
        pipeline = {
            'scaler':        splits['scaler'],
            'label_encoder': splits['label_encoder'],
            'feat_cols':     splits['feat_cols'],
            'feature_names': splits.get('feature_names', splits['feat_cols']),
            'known_cats':    splits['known_cats'],
            'zd_cats':       splits['zd_cats'],
            'categorical_maps': splits.get('categorical_maps', {}),
            'version':       MODEL_VERSION,
            'n_features':    splits['n_features'],
            'n_classes':     splits['n_classes'],
        }
        with open(PIPE_PATH, 'wb') as f:
            pickle.dump(pipeline, f)
        st.success("[OK] Pipeline da duoc tao va luu")
        return pipeline
    except Exception as e:
        st.error(f"Loi tao pipeline: {e}")
        return None

# ── Load model (cache) ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    global PIPELINE_THRESHOLDS, PIPELINE_META
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None
    try:
        train_mod = __import__(f"ids_{MODEL_VERSION}_unswnb15", fromlist=["IDSModel"])
        IDSModel = train_mod.IDSModel

        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        n_feat     = checkpoint['n_features']
        n_cls      = checkpoint['n_classes']
        hidden     = checkpoint.get('hidden', 256)
        ae_hidden  = checkpoint.get('ae_hidden', 128)

        if MODEL_VERSION == "v15":
            latent_dim = checkpoint.get('latent_dim', 32)
            model = IDSModel(
                n_features=n_feat, n_classes=n_cls,
                hidden=hidden, ae_hidden=ae_hidden, latent_dim=latent_dim,
            )
        else:
            model = IDSModel(n_features=n_feat, n_classes=n_cls, hidden=hidden, ae_hidden=ae_hidden)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        pipeline = ensure_pipeline()
        if pipeline is None:
            st.warning("[!] Thieu pipeline — chuyen sang DEMO MODE")
            return None, None, None, None

        PIPELINE_THRESHOLDS = pipeline.get('thresholds')
        PIPELINE_META = pipeline

        scaler        = pipeline['scaler']
        feature_names = pipeline.get('feature_names', pipeline.get('feat_cols', []))
        label_encoder = pipeline['label_encoder']

        return model, scaler, feature_names, label_encoder

    except Exception as e:
        st.error(f"Loi load model: {e}")
        return None, None, None, None

model, scaler, feature_names, label_encoder = load_model()

if PIPELINE_THRESHOLDS:
    AE_THRESHOLD = float(PIPELINE_THRESHOLDS.get('ae_re', AE_THRESHOLD))

# Lay CLASS_NAMES dung thu tu tu label_encoder (tranh sai do sort alphabet)
if label_encoder is not None:
    CLASS_NAMES = list(label_encoder.classes_)

def _encode_categorical_column(series: pd.Series, mapping=None) -> np.ndarray:
    values = series.astype(str).fillna('unk')
    if mapping:
        return values.map(lambda x: mapping.get(x, mapping.get('unk', -1))).astype(np.float32).values
    codes, _ = pd.factorize(values, sort=True)
    return codes.astype(np.float32)

def preprocess_raw_df(df_raw: pd.DataFrame, feat_cols: list) -> np.ndarray:
    """
    Ap dung dung cac buoc tien xu ly nhu trong prepare_splits().
    Output LUON CO DUNG SO LUONG FEATURES = len(feat_cols), thu tu khop voi scaler.
    """
    train_mod = __import__(f"ids_{MODEL_VERSION}_unswnb15", fromlist=["engineer_features"])
    engineer_features = train_mod.engineer_features

    df = df_raw.copy()
    categorical_maps = PIPELINE_META.get('categorical_maps', {}) if PIPELINE_META else {}

    # 1. Encode cac cot categorical -> _num columns
    for cat in ['proto', 'service', 'state']:
        if cat in df.columns:
            df[f'{cat}_num'] = _encode_categorical_column(df[cat], categorical_maps.get(cat))

    # 2. Chay feature engineering (tao bytes_ratio, log_sbytes, ...)
    existing_numeric = [c for c in df.columns if c not in
        {'attack_cat', 'label', 'label_binary', 'srcip', 'dstip',
         'sport', 'dsport', 'stime', 'ltime', 'id', 'proto', 'service', 'state'}
    ]
    try:
        df, _ = engineer_features(df, existing_numeric)
    except Exception:
        pass

    # 3. Chon DUNG THU TU feat_cols, padding = 0 neu thieu cot
    #    Dam bao output co dung (n_rows, len(feat_cols)) khop voi scaler
    rows = len(df)
    out  = np.zeros((rows, len(feat_cols)), dtype=np.float32)
    for i, col in enumerate(feat_cols):
        if col in df.columns:
            try:
                out[:, i] = pd.to_numeric(df[col], errors='coerce').fillna(0).values
            except Exception:
                out[:, i] = 0.0
        # else: cot khong co -> de 0

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


@st.cache_resource
def load_background(_scaler, _feature_names):
    if not os.path.exists(DATA_PATH):
        return None
    try:
        df        = pd.read_csv(DATA_PATH)
        label_col = 'label' if 'label' in df.columns else df.columns[-1]
        normal    = df[df[label_col] == 0].sample(
            min(200, len(df[df[label_col] == 0])), random_state=42
        )
        bg_arr = preprocess_raw_df(normal, _feature_names or [])
        if bg_arr.shape[1] == 0:
            st.warning("[SHAP] Khong tim thay feature khop — SHAP se bi tat")
            return None
        return bg_arr
    except Exception as e:
        st.error(f"Loi load background data: {e}")
        return None



@st.cache_resource
def get_components(_model, _scaler, _feature_names, _bg):
    comps = {}
    if HAS_EXPLAINER and _model and _scaler and _bg is not None:
        try:
            comps['explainer'] = SHAPExplainer(_model, _scaler, _feature_names, _bg)
        except Exception as e:
            st.warning(f"SHAP init loi: {type(e).__name__}: {e}")
    if HAS_MITRE:
        comps['mapper'] = MITREMapper()
    if HAS_LLM:
        comps['agent'] = SOCTriageAgent()
    return comps

# ── Demo mode neu chua co model ───────────────────────────────────
DEMO_MODE = (model is None or scaler is None)

def mock_inference(n_features=49):
    """Tao du lieu gia lap de demo khi chua co model."""
    ae_score     = float(np.random.uniform(0.3, 0.95))
    max_prob     = float(np.random.uniform(0.4, 0.99))
    pred_idx     = int(np.random.randint(0, len(CLASS_NAMES)))
    hybrid_score = 0.5 * ae_score + 0.5 * (1 - max_prob)
    is_zeroday   = ae_score > AE_THRESHOLD and max_prob < 0.6

    # SHAP gia lap
    if feature_names:
        fnames = feature_names[:10]
    else:
        fnames = [f"feature_{i}" for i in range(10)]

    shap_summary_lines = []
    for fname in fnames:
        sv  = float(np.random.uniform(-0.3, 0.3))
        fv  = float(np.random.uniform(0, 100))
        dir_str = "tang nguy co" if sv > 0 else "giam nguy co"
        shap_summary_lines.append(f"  - {fname}: gia_tri={fv:.2f}, SHAP={sv:+.4f} ({dir_str})")

    shap_summary = "\n".join(shap_summary_lines)

    return {
        "alert_id"       : str(uuid.uuid4())[:8].upper(),
        "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hybrid_score"   : hybrid_score,
        "ae_score"       : ae_score,
        "max_prob"       : max_prob,
        "predicted_class": "Zero-Day" if is_zeroday else CLASS_NAMES[pred_idx],
        "is_zeroday"     : is_zeroday,
        "shap_summary"   : shap_summary,
        "mitre_summary"  : "MITRE mapping: [T1595] Active Scanning (Reconnaissance)",
        "top_features"   : [(fnames[i], np.random.uniform(-0.3,0.3), np.random.uniform(0,100)) for i in range(min(5,len(fnames)))],
        "probs"          : np.random.dirichlet(np.ones(len(CLASS_NAMES))).tolist(),
        "demo_mode"      : True,
    }

def run_full_pipeline(raw_features: np.ndarray, comps: dict):
    """Chay toan bo pipeline that: IDS -> SHAP -> MITRE -> LLM."""
    # 1. Inference — IDSModel.forward() tra ve (logits, fv)
    scaled = scaler.transform(raw_features)
    tensor = torch.FloatTensor(scaled)
    with torch.no_grad():
        outputs = model(tensor)
        logits  = outputs[0] if isinstance(outputs, tuple) else outputs
        probs   = torch.softmax(logits, dim=1).numpy()[0]

    max_prob   = float(probs.max())
    pred_idx   = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Unknown"

    # AE score tu AutoEncoder
    with torch.no_grad():
        if hasattr(model, 'ae'):
            ae_score = float(model.ae.recon_error(tensor).mean().item())
        elif hasattr(model, 'vae'):
            ae_score = float(model.vae.recon_error(tensor).mean().item())
        elif hasattr(model, 'autoencoder'):
            recon    = model.autoencoder(tensor)
            ae_score = float(torch.mean((tensor - recon) ** 2).item())
        else:
            ae_score = float(1.0 - max_prob + np.random.uniform(0, 0.1))

    hybrid_score = 0.5 * ae_score + 0.5 * (1 - max_prob)
    is_zeroday   = ae_score > AE_THRESHOLD and max_prob < 0.6

    result = {
        "alert_id"       : str(uuid.uuid4())[:8].upper(),
        "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hybrid_score"   : hybrid_score,
        "ae_score"       : ae_score,
        "max_prob"       : max_prob,
        "predicted_class": "Zero-Day" if is_zeroday else pred_class,
        "is_zeroday"     : is_zeroday,
        "probs"          : probs.tolist(),
        "demo_mode"      : False,
    }

    # 2. SHAP
    if 'explainer' in comps:
        try:
            shap_res = comps['explainer'].explain_alert(raw_features)
            result['shap_summary']  = shap_res['summary_text']
            result['top_features']  = shap_res['top_features']
            result['shap_values']   = shap_res['shap_values']
        except Exception as e:
            result['shap_summary'] = f"SHAP loi: {e}"
            result['top_features'] = []
    else:
        result['shap_summary'] = "SHAP chua duoc kich hoat"
        result['top_features'] = []

    # 3. MITRE
    if 'mapper' in comps:
        try:
            mapper = comps['mapper']
            if is_zeroday:
                mitre_res = mapper.map_zeroday(ae_score, result.get('top_features', []))
            else:
                mitre_res = mapper.map_known_attack(pred_idx, class_names=CLASS_NAMES, top_features=result.get('top_features', []))
            result['mitre_result']  = mitre_res
            result['mitre_summary'] = mapper.format_for_llm(mitre_res)
        except Exception as e:
            result['mitre_summary'] = f"MITRE loi: {e}"
            result['mitre_result']  = None
    else:
        result['mitre_summary'] = "MITRE chua duoc kich hoat"
        result['mitre_result']  = None

    return result


def run_batch_inference(raw_features: np.ndarray, batch_size: int = 512) -> pd.DataFrame:
    """Chay inference cho toan bo file CSV (khong SHAP/LLM)."""
    if model is None or scaler is None:
        return pd.DataFrame()
    if raw_features is None or len(raw_features) == 0:
        return pd.DataFrame()

    scaled = scaler.transform(raw_features)
    tensor = torch.FloatTensor(scaled)
    probs_all = []
    ae_all = []

    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            x = tensor[i:i+batch_size]
            outputs = model(x)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(np.atleast_2d(probs))

            if hasattr(model, 'ae'):
                ae_score = model.ae.recon_error(x)
                if isinstance(ae_score, torch.Tensor):
                    ae_score = ae_score.cpu().numpy()
            elif hasattr(model, 'vae'):
                ae_score = model.vae.recon_error(x)
                if isinstance(ae_score, torch.Tensor):
                    ae_score = ae_score.cpu().numpy()
            elif hasattr(model, 'autoencoder'):
                recon = model.autoencoder(x)
                ae_score = torch.mean((x - recon) ** 2, dim=-1).cpu().numpy()
            else:
                ae_score = (1.0 - probs.max(axis=1))

            ae_score = np.atleast_1d(ae_score)
            if ae_score.ndim == 0 or ae_score.shape[0] != len(x):
                ae_score = np.full(len(x), float(np.mean(ae_score)), dtype=np.float32)
            ae_all.append(ae_score)

    if not probs_all or not ae_all:
        return pd.DataFrame()

    probs_all = np.concatenate(probs_all, axis=0)
    ae_all = np.concatenate(ae_all, axis=0)
    max_prob = probs_all.max(axis=1)
    pred_idx = probs_all.argmax(axis=1)
    pred_class = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else "Unknown" for i in pred_idx]
    hybrid_score = 0.5 * ae_all + 0.5 * (1 - max_prob)
    is_zeroday = (ae_all > AE_THRESHOLD) & (max_prob < 0.6)

    return pd.DataFrame({
        "predicted_class": pred_class,
        "max_prob": max_prob,
        "ae_score": ae_all,
        "hybrid_score": hybrid_score,
        "is_zeroday": is_zeroday.astype(bool),
    })

def severity_rank(severity: str) -> int:
    return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(str(severity).upper(), 0)

def severity_class(severity: str) -> str:
    return {
        "CRITICAL": "sev-critical",
        "HIGH": "sev-high",
        "MEDIUM": "sev-medium",
        "LOW": "sev-low",
    }.get(str(severity).upper(), "sev-medium")

def risk_score(result: dict, severity: str | None = None) -> int:
    sev_weight = {"CRITICAL": 35, "HIGH": 25, "MEDIUM": 15, "LOW": 5}.get(str(severity or "").upper(), 10)
    hybrid = float(result.get("hybrid_score", 0)) * 45
    ae = min(float(result.get("ae_score", 0)), 1.0) * 15
    zd = 15 if result.get("is_zeroday") else 0
    return int(min(100, round(sev_weight + hybrid + ae + zd)))

def render_soc_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="soc-header">
            <div class="soc-title">{title}</div>
            <div class="soc-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_alert_context_from_log(row_scores: dict) -> dict:
    source_row = int(row_scores.get("source_row", 0))
    family = str(row_scores.get("zero_day_family") or "")
    classifier_class = str(row_scores.get("predicted_class", "Unknown"))
    detection = str(row_scores.get("detection") or ("Zero-Day / " + (family or "Unknown") if row_scores.get("is_zeroday") else classifier_class))
    return {
        "alert_id": f"ZD-{source_row:06d}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_row": source_row,
        "hybrid_score": float(row_scores.get("hybrid_score", 0)),
        "ae_score": float(row_scores.get("ae_score", 0)),
        "max_prob": float(row_scores.get("max_prob", 0)),
        "predicted_class": detection,
        "classifier_class": classifier_class,
        "zero_day_family": family or "Unknown",
        "is_zeroday": bool(row_scores.get("is_zeroday", False)),
        "shap_summary": "Batch log context - SHAP explanation not computed for this row.",
        "mitre_summary": "",
        "top_features": [],
        "probs": [],
        "demo_mode": False,
        "raw_scores": {k: str(v) for k, v in row_scores.items()},
    }

def get_llm_analysis(result: dict, comps: dict):
    """Goi LLM triage agent."""
    if 'agent' not in comps:
        return {
            "severity"           : "HIGH" if result['hybrid_score'] > 0.6 else "MEDIUM",
            "verdict"            : f"{'Zero-Day' if result['is_zeroday'] else result['predicted_class']} detected - hybrid score: {result['hybrid_score']:.3f}",
            "attack_summary"     : f"AE reconstruction error cao ({result['ae_score']:.3f}) cho thay traffic bat thuong. Can kiem tra thu cong.",
            "recommended_actions": ["Kiem tra nguon IP", "Xem xet block traffic", "Escalate len Tier 2"],
            "false_positive_risk": "MEDIUM",
            "false_positive_reason": "Chua co LLM de phan tich sau hon",
            "analyst_note"       : "Kich hoat LLM agent de co phan tich chi tiet hon",
        }
    try:
        return comps['agent'].triage_alert(result)
    except Exception as e:
        return {
            "severity"           : "HIGH",
            "verdict"            : "LLM analysis loi - can review thu cong",
            "attack_summary"     : str(e),
            "recommended_actions": ["Review thu cong"],
            "false_positive_risk": "UNKNOWN",
            "false_positive_reason": "N/A",
            "analyst_note"       : f"Loi: {e}",
        }

# ── Display result ────────────────────────────────────────────────
def display_result(result: dict, llm: dict):
    sev_label = {"CRITICAL": "[CRITICAL]", "HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]"}
    sev_color = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"}
    sev = llm.get('severity', 'HIGH')

    if result.get('demo_mode'):
        st.info("DEMO MODE - Du lieu gia lap. Cai model that de co ket qua chinh xac.")

    st.subheader(f"Alert #{result['alert_id']} | {sev_label.get(sev, sev)} | {result['timestamp']}")
    st.info(f"**Verdict:** {llm.get('verdict', '')}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hybrid Score", f"{result['hybrid_score']:.3f}", delta="High" if result['hybrid_score'] > 0.6 else "Normal")
    col2.metric("AE Score", f"{result['ae_score']:.3f}")
    col3.metric("Classifier", result['predicted_class'])
    col4.metric("Zero-Day", "YES" if result['is_zeroday'] else "NO")

    # Probability bar
    st.markdown("**Class probabilities:**")
    prob_df = pd.DataFrame({
        "Class": CLASS_NAMES[:len(result['probs'])],
        "Probability": [round(p, 4) for p in result['probs']]
    })
    st.bar_chart(prob_df.set_index("Class"))

    tab1, tab2, tab3, tab4 = st.tabs(["AI Analysis", "SHAP Features", "MITRE ATT&CK", "Actions"])

    with tab1:
        st.markdown(f"**Severity:** `{sev}`")
        st.markdown("**Attack Summary:**")
        st.write(llm.get('attack_summary', 'N/A'))
        st.markdown(f"**False Positive Risk:** `{llm.get('false_positive_risk', 'N/A')}`")
        if llm.get('false_positive_reason'):
            st.write(llm['false_positive_reason'])
        st.markdown(f"**Analyst Note:** {llm.get('analyst_note', '')}")

    with tab2:
        feats = result.get('top_features', [])
        if feats:
            df_shap = pd.DataFrame(feats, columns=["Feature", "SHAP Value", "Actual Value"])
            df_shap["SHAP Value"] = df_shap["SHAP Value"].round(4)
            df_shap["Actual Value"] = df_shap["Actual Value"].round(4)
            df_shap["Direction"] = df_shap["SHAP Value"].apply(lambda x: "Tang nguy co" if x > 0 else "Giam nguy co")
            st.dataframe(df_shap, use_container_width=True)
            st.bar_chart(df_shap.set_index("Feature")["SHAP Value"])
        else:
            st.write(result.get('shap_summary', 'SHAP chua chay'))

    with tab3:
        mitre = result.get('mitre_result')
        if mitre:
            techniques = mitre.get('techniques') or mitre.get('suspected_techniques', [])
            for t in techniques:
                url = f"https://attack.mitre.org/techniques/{t['id']}/"
                st.markdown(f"**[{t['id']}]** {t['name']} — *{t['tactic']}* → [Xem chi tiet]({url})")
        else:
            st.write(result.get('mitre_summary', 'MITRE chua chay'))

    with tab4:
        actions = llm.get('recommended_actions', [])
        for i, action in enumerate(actions, 1):
            st.checkbox(f"{i}. {action}", key=f"act_{result['alert_id']}_{i}")
        if st.button("Export JSON Report", key=f"export_{result['alert_id']}"):
            report = {**result, "llm_analysis": llm}
            report.pop('shap_values', None)  # khong serialize numpy array
            report.pop('probs', None)
            st.download_button(
                "Download JSON",
                data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
                file_name=f"alert_{result['alert_id']}.json",
                mime="application/json"
            )

    st.session_state['last_alert']  = result
    st.session_state['last_llm']    = llm

# ═════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═════════════════════════════════════════════════════════════════
def display_result(result: dict, llm: dict):
    """SOC-style alert detail view. Overrides the earlier basic renderer."""
    sev = str(llm.get('severity', 'HIGH')).upper()
    risk = risk_score(result, sev)

    if result.get('demo_mode'):
        st.info("DEMO MODE - Du lieu gia lap. Cai model that de co ket qua chinh xac.")

    st.markdown(
        f"""
        <div class="soc-panel">
            <span class="soc-badge">ALERT {result['alert_id']}</span>
            <span class="soc-badge">{result['timestamp']}</span>
            <span class="soc-badge {severity_class(sev)}">{sev}</span>
            <span class="soc-badge">Risk {risk}/100</span>
            <div style="margin-top:10px;font-size:1.05rem;font-weight:600;">{llm.get('verdict', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Risk Score", f"{risk}/100")
    col2.metric("Hybrid", f"{result['hybrid_score']:.3f}")
    col3.metric("AE / VAE", f"{result['ae_score']:.3f}")
    col4.metric("Classifier", result['predicted_class'])
    col5.metric("Zero-Day", "YES" if result['is_zeroday'] else "NO")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Triage", "Model Evidence", "MITRE ATT&CK", "Response", "Raw Report"])

    with tab1:
        st.markdown("**Attack Summary**")
        st.write(llm.get('attack_summary', 'N/A'))
        a, b, c = st.columns(3)
        a.metric("Severity", sev)
        b.metric("False Positive Risk", llm.get('false_positive_risk', 'N/A'))
        c.metric("MITRE Confidence", (result.get("mitre_result") or {}).get("confidence", "N/A"))
        st.markdown("**Analyst Note**")
        st.write(llm.get('analyst_note', 'N/A'))
        if llm.get('false_positive_reason'):
            st.caption(f"False-positive rationale: {llm['false_positive_reason']}")

    with tab2:
        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES[:len(result['probs'])],
            "Probability": [round(p, 4) for p in result['probs']]
        })
        st.markdown("**Class Probabilities**")
        st.dataframe(prob_df.sort_values("Probability", ascending=False), use_container_width=True, hide_index=True)
        st.bar_chart(prob_df.set_index("Class"))

        feats = result.get('top_features', [])
        if feats:
            df_shap = pd.DataFrame(feats, columns=["Feature", "SHAP Value", "Actual Value"])
            df_shap["SHAP Value"] = df_shap["SHAP Value"].round(4)
            df_shap["Actual Value"] = df_shap["Actual Value"].round(4)
            df_shap["Direction"] = df_shap["SHAP Value"].apply(lambda x: "Tang nguy co" if x > 0 else "Giam nguy co")
            st.dataframe(df_shap, use_container_width=True, hide_index=True)
            st.bar_chart(df_shap.set_index("Feature")["SHAP Value"])
        else:
            st.write(result.get('shap_summary', 'SHAP chua chay'))

    with tab3:
        mitre = result.get('mitre_result')
        if mitre:
            techniques = mitre.get('techniques') or mitre.get('suspected_techniques', [])
            st.markdown(
                f"**Mode:** `{mitre.get('mapping_mode', 'unknown')}` | "
                f"**Primary tactic:** `{mitre.get('primary_tactic', 'N/A')}` | "
                f"**Confidence:** `{mitre.get('confidence', 'LOW')}`"
            )
            if techniques:
                mitre_df = pd.DataFrame([{
                    "Technique": f"{t['id']} - {t['name']}",
                    "Tactic": t.get('tactic', ''),
                    "Confidence": t.get('confidence', ''),
                    "Rationale": t.get('rationale', ''),
                    "Evidence": ", ".join(t.get('evidence', [])),
                    "ATT&CK": t.get('url', f"https://attack.mitre.org/techniques/{t['id']}/"),
                } for t in techniques])
                st.dataframe(mitre_df, use_container_width=True, hide_index=True)
                checks = []
                for t in techniques:
                    for action in t.get("response_actions", []):
                        checks.append({"Technique": t["id"], "Check": action})
                if checks:
                    st.markdown("**Recommended MITRE-driven checks**")
                    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
            st.caption(mitre.get("coverage_note", ""))
        else:
            st.write(result.get('mitre_summary', 'MITRE chua chay'))

    with tab4:
        mitre = result.get('mitre_result') or {}
        actions = list(llm.get('recommended_actions', []))
        for t in mitre.get("techniques", []) or mitre.get("suspected_techniques", []):
            actions.extend(t.get("response_actions", []))
        actions = list(dict.fromkeys(actions))
        st.markdown("**Analyst Response Checklist**")
        for i, action in enumerate(actions, 1):
            st.checkbox(f"{i}. {action}", key=f"act_{result['alert_id']}_{i}")
        if st.button("Export JSON Report", key=f"export_{result['alert_id']}"):
            report = {**result, "llm_analysis": llm}
            report.pop('shap_values', None)
            report.pop('probs', None)
            st.download_button(
                "Download JSON",
                data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
                file_name=f"alert_{result['alert_id']}.json",
                mime="application/json"
            )

    with tab5:
        report = {**result, "llm_analysis": llm}
        report.pop('shap_values', None)
        st.json(report)

    st.session_state['last_alert'] = result
    st.session_state['last_llm'] = llm


if page == "[1] Dashboard":
    render_soc_header(
        "SOC Operations Console",
        f"IDS {MODEL_VERSION.upper()} real-time triage workspace | Model, anomaly scoring, MITRE mapping and analyst queue",
    )

    history = st.session_state.get('alert_history', [])
    zd = sum(1 for a in history if a.get('is_zeroday'))
    hi = sum(1 for a in history if a.get('llm_severity') in ['CRITICAL','HIGH'])
    avg_risk = 0
    if history:
        avg_risk = int(np.mean([risk_score(a, a.get('llm_severity')) for a in history]))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Alert Queue", len(history))
    c2.metric("Critical / High", hi)
    c3.metric("Zero-Day Hypotheses", zd)
    c4.metric("Average Risk", f"{avg_risk}/100")
    c5.metric("Model Mode", "DEMO" if DEMO_MODE else MODEL_VERSION.upper())

    st.markdown(
        f"""
        <div class="soc-panel">
            <span class="soc-badge">Model: {os.path.basename(MODEL_PATH)}</span>
            <span class="soc-badge">Pipeline: {os.path.basename(PIPE_PATH)}</span>
            <span class="soc-badge">SHAP: {'ON' if HAS_EXPLAINER else 'OFF'}</span>
            <span class="soc-badge">MITRE: {'ON' if HAS_MITRE else 'OFF'}</span>
            <span class="soc-badge">LLM: {'ON' if HAS_LLM and LLM_KEY_OK else 'OFF'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history:
        st.markdown("### Active Alert Queue")
        hist_df = pd.DataFrame([{
            "Alert ID"    : a['alert_id'],
            "Time"        : a['timestamp'],
            "Severity"    : a.get('llm_severity', 'N/A'),
            "Risk"        : risk_score(a, a.get('llm_severity')),
            "Class"       : a['predicted_class'],
            "Hybrid Score": round(a['hybrid_score'], 3),
            "AE Score"    : round(a.get('ae_score', 0), 3),
            "Zero-Day"    : "YES" if a['is_zeroday'] else "NO",
        } for a in history])
        hist_df = hist_df.sort_values(["Risk", "Time"], ascending=[False, False])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("### Severity Distribution")
            sev_counts = hist_df["Severity"].value_counts().rename_axis("Severity").reset_index(name="Count")
            st.bar_chart(sev_counts.set_index("Severity"))
        with right:
            st.markdown("### Zero-Day Mix")
            zd_counts = hist_df["Zero-Day"].value_counts().rename_axis("Zero-Day").reset_index(name="Count")
            st.bar_chart(zd_counts.set_index("Zero-Day"))
    else:
        st.markdown(
            """
            <div class="soc-panel">
                No alerts in the current analyst session. Use Analyze Alert to pull a sample, upload a CSV, or run a manual triage scenario.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if DEMO_MODE:
        st.warning("Model chua duoc load. Xem tab '[4] Setup Guide' de biet cach cai dat.")

# ═════════════════════════════════════════════════════════════════
# PAGE: Analyze Alert
# ═════════════════════════════════════════════════════════════════
elif page == "[2] Analyze Alert":
    render_soc_header(
        "Alert Investigation",
        "Run single-alert triage, inspect model evidence, map hypotheses to MITRE ATT&CK, and generate response actions.",
    )

    if DEMO_MODE:
        st.warning("DEMO MODE: Model chua duoc load. Dang dung du lieu gia lap.")

    bg_data = None
    comps   = {}

    if not DEMO_MODE:
        bg_data = load_background(scaler, feature_names)
        comps   = get_components(model, scaler, feature_names, bg_data)

    mode = st.radio(
        "Input source",
        ["Random sample tu dataset", "Nhap thu cong (demo)", "Upload CSV"],
        horizontal=True,
    )

    if mode == "Random sample tu dataset":
        if not os.path.exists(DATA_PATH) and not DEMO_MODE:
            st.error(f"Khong tim thay data tai: {DATA_PATH}")
            st.info("Tai UNSW-NB15 tu Kaggle va dat vao thu muc data/")
        else:
            prefer_attack = st.checkbox("Uu tien lay mau attack (de demo ro hon)", value=True)
            if st.button("Lay sample ngau nhien + Phan tich", type="primary"):
                if DEMO_MODE:
                    with st.spinner("Dang phan tich (DEMO)..."):
                        result = mock_inference()
                        llm    = get_llm_analysis(result, comps)
                else:
                    with st.spinner("Dang phan tich... (SHAP mat ~30 giay)"):
                        df = pd.read_csv(DATA_PATH)
                        label_col = 'label' if 'label' in df.columns else df.columns[-1]
                        pool   = df[df[label_col] == 1] if prefer_attack else df
                        sample = pool.sample(1, random_state=np.random.randint(0, 9999))
                        # Tien xu ly dung nhu luc train (encode + engineer features)
                        raw    = preprocess_raw_df(sample, feature_names or [])
                        result = run_full_pipeline(raw, comps)
                        llm    = get_llm_analysis(result, comps)

                display_result(result, llm)

                # Luu vao history
                if 'alert_history' not in st.session_state:
                    st.session_state['alert_history'] = []
                result['llm_severity'] = llm.get('severity', 'N/A')
                st.session_state['alert_history'].append(result)

    elif mode == "Nhap thu cong (demo)":
        st.info("Nhap gia tri hybrid score va AE score de xem LLM phan tich.")
        col1, col2 = st.columns(2)
        ae_val  = col1.slider("AE Reconstruction Error", 0.0, 1.0, 0.75, 0.01)
        max_p   = col2.slider("Classifier Max Probability", 0.0, 1.0, 0.45, 0.01)
        atk_cls = st.selectbox("Predicted Class", CLASS_NAMES)

        if st.button("Phan tich", type="primary"):
            hybrid = 0.5 * ae_val + 0.5 * (1 - max_p)
            is_zd  = ae_val > AE_THRESHOLD and max_p < 0.6
            result = {
                "alert_id"       : str(uuid.uuid4())[:8].upper(),
                "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "hybrid_score"   : hybrid,
                "ae_score"       : ae_val,
                "max_prob"       : max_p,
                "predicted_class": "Zero-Day" if is_zd else atk_cls,
                "is_zeroday"     : is_zd,
                "shap_summary"   : "Manual input - khong co SHAP data",
                "mitre_summary"  : "",
                "top_features"   : [],
                "probs"          : [1/len(CLASS_NAMES)] * len(CLASS_NAMES),
                "demo_mode"      : True,
            }
            if HAS_MITRE:
                mapper = MITREMapper()
                if is_zd:
                    mr = mapper.map_zeroday(ae_val, [])
                else:
                    mr = mapper.map_known_attack(
                        CLASS_NAMES.index(atk_cls) if atk_cls in CLASS_NAMES else 1,
                        class_names=CLASS_NAMES,
                    )
                result['mitre_result']  = mr
                result['mitre_summary'] = mapper.format_for_llm(mr)

            with st.spinner("LLM dang phan tich..."):
                llm = get_llm_analysis(result, comps)
            display_result(result, llm)

    else:  # Upload CSV
        uploaded = st.file_uploader(
            "Upload CSV (du lieu mang UNSW-NB15 raw, co cac cot feature goc)", type="csv"
        )
        if uploaded:
            raw_df = pd.read_csv(uploaded)
            st.write("Preview:", raw_df.head())

            if st.button("Phan tich TOAN BO file"):
                if DEMO_MODE:
                    st.warning("DEMO MODE khong ho tro phan tich toan bo file.")
                else:
                    raw = preprocess_raw_df(raw_df, feature_names or [])
                    if raw.shape[1] == 0 or (raw == 0).all():
                        st.error(
                            "Khong tim thay features hop le trong file CSV nay. "
                            "Vui long upload file CSV co cac cot feature goc cua UNSW-NB15."
                        )
                    else:
                        with st.spinner("Dang phan tich toan bo file..."):
                            result_df = run_batch_inference(raw)
                        if result_df.empty:
                            st.error("Khong the chay inference cho file nay.")
                        else:
                            result_df = result_df.copy()
                            result_df.insert(0, "source_row", np.arange(len(result_df)))
                            family_col = next(
                                (c for c in ["true_label", "attack_cat", "zero_day_family", "label"] if c in raw_df.columns),
                                None,
                            )
                            if family_col:
                                result_df["zero_day_family"] = raw_df[family_col].astype(str).reset_index(drop=True)
                            else:
                                result_df["zero_day_family"] = ""
                            result_df["detection"] = np.where(
                                result_df["is_zeroday"].astype(bool),
                                "Zero-Day / " + result_df["zero_day_family"].replace("", "Unknown").astype(str),
                                "Known / " + result_df["predicted_class"].astype(str),
                            )
                            st.session_state['bulk_result_df'] = result_df
                            st.session_state['bulk_raw_df'] = raw_df.reset_index(drop=True)

            result_df = st.session_state.get('bulk_result_df')
            if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                st.success(f"Da phan tich {len(result_df):,} dong.")

                # Luu alert gan nhat de tab Ask AI co the su dung
                try:
                    top_idx = result_df['hybrid_score'].idxmax()
                    top_row = result_df.loc[top_idx]
                    st.session_state['last_alert'] = {
                        "alert_id": f"BULK-{int(top_idx):06d}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "hybrid_score": float(top_row['hybrid_score']),
                        "ae_score": float(top_row['ae_score']),
                        "max_prob": float(top_row['max_prob']),
                        "predicted_class": str(top_row['predicted_class']),
                        "is_zeroday": bool(top_row['is_zeroday']),
                        "demo_mode": False,
                    }
                except Exception:
                    pass

                max_rows = min(1000, len(result_df))
                rows_to_show = st.slider(
                    "So dong muon xem",
                    min_value=10,
                    max_value=max_rows,
                    value=min(100, max_rows),
                    step=10,
                )
                st.dataframe(result_df.head(rows_to_show), use_container_width=True)

                total = len(result_df)
                zd_cnt = int(result_df['is_zeroday'].sum())
                st.metric("Zero-Day detected", zd_cnt)
                st.metric("Zero-Day rate", f"{(zd_cnt/total*100):.2f}%")

                st.download_button(
                    "Download ket qua (CSV)",
                    data=result_df.to_csv(index=False).encode('utf-8'),
                    file_name="ids_bulk_results.csv",
                    mime="text/csv"
                )

# ═════════════════════════════════════════════════════════════════
# PAGE: Zero-Day Logs
# ═════════════════════════════════════════════════════════════════
elif page == "[3] Zero-Day Logs":
    render_soc_header(
        "Zero-Day Logs",
        "Zero-day verdicts are shown separately from the known-class classifier so analysts can see both the anomaly decision and the original attack family when available.",
    )

    result_df = st.session_state.get('bulk_result_df')
    raw_df = st.session_state.get('bulk_raw_df')

    if not isinstance(result_df, pd.DataFrame) or result_df.empty:
        st.info("Chua co batch log. Vao Analyze Alert -> Upload CSV -> Phan tich TOAN BO file truoc.")
    else:
        logs = result_df.copy()
        if "source_row" not in logs.columns:
            logs.insert(0, "source_row", np.arange(len(logs)))
        if "zero_day_family" not in logs.columns:
            family_col = None
            if isinstance(raw_df, pd.DataFrame):
                family_col = next(
                    (c for c in ["true_label", "attack_cat", "zero_day_family", "label"] if c in raw_df.columns),
                    None,
                )
            logs["zero_day_family"] = (
                raw_df[family_col].astype(str).reset_index(drop=True)
                if family_col else ""
            )
        if "detection" not in logs.columns:
            family = logs["zero_day_family"].replace("", "Unknown").astype(str)
            logs["detection"] = np.where(
                logs["is_zeroday"].astype(bool),
                "Zero-Day / " + family,
                "Known / " + logs["predicted_class"].astype(str),
            )

        zd_logs = logs[logs["is_zeroday"].astype(bool)].copy()
        total = len(logs)
        zd_total = len(zd_logs)
        attack_classes = sorted(logs["predicted_class"].astype(str).unique().tolist())
        families = sorted([x for x in logs["zero_day_family"].astype(str).unique().tolist() if x and x != "nan"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Batch Rows", f"{total:,}")
        c2.metric("Zero-Day Logs", f"{zd_total:,}")
        c3.metric("Zero-Day Rate", f"{(zd_total / total * 100):.2f}%" if total else "0.00%")
        c4.metric("Known Classifier Classes", len(attack_classes))
        c5.metric("ZD Families", len(families))

        st.markdown("### Filters")
        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
        show_only_zd = f1.toggle("Only Zero-Day", value=True)
        selected_family = f2.selectbox("Zero-day family", ["All"] + families)
        selected_class = f3.selectbox("Classifier class", ["All"] + attack_classes)
        min_score = f4.number_input("Min hybrid score", min_value=0.0, value=0.0, step=0.1)

        view_df = zd_logs if show_only_zd else logs
        if selected_family != "All":
            view_df = view_df[view_df["zero_day_family"].astype(str) == selected_family]
        if selected_class != "All":
            view_df = view_df[view_df["predicted_class"].astype(str) == selected_class]
        view_df = view_df[view_df["hybrid_score"] >= min_score].copy()
        view_df["risk"] = view_df.apply(lambda r: risk_score({
            "hybrid_score": r.get("hybrid_score", 0),
            "ae_score": r.get("ae_score", 0),
            "is_zeroday": bool(r.get("is_zeroday", False)),
        }, "HIGH" if r.get("is_zeroday", False) else "MEDIUM"), axis=1)
        view_df = view_df.sort_values(["risk", "hybrid_score", "ae_score"], ascending=False)

        left_col, right_col = st.columns([1.55, 1.0], gap="large")
        selected_source_row = None

        with left_col:
            st.markdown("### Detection Log")
            if view_df.empty:
                st.info("Khong co log nao khop filter hien tai.")
            else:
                display_cols = [c for c in [
                    "source_row", "detection", "zero_day_family", "predicted_class",
                    "risk", "hybrid_score", "ae_score", "max_prob", "is_zeroday"
                ] if c in view_df.columns]
                try:
                    event = st.dataframe(
                        view_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        selection_mode="single-row",
                        on_select="rerun",
                    )
                    selected_rows = event.selection.rows if event and hasattr(event, "selection") else []
                    if selected_rows:
                        selected_source_row = int(view_df.iloc[selected_rows[0]]["source_row"])
                except TypeError:
                    st.dataframe(view_df[display_cols], use_container_width=True, hide_index=True)

                if selected_source_row is None:
                    labels = []
                    for _, r in view_df.head(500).iterrows():
                        labels.append(
                            f"row={int(r['source_row'])} | {r.get('detection', r.get('predicted_class'))} | "
                            f"hybrid={float(r['hybrid_score']):.3g} | ae={float(r['ae_score']):.3g}"
                        )
                    selected_label = st.selectbox("Select log row", labels)
                    selected_pos = labels.index(selected_label)
                    selected_source_row = int(view_df.head(500).iloc[selected_pos]["source_row"])

            st.download_button(
                "Download filtered zero-day log",
                data=view_df.to_csv(index=False).encode("utf-8"),
                file_name="zero_day_logs_filtered.csv",
                mime="text/csv",
                disabled=view_df.empty,
            )

        with right_col:
            st.markdown("### Event Detail")
            if selected_source_row is None:
                st.markdown(
                    """
                    <div class="soc-panel">
                        Select a row on the left to inspect classifier output, zero-day verdict, MITRE hypothesis and full features.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                row_scores = logs[logs["source_row"] == selected_source_row].iloc[0].to_dict()
                row_scores["risk"] = risk_score({
                    "hybrid_score": row_scores.get("hybrid_score", 0),
                    "ae_score": row_scores.get("ae_score", 0),
                    "is_zeroday": bool(row_scores.get("is_zeroday", False)),
                }, "HIGH" if row_scores.get("is_zeroday") else "MEDIUM")
                family = str(row_scores.get("zero_day_family") or "Unknown")
                classifier_class = str(row_scores.get("predicted_class", "Unknown"))
                detection = str(row_scores.get("detection") or ("Zero-Day / " + family if row_scores.get("is_zeroday") else classifier_class))
                verdict_badge = "ZERO-DAY" if row_scores.get("is_zeroday") else "KNOWN"
                badge_class = "soc-pill-red" if row_scores.get("is_zeroday") else "soc-pill-green"

                st.markdown(
                    f"""
                    <div class="soc-panel">
                        <span class="soc-badge {badge_class}">{verdict_badge}</span>
                        <span class="soc-badge soc-pill-blue">row {selected_source_row}</span>
                        <div class="soc-detail-title" style="margin-top:12px;">Detection</div>
                        <div class="soc-detail-value">{detection}</div>
                        <div class="soc-detail-title">Zero-Day Family / Ground Truth</div>
                        <div class="soc-detail-value">{family}</div>
                        <div class="soc-detail-title">Known-Class Classifier Output</div>
                        <div class="soc-detail-value">{classifier_class}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                d1, d2 = st.columns(2)
                d1.metric("Hybrid", f"{float(row_scores.get('hybrid_score', 0)):.3g}")
                d2.metric("AE / VAE", f"{float(row_scores.get('ae_score', 0)):.3g}")
                d3, d4 = st.columns(2)
                d3.metric("Max Prob", f"{float(row_scores.get('max_prob', 0)):.3f}")
                d4.metric("Risk", f"{int(row_scores.get('risk', 0))}/100")
                if st.button("Ask AI about this log", type="primary", key=f"ask_ai_log_{selected_source_row}"):
                    alert_context = build_alert_context_from_log(row_scores)
                    if HAS_MITRE:
                        mapper = MITREMapper()
                        if alert_context["is_zeroday"]:
                            mitre_res = mapper.map_zeroday(alert_context["ae_score"], [])
                        else:
                            class_idx = CLASS_NAMES.index(classifier_class) if classifier_class in CLASS_NAMES else 0
                            mitre_res = mapper.map_known_attack(class_idx, class_names=CLASS_NAMES)
                        alert_context["mitre_result"] = mitre_res
                        alert_context["mitre_summary"] = mapper.format_for_llm(mitre_res)
                    st.session_state["last_alert"] = alert_context
                    st.session_state["ask_ai_context_source"] = "zero_day_log"
                    st.session_state["messages"] = []
                    st.session_state["nav_page"] = "[4] Ask AI"
                    st.rerun()

                tabs = st.tabs(["Full Features", "Scores", "MITRE"])
                with tabs[0]:
                    if isinstance(raw_df, pd.DataFrame) and selected_source_row < len(raw_df):
                        feature_row = raw_df.iloc[selected_source_row]
                        feature_table = pd.DataFrame({
                            "Feature": feature_row.index.astype(str),
                            "Value": feature_row.astype(str).values,
                        })
                        priority = feature_table["Feature"].isin([
                            "true_label", "attack_cat", "label", "dur", "sbytes", "dbytes",
                            "sload", "dload", "spkts", "dpkts", "ct_srv_dst", "ct_dst_ltm",
                            "ct_src_ltm", "state_num", "proto_num", "service_num",
                        ])
                        feature_table = pd.concat([feature_table[priority], feature_table[~priority]], ignore_index=True)
                        search = st.text_input("Search feature", "")
                        if search:
                            feature_table = feature_table[
                                feature_table["Feature"].str.contains(search, case=False, na=False)
                            ]
                        st.dataframe(feature_table, use_container_width=True, hide_index=True, height=430)
                    else:
                        st.warning("Khong tim thay raw feature row tu batch upload.")

                with tabs[1]:
                    score_table = pd.DataFrame([
                        {"Metric": k, "Value": v}
                        for k, v in row_scores.items()
                        if k != "source_row"
                    ])
                    st.dataframe(score_table, use_container_width=True, hide_index=True, height=360)

                with tabs[2]:
                    if HAS_MITRE:
                        mapper = MITREMapper()
                        if bool(row_scores.get("is_zeroday")):
                            mitre_res = mapper.map_zeroday(float(row_scores.get("ae_score", 0)), [])
                        else:
                            class_idx = CLASS_NAMES.index(classifier_class) if classifier_class in CLASS_NAMES else 0
                            mitre_res = mapper.map_known_attack(class_idx, class_names=CLASS_NAMES)
                        techniques = mitre_res.get("techniques") or mitre_res.get("suspected_techniques", [])
                        st.markdown(
                            f"**Mode:** `{mitre_res.get('mapping_mode', 'unknown')}` | "
                            f"**Confidence:** `{mitre_res.get('confidence', 'LOW')}`"
                        )
                        if techniques:
                            st.dataframe(pd.DataFrame([{
                                "Technique": f"{t['id']} - {t['name']}",
                                "Tactic": t.get("tactic", ""),
                                "Confidence": t.get("confidence", ""),
                                "Rationale": t.get("rationale", ""),
                                "ATT&CK": t.get("url", ""),
                            } for t in techniques]), use_container_width=True, hide_index=True)
                        st.caption(mitre_res.get("coverage_note", ""))
                    else:
                        st.warning("MITRE module chua duoc kich hoat.")

# ═════════════════════════════════════════════════════════════════
# PAGE: Ask AI
# ═════════════════════════════════════════════════════════════════
elif page == "[4] Ask AI":
    render_soc_header(
        "SOC AI Analyst",
        "Ask follow-up questions against the latest alert context, MITRE mapping and model evidence.",
    )

    context_options = []
    history = st.session_state.get("alert_history", [])
    for i, item in enumerate(history[-20:]):
        context_options.append((
            f"Alert history | {item.get('alert_id')} | {item.get('predicted_class')} | score={float(item.get('hybrid_score', 0)):.3g}",
            item,
        ))

    bulk_logs = st.session_state.get("bulk_result_df")
    if isinstance(bulk_logs, pd.DataFrame) and not bulk_logs.empty:
        logs_for_ai = bulk_logs.copy()
        if "source_row" not in logs_for_ai.columns:
            logs_for_ai.insert(0, "source_row", np.arange(len(logs_for_ai)))
        logs_for_ai = logs_for_ai.sort_values(["is_zeroday", "hybrid_score", "ae_score"], ascending=False).head(50)
        for _, row in logs_for_ai.iterrows():
            row_dict = row.to_dict()
            context_options.append((
                f"Zero-day log | row={int(row_dict.get('source_row', 0))} | "
                f"{row_dict.get('detection', row_dict.get('predicted_class'))} | "
                f"hybrid={float(row_dict.get('hybrid_score', 0)):.3g}",
                build_alert_context_from_log(row_dict),
            ))

    if context_options:
        labels = [x[0] for x in context_options]
        current_alert_id = st.session_state.get("last_alert", {}).get("alert_id")
        default_idx = 0
        for i, (_, ctx) in enumerate(context_options):
            if ctx.get("alert_id") == current_alert_id:
                default_idx = i
                break
        selected_context = st.selectbox("AI context", labels, index=default_idx)
        selected_ctx = context_options[labels.index(selected_context)][1]
        if st.button("Use selected context", key="use_ai_context"):
            st.session_state["last_alert"] = selected_ctx
            st.session_state["messages"] = []
            st.rerun()

    if 'last_alert' not in st.session_state:
        st.warning("Chua co alert/log nao. Hay phan tich 1 alert hoac upload CSV batch truoc.")
        st.stop()

    last = st.session_state['last_alert']
    st.markdown(
        f"""
        <div class="soc-panel">
            <span class="soc-badge soc-pill-blue">Context {last.get('alert_id', 'N/A')}</span>
            <span class="soc-badge">{last.get('predicted_class', 'Unknown')}</span>
            <span class="soc-badge">hybrid {float(last.get('hybrid_score', 0)):.3g}</span>
            <span class="soc-badge">ae {float(last.get('ae_score', 0)):.3g}</span>
            <div class="soc-detail-title" style="margin-top:10px;">Current AI Context</div>
            <div class="soc-detail-value">{last.get('zero_day_family', last.get('classifier_class', ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Goi y cau hoi
    st.markdown("**Goi y cau hoi:**")
    suggestions = [
        "Tai sao alert nay bi danh dau la nguy hiem?",
        "MITRE technique nao lien quan nhat?",
        "Buoc xu ly khan cap la gi?",
        "Day co the la false positive khong?",
        "Giai thich feature bat thuong nhat cho toi hieu",
    ]
    cols = st.columns(len(suggestions))
    for i, (col, q) in enumerate(zip(cols, suggestions)):
        if col.button(q[:25]+"...", key=f"sug_{i}"):
            st.session_state['pending_question'] = q

    st.markdown("---")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role_label = "Analyst" if msg["role"] == "user" else "SOC AI"
        with st.chat_message(msg["role"]):
            st.write(f"**{role_label}:** {msg['content']}")

    # Xu ly cau hoi tu goi y
    pending = st.session_state.pop('pending_question', None)

    question = st.chat_input("Hoi AI ve alert nay...") or pending

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(f"**Analyst:** {question}")

        with st.chat_message("assistant"):
            with st.spinner("Dang suy nghi..."):
                if HAS_LLM:
                    try:
                        agent = SOCTriageAgent()
                        answer = agent.explain_to_analyst(question, last)
                    except Exception as e:
                        answer = f"Loi LLM Agent: {e}. Vui long kiem tra lai provider va API Key trong file .env."
                else:
                    if LLM_DEP and not HAS_LLM_DEPS:
                        answer = f"Chua cai thu vien cho LLM provider '{LLM_PROVIDER}'. Can cai: {LLM_DEP}."
                    else:
                        answer = "Khong tim thay llm_agent.py. Vui long kiem tra lai source code."

            st.write(f"**SOC AI:** {answer}")
            st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("Xoa lich su chat"):
        st.session_state.messages = []
        st.rerun()

# ═════════════════════════════════════════════════════════════════
# PAGE: Setup Guide
# ═════════════════════════════════════════════════════════════════
elif page == "[5] Setup Guide":
    render_soc_header(
        "Deployment Checklist",
        "Runtime paths, model artifacts, data availability and optional provider configuration.",
    )

    st.markdown("### Buoc 1: Cau truc thu muc")
    st.code("""
SOC-AI-Platform-v15/
├── dashboard/
│   └── app.py          <- file nay
├── src/
│   ├── explainer.py
│   ├── mitre_mapper.py
│   └── llm_agent.py
├── checkpoints/
│   ├── ids_v14_model.pth
│   └── ids_v14_pipeline.pkl
├── data/
│   └── UNSW_NB15_training-set.csv
└── .env
    """)

    st.markdown("### Buoc 2: Export model tu IDS v14")
    st.code("""
# Them vao cuoi file ids_v14_unswnb15.py cua ban:
import pickle, torch

# Save model
torch.save({'model': model}, 'checkpoints/ids_v14_model.pth')

# Save pipeline (scaler + feature names)
pipeline = {
    'scaler'       : scaler,
    'feature_names': list(X_train.columns),
    'label_encoder': le,
}
with open('checkpoints/ids_v14_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Da luu model va pipeline!")
    """, language="python")

    st.markdown("### Buoc 3: Cau hinh API Key cho LLM")
    st.markdown("1. Dang ky API Key tai nha cung cap ban chon (Vi du: https://console.groq.com/keys)")
    st.markdown("2. Mo file `src/llm_agent.py` va chon `LLM_PROVIDER`")
    st.markdown("3. Tao file `.env` trong thu muc goc va them key tuong ung:")
    st.code("GROQ_API_KEY=gsk_...\n# GEMINI_API_KEY=AIzaSy...\n# OPENAI_API_KEY=sk-...", language="bash")

    st.markdown("**Ghi chu:** Neu thieu pipeline, app se co gang tao tu data/ khi khoi dong.")

    st.markdown("### Buoc 4: Chay app")
    st.code("""
cd dashboard
streamlit run app.py
    """, language="bash")

    st.markdown("### Trang thai hien tai:")
    status_items = {
        f"Model file ({os.path.basename(MODEL_PATH)})": os.path.exists(MODEL_PATH),
        f"Pipeline file ({os.path.basename(PIPE_PATH)})": os.path.exists(PIPE_PATH),
        f"Data file ({os.path.basename(DATA_PATH)})": os.path.exists(DATA_PATH),
        "API Key (Bat ky trong .env)": any([os.getenv("GROQ_API_KEY"), os.getenv("GEMINI_API_KEY"), os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY")]),
        "Module explainer.py": HAS_EXPLAINER,
        "Module mitre_mapper.py": HAS_MITRE,
        "Module llm_agent.py": HAS_LLM,
    }
    for item, ok in status_items.items():
        icon = "OK" if ok else "MISSING"
        color = "green" if ok else "red"
        st.markdown(f":{color}[{icon}] {item}")
