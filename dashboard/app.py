# -*- coding: utf-8 -*-
"""
SOC AI Platform v15 - Dashboard
Compatible: Windows, Python 3.9+
Run: streamlit run app.py
"""

import streamlit as st
import sys
import os
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

# ── Them src/ vao path de import module ──────────────────────────
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC_DIR)

# ── Kiem tra thu vien can thiet ───────────────────────────────────
MISSING = []
try:
    import torch
except ImportError:
    MISSING.append("torch")
try:
    import shap
except ImportError:
    MISSING.append("shap")
try:
    import google.generativeai as genai
except ImportError:
    MISSING.append("google-generativeai")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    MISSING.append("python-dotenv")

if MISSING:
    st.error(f"Thieu thu vien: {', '.join(MISSING)}")
    st.code(f"pip install {' '.join(MISSING)}")
    st.stop()

# ── Import modules v15 ────────────────────────────────────────────
try:
    from explainer import SHAPExplainer
    HAS_EXPLAINER = True
except ImportError:
    HAS_EXPLAINER = False
    st.warning("[!] Khong tim thay explainer.py trong src/ - SHAP se bi tat")

try:
    from mitre_mapper import MITREMapper
    HAS_MITRE = True
except ImportError:
    HAS_MITRE = False
    st.warning("[!] Khong tim thay mitre_mapper.py trong src/")

try:
    from llm_agent import SOCTriageAgent
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    st.warning("[!] Khong tim thay llm_agent.py trong src/")

# ── Duong dan checkpoint (chinh lai neu can) ──────────────────────
BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CKPT_DIR   = os.path.join(BASE_DIR, 'checkpoints')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(CKPT_DIR, 'ids_v14_model.pth')
PIPE_PATH  = os.path.join(CKPT_DIR, 'ids_v14_pipeline.pkl')
DATA_PATH  = os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')

# CLASS_NAMES se duoc load tu label_encoder sau khi model load
# De tranh sai thu tu do LabelEncoder sort theo alphabet
CLASS_NAMES  = ["Normal", "DoS", "Exploits", "Reconnaissance", "Generic"]  # fallback
AE_THRESHOLD = 0.5

# ── Sidebar navigation ────────────────────────────────────────────
st.sidebar.title("SOC AI Platform v15")
st.sidebar.caption("IDS v14 + SHAP + MITRE ATT&CK + LLM")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["[1] Dashboard", "[2] Analyze Alert", "[3] Ask AI", "[4] Setup Guide"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Status:**")
st.sidebar.write("SHAP:", "OK" if HAS_EXPLAINER else "MISSING")
st.sidebar.write("MITRE:", "OK" if HAS_MITRE else "MISSING")
st.sidebar.write("LLM:", "OK" if HAS_LLM else "MISSING")
st.sidebar.write("Model:", "OK" if os.path.exists(MODEL_PATH) else "NOT FOUND")
st.sidebar.write("Data:", "OK" if os.path.exists(DATA_PATH) else "NOT FOUND")

# ── Load model (cache) ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None
    try:
        # Import IDSModel từ src/ (da duoc them vao sys.path o tren)
        from ids_v14_unswnb15 import IDSModel

        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        n_feat     = checkpoint['n_features']
        n_cls      = checkpoint['n_classes']
        # Lay hidden size truc tiep tu checkpoint de tranh size mismatch
        hidden    = checkpoint.get('hidden', 256)
        ae_hidden = checkpoint.get('ae_hidden', 128)

        model = IDSModel(n_features=n_feat, n_classes=n_cls, hidden=hidden, ae_hidden=ae_hidden)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        with open(PIPE_PATH, 'rb') as f:
            pipeline = pickle.load(f)

        scaler        = pipeline['scaler']
        feature_names = pipeline.get('feature_names', pipeline.get('feat_cols', []))
        label_encoder = pipeline['label_encoder']

        return model, scaler, feature_names, label_encoder

    except Exception as e:
        st.error(f"Loi load model: {e}")
        return None, None, None, None

model, scaler, feature_names, label_encoder = load_model()

# Lay CLASS_NAMES dung thu tu tu label_encoder (tranh sai do sort alphabet)
if label_encoder is not None:
    CLASS_NAMES = list(label_encoder.classes_)

def preprocess_raw_df(df_raw: pd.DataFrame, feat_cols: list) -> np.ndarray:
    """
    Ap dung dung cac buoc tien xu ly nhu trong prepare_splits().
    Output LUON CO DUNG SO LUONG FEATURES = len(feat_cols), thu tu khop voi scaler.
    """
    from ids_v14_unswnb15 import engineer_features
    from sklearn.preprocessing import LabelEncoder as _LE

    df = df_raw.copy()

    # 1. Encode cac cot categorical -> _num columns
    for cat in ['proto', 'service', 'state']:
        if cat in df.columns:
            le = _LE()
            df[f'{cat}_num'] = le.fit_transform(df[cat].astype(str).fillna('unk'))

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
DEMO_MODE = (model is None)

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
                mitre_res = mapper.map_known_attack(pred_idx)
            result['mitre_result']  = mitre_res
            result['mitre_summary'] = mapper.format_for_llm(mitre_res)
        except Exception as e:
            result['mitre_summary'] = f"MITRE loi: {e}"
            result['mitre_result']  = None
    else:
        result['mitre_summary'] = "MITRE chua duoc kich hoat"
        result['mitre_result']  = None

    return result

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
if page == "[1] Dashboard":
    st.title("SOC AI Platform v15 - Dashboard")
    st.caption("IDS v14 Zero-Day Detection + SHAP + MITRE ATT&CK + LLM Reasoning")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    history = st.session_state.get('alert_history', [])
    c1.metric("Tong alerts da phan tich", len(history))
    zd = sum(1 for a in history if a.get('is_zeroday'))
    c2.metric("Zero-Day detected", zd)
    hi = sum(1 for a in history if a.get('llm_severity') in ['CRITICAL','HIGH'])
    c3.metric("Critical/High", hi)

    if history:
        st.markdown("### Lich su alerts")
        hist_df = pd.DataFrame([{
            "Alert ID"    : a['alert_id'],
            "Time"        : a['timestamp'],
            "Class"       : a['predicted_class'],
            "Hybrid Score": round(a['hybrid_score'], 3),
            "Zero-Day"    : "YES" if a['is_zeroday'] else "NO",
            "Severity"    : a.get('llm_severity', 'N/A'),
        } for a in history])
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("Chua co alert nao. Vao tab '[2] Analyze Alert' de bat dau phan tich.")

    if DEMO_MODE:
        st.warning("Model chua duoc load. Xem tab '[4] Setup Guide' de biet cach cai dat.")

# ═════════════════════════════════════════════════════════════════
# PAGE: Analyze Alert
# ═════════════════════════════════════════════════════════════════
elif page == "[2] Analyze Alert":
    st.title("Analyze Alert")

    if DEMO_MODE:
        st.warning("DEMO MODE: Model chua duoc load. Dang dung du lieu gia lap.")

    bg_data = None
    comps   = {}

    if not DEMO_MODE:
        bg_data = load_background(scaler, feature_names)
        comps   = get_components(model, scaler, feature_names, bg_data)

    mode = st.radio("Chon input:", ["Random sample tu dataset", "Nhap thu cong (demo)", "Upload CSV"])

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
                    mr = mapper.map_known_attack(CLASS_NAMES.index(atk_cls) if atk_cls in CLASS_NAMES else 1)
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
            if st.button("Phan tich file nay"):
                if DEMO_MODE:
                    result = mock_inference()
                    llm    = get_llm_analysis(result, comps)
                else:
                    # Dung preprocess_raw_df de xu ly giong luc train
                    raw = preprocess_raw_df(raw_df.head(1), feature_names or [])
                    if raw.shape[1] == 0 or (raw == 0).all():
                        st.error(
                            "Khong tim thay features hop le trong file CSV nay. "
                            "Vui long upload file CSV co cac cot feature goc cua UNSW-NB15 "
                            "(khong phai file ket qua phan tich)."
                        )
                    else:
                        result = run_full_pipeline(raw, comps)
                        llm    = get_llm_analysis(result, comps)
                        display_result(result, llm)

# ═════════════════════════════════════════════════════════════════
# PAGE: Ask AI
# ═════════════════════════════════════════════════════════════════
elif page == "[3] Ask AI":
    st.title("Chat voi SOC AI")

    if 'last_alert' not in st.session_state:
        st.warning("Chua co alert nao. Hay phan tich 1 alert o tab '[2] Analyze Alert' truoc.")
        st.stop()

    last = st.session_state['last_alert']
    st.caption(f"Context hien tai: Alert #{last['alert_id']} | {last['predicted_class']} | Score: {last['hybrid_score']:.3f}")

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
                    answer = "Khong tim thay llm_agent.py. Vui long kiem tra lai source code."

            st.write(f"**SOC AI:** {answer}")
            st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("Xoa lich su chat"):
        st.session_state.messages = []
        st.rerun()

# ═════════════════════════════════════════════════════════════════
# PAGE: Setup Guide
# ═════════════════════════════════════════════════════════════════
elif page == "[4] Setup Guide":
    st.title("Setup Guide")

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

    st.markdown("### Buoc 4: Chay app")
    st.code("""
cd dashboard
streamlit run app.py
    """, language="bash")

    st.markdown("### Trang thai hien tai:")
    status_items = {
        "Model file (ids_v14_model.pth)": os.path.exists(MODEL_PATH),
        "Pipeline file (ids_v14_pipeline.pkl)": os.path.exists(PIPE_PATH),
        "Data file (UNSW_NB15_training-set.csv)": os.path.exists(DATA_PATH),
        "API Key (Bat ky trong .env)": any([os.getenv("GROQ_API_KEY"), os.getenv("GEMINI_API_KEY"), os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY")]),
        "Module explainer.py": HAS_EXPLAINER,
        "Module mitre_mapper.py": HAS_MITRE,
        "Module llm_agent.py": HAS_LLM,
    }
    for item, ok in status_items.items():
        icon = "OK" if ok else "MISSING"
        color = "green" if ok else "red"
        st.markdown(f":{color}[{icon}] {item}")