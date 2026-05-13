# -*- coding: utf-8 -*-
"""
llm_agent.py — SOC Triage Agent v15
=====================================
Cải tiến so với v14:
  [NEW]  Nhận thêm attention_summary từ AttentionGate (v15 model)
  [NEW]  Nhận thêm ood_ensemble_score từ OOD Ensemble detector
  [NEW]  Nhận thêm knn_dist_score từ KNN OOD detector
  [UPD]  Prompt cập nhật để phân tích đầy đủ hơn với dữ liệu v15
  [UPD]  JSON schema thêm trường ood_confidence
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── CHON LLM O DAY ────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
# ──────────────────────────────────────────────────────────────


def _build_client():
    if LLM_PROVIDER == "groq":
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Chua co GROQ_API_KEY trong file .env")
        client = Groq(api_key=api_key)

        def call(prompt: str) -> str:
            r = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            return r.choices[0].message.content
        return call

    elif LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Chua co GEMINI_API_KEY trong file .env")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        def call(prompt: str) -> str:
            return model.generate_content(prompt).text
        return call

    elif LLM_PROVIDER == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Chua co OPENAI_API_KEY trong file .env")
        client = OpenAI(api_key=api_key)

        def call(prompt: str) -> str:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return r.choices[0].message.content
        return call

    elif LLM_PROVIDER == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Chua co ANTHROPIC_API_KEY trong file .env")
        client = anthropic.Anthropic(api_key=api_key)

        def call(prompt: str) -> str:
            r = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text
        return call

    else:
        raise ValueError(f"LLM_PROVIDER khong hop le: {LLM_PROVIDER}")


# Khoi tao client 1 lan duy nhat
try:
    _call_llm = _build_client()
    print(f"[LLM] Provider: {LLM_PROVIDER} - OK")
except Exception as e:
    init_error = str(e)
    print(f"[LLM] Loi khoi tao {LLM_PROVIDER}: {init_error}")
    # Fallback: tra ve message loi thay vi crash
    def _call_llm(prompt: str) -> str:
        return json.dumps({
            "severity": "HIGH",
            "verdict": f"LLM chua san sang: {init_error}",
            "attack_summary": "Kiem tra lai API key va provider trong llm_agent.py",
            "recommended_actions": ["Kiem tra .env file", "Kiem tra API key con han"],
            "false_positive_risk": "UNKNOWN",
            "false_positive_reason": "N/A",
            "ood_confidence": "UNKNOWN",
            "analyst_note": init_error,
        }, ensure_ascii=False)


class SOCTriageAgent:
    """
    Nhan output tu IDS v15 + SHAP/Attention + MITRE → LLM phan tich → structured report.

    alert_data keys (v15):
        alert_id, timestamp,
        hybrid_score, ae_score,            # v14 compatible
        vae_recon_score,                   # [NEW v15] VAE reconstruction error
        ood_ensemble_score,                # [NEW v15] vote majority score
        knn_dist_score,                    # [NEW v15] KNN feature space distance
        predicted_class, max_prob,
        is_zeroday,
        shap_summary,                      # SHAP top features text
        attention_summary,                 # [NEW v15] AttentionGate weights text
        mitre_summary,
    """

    SYSTEM_CONTEXT = (
        "Ban la mot SOC Analyst AI chuyen nghiep. "
        "Nhiem vu: phan tich alert tu he thong IDS v15 va dua ra danh gia ro rang. "
        "Luon tra loi bang JSON hop le theo schema duoc yeu cau. "
        "Ngon ngu: Tieng Viet, ngan gon, suc tich, chuyen nghiep."
    )

    def triage_alert(self, alert_data: dict) -> dict:
        zeroday_str = "CO ⚠" if alert_data.get('is_zeroday') else "KHONG"

        # [NEW v15] Thêm các score mới vào prompt
        vae_score     = alert_data.get('vae_recon_score', alert_data.get('ae_score', 0))
        ood_ens_score = alert_data.get('ood_ensemble_score', 'N/A')
        knn_score     = alert_data.get('knn_dist_score', 'N/A')
        attn_summary  = alert_data.get('attention_summary', 'Chua co')

        # Format OOD scores
        ood_ens_str = f"{ood_ens_score:.4f}" if isinstance(ood_ens_score, float) else str(ood_ens_score)
        knn_str     = f"{knn_score:.4f}"     if isinstance(knn_score, float)     else str(knn_score)

        prompt = f"""NHIEM VU: Phan tich alert bao mat va tra ve DUY NHAT mot JSON object. TUYET DOI KHONG viet code, markdown, giai thich, hay bat ky text nao ngoai JSON.

=== DU LIEU ALERT (IDS v15) ===
Alert ID        : {alert_data.get('alert_id', 'N/A')}
Thoi gian       : {alert_data.get('timestamp', 'N/A')}
Hybrid Score    : {alert_data.get('hybrid_score', 0):.4f} (nguong: >0.5)
VAE Recon Error : {vae_score:.4f} (cao = bất thường — v15 VAE)
OOD Ensemble    : {ood_ens_str} (vote majority ae_re+knn+hybrid — v15)
KNN Distance    : {knn_str} (khoang cach feature space — v15)
Phan loai       : {alert_data.get('predicted_class', 'Unknown')} (do tin cay: {alert_data.get('max_prob', 0):.1%})
Zero-Day        : {zeroday_str}
SHAP            : {alert_data.get('shap_summary', 'Chua co')[:300]}
Attention Gate  : {attn_summary[:200]}
MITRE           : {alert_data.get('mitre_summary', 'Chua co')[:200]}

=== YEU CAU OUTPUT ===
Chi tra ve JSON object theo dung dinh dang sau, khong them gi khac:
{{"severity":"CRITICAL|HIGH|MEDIUM|LOW","verdict":"<1 cau mo ta alert>","attack_summary":"<giai thich dua tren SHAP/Attention va MITRE>","recommended_actions":["<hanh dong 1>","<hanh dong 2>","<hanh dong 3>"],"false_positive_risk":"HIGH|MEDIUM|LOW","false_positive_reason":"<ly do>","ood_confidence":"HIGH|MEDIUM|LOW","analyst_note":"<ghi chu>"}}"""

        try:
            raw = _call_llm(prompt)
        except Exception as e:
            return {
                "severity"           : "HIGH",
                "verdict"            : "LLM gap loi - can review thu cong",
                "attack_summary"     : "LLM khong phan hoi. Kiem tra lai API key va provider.",
                "recommended_actions": ["Review thu cong", "Kiem tra API key"],
                "false_positive_risk": "UNKNOWN",
                "false_positive_reason": "LLM error",
                "ood_confidence"     : "UNKNOWN",
                "analyst_note"       : str(e),
                "alert_id"           : alert_data.get("alert_id", ""),
                "timestamp"          : alert_data.get("timestamp", ""),
                "hybrid_score"       : alert_data.get("hybrid_score", 0),
                "vae_recon_score"    : vae_score,
                "ood_ensemble_score" : alert_data.get("ood_ensemble_score", None),
                "knn_dist_score"     : alert_data.get("knn_dist_score", None),
            }

        # Parse JSON
        cleaned = raw.strip()
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    cleaned = part
                    break

        # Tim JSON object trong response
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            cleaned = cleaned[start:end]

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            result = {
                "severity"           : "HIGH",
                "verdict"            : "Phat hien traffic bat thuong - can review",
                "attack_summary"     : raw[:400] if raw else "Khong co du lieu",
                "recommended_actions": ["Review thu cong", "Escalate len Tier 2"],
                "false_positive_risk": "MEDIUM",
                "false_positive_reason": "LLM response parsing loi",
                "ood_confidence"     : "MEDIUM",
                "analyst_note"       : "Kiem tra lai prompt va LLM response",
            }

        # Them metadata
        result["alert_id"]            = alert_data.get("alert_id", "")
        result["timestamp"]           = alert_data.get("timestamp", "")
        result["hybrid_score"]        = alert_data.get("hybrid_score", 0)
        result["vae_recon_score"]     = vae_score
        result["ood_ensemble_score"]  = alert_data.get("ood_ensemble_score", None)
        result["knn_dist_score"]      = alert_data.get("knn_dist_score", None)
        return result

    def explain_to_analyst(self, question: str, alert_context: dict) -> str:
        """Chat tự do với analyst về một alert cụ thể."""
        # Loai bo numpy arrays khoi context truoc khi serialize
        safe_context = {}
        for k, v in alert_context.items():
            if k in ('shap_values', 'probs'):
                continue
            try:
                json.dumps(v)
                safe_context[k] = v
            except (TypeError, ValueError):
                safe_context[k] = str(v)

        prompt = f"""
{self.SYSTEM_CONTEXT}

Context ve alert hien tai (IDS v15):
{json.dumps(safe_context, ensure_ascii=False, indent=2, default=str)}

Cau hoi cua analyst: {question}

Tra loi ngan gon, ro rang, dung thuat ngu SOC chuyen nghiep bang Tieng Viet.
"""
        try:
            return _call_llm(prompt)
        except Exception as e:
            return f"LLM khong phan hoi. Kiem tra lai API key va provider. Loi: {e}"
