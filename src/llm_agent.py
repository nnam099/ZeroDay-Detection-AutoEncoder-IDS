# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── CHON LLM O DAY ────────────────────────────────────────────
LLM_PROVIDER = "groq"   # "groq" | "gemini" | "openai" | "anthropic"
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
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())

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
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "").strip())

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
    print(f"[LLM] Loi khoi tao {LLM_PROVIDER}: {e}")
    # Fallback: tra ve message loi thay vi crash
    def _call_llm(prompt: str) -> str:
        return json.dumps({
            "severity": "HIGH",
            "verdict": f"LLM chua san sang: {e}",
            "attack_summary": "Kiem tra lai API key va provider trong llm_agent.py",
            "recommended_actions": ["Kiem tra .env file", "Kiem tra API key con han"],
            "false_positive_risk": "UNKNOWN",
            "false_positive_reason": "N/A",
            "analyst_note": str(e),
        }, ensure_ascii=False)


class SOCTriageAgent:
    """
    Nhan output tu IDS v14 + SHAP + MITRE → LLM phan tich → tra ve structured report.
    """

    SYSTEM_CONTEXT = (
        "Ban la mot SOC Analyst AI chuyen nghiep. "
        "Nhiem vu: phan tich alert tu he thong IDS va dua ra danh gia ro rang. "
        "Luon tra loi bang JSON hop le theo schema duoc yeu cau. "
        "Ngon ngu: Tieng Viet, ngan gon, suc tich, chuyen nghiep."
    )

    def triage_alert(self, alert_data: dict) -> dict:
        zeroday_str = "CO ⚠" if alert_data.get('is_zeroday') else "KHONG"

        prompt = f"""NHIEM VU: Phan tich alert bao mat va tra ve DUY NHAT mot JSON object. TUYET DOI KHONG viet code, markdown, giai thich, hay bat ky text nao ngoai JSON.

=== DU LIEU ALERT ===
Alert ID    : {alert_data.get('alert_id', 'N/A')}
Thoi gian   : {alert_data.get('timestamp', 'N/A')}
Hybrid Score: {alert_data.get('hybrid_score', 0):.4f} (nguong canh bao: >0.5)
AE Score    : {alert_data.get('ae_score', 0):.4f} (loi tai tao - cao = bat thuong)
Phan loai   : {alert_data.get('predicted_class', 'Unknown')} (do tin cay: {alert_data.get('max_prob', 0):.1%})
Zero-Day    : {zeroday_str}
SHAP        : {alert_data.get('shap_summary', 'Chua co')[:300]}
MITRE       : {alert_data.get('mitre_summary', 'Chua co')[:200]}

=== YEU CAU OUTPUT ===
Chi tra ve JSON object theo dung dinh dang sau, khong them gi khac:
{{"severity":"CRITICAL|HIGH|MEDIUM|LOW","verdict":"<1 cau mo ta alert>","attack_summary":"<giai thich dua tren SHAP va MITRE>","recommended_actions":["<hanh dong 1>","<hanh dong 2>","<hanh dong 3>"],"false_positive_risk":"HIGH|MEDIUM|LOW","false_positive_reason":"<ly do>","analyst_note":"<ghi chu>"}}"""
        raw = _call_llm(prompt)

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
                "analyst_note"       : "Kiem tra lai prompt va LLM response",
            }

        # Them metadata
        result["alert_id"]     = alert_data.get("alert_id", "")
        result["timestamp"]    = alert_data.get("timestamp", "")
        result["hybrid_score"] = alert_data.get("hybrid_score", 0)
        return result

    def explain_to_analyst(self, question: str, alert_context: dict) -> str:
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

Context ve alert hien tai:
{json.dumps(safe_context, ensure_ascii=False, indent=2, default=str)}

Cau hoi cua analyst: {question}

Tra loi ngan gon, ro rang, dung thuat ngu SOC chuyen nghiep bang Tieng Viet.
"""
        return _call_llm(prompt)