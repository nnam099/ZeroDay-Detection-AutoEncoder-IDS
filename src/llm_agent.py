import os
import json
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG: đổi LLM ở đây ──────────────────────────────────────────
LLM_PROVIDER = "gemini"   # "gemini" | "openai" | "anthropic"
# ───────────────────────────────────────────────────────────────────

if LLM_PROVIDER == "gemini":
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    _model = genai.GenerativeModel("gemini-1.5-flash")

    def _call_llm(prompt: str) -> str:
        response = _model.generate_content(prompt)
        return response.text

elif LLM_PROVIDER == "openai":
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _call_llm(prompt: str) -> str:
        r = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content

elif LLM_PROVIDER == "anthropic":
    import anthropic
    _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _call_llm(prompt: str) -> str:
        r = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return r.content[0].text


class SOCTriageAgent:
    """
    Nhận output từ IDS v14 + SHAP + MITRE → LLM phân tích → trả về
    structured report để hiển thị trên dashboard hoặc export PDF.
    """

    SYSTEM_CONTEXT = """Bạn là một SOC Analyst AI chuyên nghiệp.
Nhiệm vụ: phân tích alert từ hệ thống IDS và đưa ra đánh giá rõ ràng.
Luôn trả lời bằng JSON hợp lệ theo schema được yêu cầu.
Ngôn ngữ: Tiếng Việt, ngắn gọn, súc tích, chuyên nghiệp."""

    def triage_alert(self, alert_data: dict) -> dict:
        """
        alert_data cần có:
            - hybrid_score    : float (0-1)
            - ae_score        : float — reconstruction error
            - max_prob        : float — classifier confidence
            - predicted_class : str — tên class
            - is_zeroday      : bool
            - shap_summary    : str — text từ SHAPExplainer
            - mitre_summary   : str — text từ MITREMapper
            - timestamp       : str
            - alert_id        : str
        """

        prompt = f"""
{self.SYSTEM_CONTEXT}

=== ALERT DATA ===
Alert ID    : {alert_data['alert_id']}
Timestamp   : {alert_data['timestamp']}
Hybrid Score: {alert_data['hybrid_score']:.4f} (ngưỡng: >0.5 = đáng ngờ)
AE Score    : {alert_data['ae_score']:.4f} (reconstruction error — cao = bất thường)
Classifier  : {alert_data['predicted_class']} (confidence: {alert_data['max_prob']:.2%})
Zero-Day    : {'CÓ ⚠️' if alert_data['is_zeroday'] else 'KHÔNG'}

=== SHAP EXPLANATION (top features bất thường) ===
{alert_data['shap_summary']}

=== MITRE ATT&CK MAPPING ===
{alert_data['mitre_summary']}

=== YÊU CẦU ===
Trả về JSON với schema sau (không thêm text ngoài JSON):
{{
  "severity": "CRITICAL | HIGH | MEDIUM | LOW",
  "verdict": "Mô tả ngắn gọn 1 câu về alert này là gì",
  "attack_summary": "Giải thích cụ thể tại sao đây là tấn công, dựa trên SHAP và MITRE",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "false_positive_risk": "HIGH | MEDIUM | LOW",
  "false_positive_reason": "Lý do có thể là false positive (nếu có)",
  "analyst_note": "Điều analyst cần kiểm tra thêm"
}}
"""

        raw_response = _call_llm(prompt)

        # Parse JSON — xử lý trường hợp LLM trả về markdown code block
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback nếu LLM không trả đúng JSON
            result = {
                "severity": "HIGH",
                "verdict": "Phân tích LLM thất bại — cần review thủ công",
                "attack_summary": raw_response[:500],
                "recommended_actions": ["Escalate to Tier 2 analyst"],
                "false_positive_risk": "UNKNOWN",
                "false_positive_reason": "N/A",
                "analyst_note": "LLM response parsing error"
            }

        result["alert_id"] = alert_data["alert_id"]
        result["timestamp"] = alert_data["timestamp"]
        result["hybrid_score"] = alert_data["hybrid_score"]
        return result

    def explain_to_analyst(self, question: str, alert_context: dict) -> str:
        """
        Chat interface: analyst hỏi bằng ngôn ngữ tự nhiên về alert cụ thể.
        Ví dụ: "Tại sao alert này lại bị đánh dấu là Critical?"
        """
        prompt = f"""
{self.SYSTEM_CONTEXT}

Context về alert hiện tại:
{json.dumps(alert_context, ensure_ascii=False, indent=2)}

Câu hỏi của analyst: {question}

Trả lời ngắn gọn, rõ ràng, dùng thuật ngữ SOC chuyên nghiệp.
"""
        return _call_llm(prompt)