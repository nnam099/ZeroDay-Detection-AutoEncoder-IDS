from mitreattack.stix20 import MitreAttackData
import json

# Mapping thủ công từ attack class UNSW-NB15 → MITRE Techniques
# Dựa trên đặc điểm của từng loại tấn công trong dataset
UNSW_TO_MITRE = {
    "DoS": [
        {"id": "T1498", "name": "Network Denial of Service",     "tactic": "Impact"},
        {"id": "T1499", "name": "Endpoint Denial of Service",    "tactic": "Impact"},
    ],
    "Exploits": [
        {"id": "T1190", "name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
        {"id": "T1203", "name": "Exploitation for Client Execution", "tactic": "Execution"},
        {"id": "T1068", "name": "Exploitation for Privilege Escalation", "tactic": "Privilege Escalation"},
    ],
    "Reconnaissance": [
        {"id": "T1595", "name": "Active Scanning",               "tactic": "Reconnaissance"},
        {"id": "T1592", "name": "Gather Victim Host Information", "tactic": "Reconnaissance"},
        {"id": "T1046", "name": "Network Service Discovery",      "tactic": "Discovery"},
    ],
    "Generic": [
        {"id": "T1071", "name": "Application Layer Protocol",    "tactic": "Command and Control"},
    ],
    "Backdoors": [
        {"id": "T1543", "name": "Create or Modify System Process", "tactic": "Persistence"},
        {"id": "T1078", "name": "Valid Accounts",                 "tactic": "Defense Evasion"},
    ],
    "Worms": [
        {"id": "T1091", "name": "Replication Through Removable Media", "tactic": "Lateral Movement"},
        {"id": "T1210", "name": "Exploitation of Remote Services",     "tactic": "Lateral Movement"},
    ],
    "Shellcode": [
        {"id": "T1055", "name": "Process Injection",             "tactic": "Defense Evasion"},
        {"id": "T1059", "name": "Command and Scripting Interpreter", "tactic": "Execution"},
    ],
    "Fuzzers": [
        {"id": "T1110", "name": "Brute Force",                   "tactic": "Credential Access"},
        {"id": "T1595", "name": "Active Scanning",               "tactic": "Reconnaissance"},
    ],
    "Analysis": [
        {"id": "T1040", "name": "Network Sniffing",              "tactic": "Credential Access"},
        {"id": "T1046", "name": "Network Service Discovery",     "tactic": "Discovery"},
    ],
    "Zero-Day": [
        # Zero-day chưa rõ technique — LLM sẽ suy luận thêm
        {"id": "T1190", "name": "Exploit Public-Facing Application", "tactic": "Initial Access"},
    ],
}

CLASS_NAMES = ["Normal", "DoS", "Exploits", "Reconnaissance", "Generic"]

class MITREMapper:
    def __init__(self):
        self.mapping = UNSW_TO_MITRE

    def map_known_attack(self, class_idx):
        """
        Nhận class index từ classifier → trả về MITRE techniques.
        class_idx: int (0=Normal, 1=DoS, 2=Exploits, 3=Recon, 4=Generic)
        """
        if class_idx == 0:
            return None  # Normal traffic, không cần map

        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Generic"
        techniques = self.mapping.get(class_name, [])

        return {
            "attack_class": class_name,
            "techniques": techniques,
            "mitre_url": f"https://attack.mitre.org/techniques/{techniques[0]['id']}/" if techniques else None
        }

    def map_zeroday(self, ae_score, top_shap_features):
        """
        Zero-day không có label rõ ràng.
        Dùng heuristic từ SHAP features để gợi ý technique có thể liên quan.

        ae_score         : float — reconstruction error (càng cao càng lạ)
        top_shap_features: list of (feature_name, shap_value, feature_value)
        """
        feature_names_flagged = [f[0] for f in top_shap_features[:5]]

        # Heuristic rules dựa trên feature patterns của UNSW-NB15
        suspected = []

        # Nếu bytes-related features bất thường → có thể data exfil hoặc DoS
        if any(f in feature_names_flagged for f in ['sbytes', 'dbytes', 'sload', 'dload']):
            suspected.append({"id": "T1030", "name": "Data Transfer Size Limits", "tactic": "Exfiltration"})
            suspected.append({"id": "T1498", "name": "Network Denial of Service", "tactic": "Impact"})

        # Nếu connection count bất thường → scanning hoặc C2
        if any(f in feature_names_flagged for f in ['ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_srv_src']):
            suspected.append({"id": "T1046", "name": "Network Service Discovery", "tactic": "Discovery"})
            suspected.append({"id": "T1071", "name": "Application Layer Protocol", "tactic": "C2"})

        # Nếu duration bất thường → beaconing (C2)
        if 'dur' in feature_names_flagged:
            suspected.append({"id": "T1132", "name": "Data Encoding", "tactic": "Command and Control"})

        return {
            "attack_class": "Zero-Day",
            "ae_reconstruction_error": ae_score,
            "suspected_techniques": suspected if suspected else self.mapping["Zero-Day"],
            "confidence": "low — requires analyst review",
        }

    def format_for_llm(self, mitre_result):
        """Chuyển MITRE result thành text để nhét vào LLM prompt."""
        if not mitre_result:
            return "Traffic bình thường, không phát hiện tấn công."

        lines = [f"Attack class: {mitre_result.get('attack_class', 'Unknown')}"]

        techniques = mitre_result.get('techniques') or mitre_result.get('suspected_techniques', [])
        if techniques:
            lines.append("MITRE ATT&CK techniques liên quan:")
            for t in techniques:
                lines.append(f"  - [{t['id']}] {t['name']} (Tactic: {t['tactic']})")

        if 'confidence' in mitre_result:
            lines.append(f"Confidence: {mitre_result['confidence']}")

        return "\n".join(lines)