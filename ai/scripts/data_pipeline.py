import re
import os
import pandas as pd
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Regex cho Apache Combined Log Format:
#   IP  ident  user  [timestamp]  "METHOD URL PROTOCOL"  status  bytes  "referrer"  "user_agent"
# ────────────────────────────────────────────────────────────────────────────
_LOG_PATTERN = re.compile(
    r'(?P<ip>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<url>\S+)\s+(?P<protocol>[^"]+)"\s+'
    r'(?P<status>\d{3})\s+'
    r'(?P<bytes>\S+)\s+'
    r'"(?P<referrer>[^"]*)"\s+'
    r'"(?P<user_agent>[^"]*)"'
)


def parse_apache_log(log_path: str, output_csv: str) -> bool:
    """
    Parse Apache Combined Log Format -> CSV.
    Trả về True nếu thành công, False nếu file không tồn tại hoặc rỗng.
    """
    if not os.path.exists(log_path):
        print(f"[WARN] Không tìm thấy file log: {log_path}")
        return False

    records = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = _LOG_PATTERN.match(line.strip())
            if m:
                records.append(m.groupdict())

    if not records:
        print(f"[WARN] Không parse được dòng nào từ: {log_path}")
        return False

    df = pd.DataFrame(records)

    # ── Ép kiểu số ──────────────────────────────────────────────────────────
    df['status'] = pd.to_numeric(df['status'], errors='coerce').fillna(0).astype(int)
    df['bytes']  = pd.to_numeric(df['bytes'].replace('-', '0'), errors='coerce').fillna(0).astype(int)

    # ── Parse timestamp ──────────────────────────────────────────────────────
    # Định dạng Apache: 26/Mar/2026:14:23:11 +0700
    df['datetime'] = pd.to_datetime(
        df['timestamp'],
        format='%d/%b/%Y:%H:%M:%S %z',
        errors='coerce',
        utc=True
    )

    # ── Thêm đặc trưng hỗ trợ phát hiện DDoS ────────────────────────────────
    # 1. method_cat: mã hoá HTTP method (GET=0, POST=1, HEAD=2, ...)
    method_map = {'GET': 0, 'POST': 1, 'HEAD': 2, 'PUT': 3,
                  'DELETE': 4, 'OPTIONS': 5, 'PATCH': 6}
    df['method_cat'] = df['method'].map(method_map).fillna(7).astype(int)

    # 2. url_length: độ dài URL (DDoS thường dùng URL ngắn, lặp đi lặp lại)
    df['url_length'] = df['url'].str.len()

    # 3. hour: giờ trong ngày (DDoS thường xảy ra ngoài giờ hành chính)
    df['hour'] = df['datetime'].dt.hour.fillna(0).astype(int)

    # 4. is_4xx / is_5xx: tỷ lệ lỗi cao là dấu hiệu tấn công
    df['is_4xx'] = (df['status'] // 100 == 4).astype(int)
    df['is_5xx'] = (df['status'] // 100 == 5).astype(int)

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Đã parse {len(df)} dòng log -> {output_csv}")
    return True


def aggregate_by_ip_window(df: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
    """
    Gộp log theo từng IP trong cửa sổ thời gian `window_seconds`.
    Tạo ra các đặc trưng hành vi (behavioral features) mạnh hơn cho DDoS.

    Đặc trưng đầu ra:
        - req_count    : số request trong cửa sổ (DDoS có req_count rất cao)
        - avg_bytes    : trung bình bytes/request
        - error_ratio  : tỷ lệ request lỗi (4xx+5xx)
        - unique_urls  : số URL duy nhất (DDoS thường lặp 1-2 URL)
        - req_per_sec  : tốc độ request/giây
    """
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'],
                                        format='%d/%b/%Y:%H:%M:%S %z',
                                        errors='coerce', utc=True)

    df = df.dropna(subset=['datetime']).copy()
    df['window'] = (df['datetime'].astype(np.int64) // 1e9 // window_seconds).astype(int)

    agg = df.groupby(['ip', 'window']).agg(
        req_count   =('status',   'count'),
        avg_bytes   =('bytes',    'mean'),
        error_ratio =('is_4xx',   'mean'),   # is_4xx xấp xỉ tỷ lệ lỗi
        unique_urls =('url',      'nunique'),
        hour        =('hour',     'first'),
        timestamp   =('timestamp','first'),
    ).reset_index()

    # req_per_sec: chuẩn hoá req_count theo khoảng cửa sổ
    agg['req_per_sec'] = agg['req_count'] / window_seconds

    return agg