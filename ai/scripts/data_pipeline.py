import pandas as pd
import re
import os

def parse_apache_log(log_path, output_csv):
    # Regex chuẩn cho định dạng Apache Common Log
    regex = r'([^ ]*) ([^ ]*) ([^ ]*) \[(.*?)\] "([^ ]*) ([^ ]*) ([^ ]*)" ([0-9]*) ([0-9]*) "(.*?)" "(.*?)"'
    
    logs = []
    if not os.path.exists(log_path):
        return False

    with open(log_path, 'r') as f:
        for line in f:
            match = re.match(regex, line)
            if match:
                logs.append(match.groups())

    columns = ['ip', 'identity', 'user', 'timestamp', 'method', 'url', 'protocol', 'status', 'bytes', 'referrer', 'user_agent']
    df = pd.DataFrame(logs, columns=columns)
    
    # Ép kiểu dữ liệu số
    df['status'] = pd.to_numeric(df['status'], errors='coerce')
    df['bytes'] = pd.to_numeric(df['bytes'], errors='coerce')
    
    df.to_csv(output_csv, index=False)
    return True