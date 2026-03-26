import numpy as np

def calculate_dynamic_thresholds(train_errors, uncertainties):
    # Ngưỡng RE: Mean + 3*Std (Quy tắc 3-sigma bao phủ 99.7% dữ liệu bình thường)
    re_threshold = np.mean(train_errors) + 3 * np.std(train_errors)
    
    # Ngưỡng Uncertainty: Lấy phân vị thứ 95 (Top 5% bối rối nhất của mẫu sạch)
    u_threshold = np.percentile(uncertainties, 95)
    
    return re_threshold, u_threshold