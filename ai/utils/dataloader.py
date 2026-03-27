from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch


# Danh sách features được dùng để huấn luyện & phát hiện
# Thứ tự phải khớp với config['model']['feature_size']
DDOS_FEATURES = [
    'status',       # HTTP status code (200/4xx/5xx)
    'bytes',        # Kích thước response
    'method_cat',   # Method (GET=0, POST=1, ...)
    'url_length',   # Độ dài URL
    'hour',         # Giờ trong ngày
    'is_4xx',       # Lỗi client (1/0)
    'is_5xx',       # Lỗi server (1/0)
]


class LogDataLoader:
    """
    Load và tiền xử lý Apache access log cho mô hình TransformerVAE.

    Pipeline:
        CSV / DataFrame
            -> Trích đặc trưng DDoS
            -> Chuẩn hoá MinMax [0, 1]
            -> Sliding Window [batch, seq_length, feature_size]
            -> Tensor PyTorch
    """

    def __init__(self, data_path: str = None, seq_length: int = 10,
                 features: list = None):
        self.seq_length  = seq_length
        self.data_path   = data_path
        self.features    = features or DDOS_FEATURES
        self.raw_data    = None
        self.scaler      = MinMaxScaler()
        self.original_df_subset = None   # Dùng để map kết quả về IP/Time gốc

        if data_path:
            self.raw_data = pd.read_csv(data_path)

    # ── CHUẨN HOÁ ────────────────────────────────────────────────────────────
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Fit-transform nếu chưa fit, transform nếu đã fit."""
        try:
            # Thử transform (scaler đã được fit trước)
            return self.scaler.transform(data)
        except Exception:
            return self.scaler.fit_transform(data)

    def fit_scaler(self, data: np.ndarray):
        """Fit scaler trên tập train rồi dùng transform khi inference."""
        self.scaler.fit(data)

    # ── SLIDING WINDOW ───────────────────────────────────────────────────────
    def create_sliding_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Tạo chuỗi thời gian dạng [N, seq_length, feature_size].
        Ví dụ: seq_length=10 => mỗi cửa sổ gồm 10 requests liên tiếp.
        """
        n = len(data)
        if n < self.seq_length:
            return np.empty((0, self.seq_length, data.shape[1]), dtype=np.float32)

        windows = np.stack(
            [data[i: i + self.seq_length] for i in range(n - self.seq_length + 1)],
            axis=0
        )
        return windows.astype(np.float32)

    # ── TRÍCH XUẤT ĐẶC TRƯNG ────────────────────────────────────────────────
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tự động thêm các cột đặc trưng nếu chưa có.
        Tương thích cả với dữ liệu parse raw lẫn dữ liệu đã tổng hợp.
        """
        df = df.copy()

        # method_cat
        if 'method_cat' not in df.columns and 'method' in df.columns:
            method_map = {'GET': 0, 'POST': 1, 'HEAD': 2, 'PUT': 3,
                          'DELETE': 4, 'OPTIONS': 5, 'PATCH': 6}
            df['method_cat'] = df['method'].map(method_map).fillna(7).astype(int)

        # url_length
        if 'url_length' not in df.columns and 'url' in df.columns:
            df['url_length'] = df['url'].str.len()

        # hour
        if 'hour' not in df.columns and 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(
                df['timestamp'],
                format='%d/%b/%Y:%H:%M:%S %z',
                errors='coerce',
                utc=True
            ).dt.hour.fillna(0).astype(int)

        # is_4xx, is_5xx
        if 'is_4xx' not in df.columns and 'status' in df.columns:
            status = pd.to_numeric(df['status'], errors='coerce').fillna(0).astype(int)
            df['is_4xx'] = (status // 100 == 4).astype(int)
            df['is_5xx'] = (status // 100 == 5).astype(int)

        # Đảm bảo tất cả features tồn tại (gán 0 nếu thiếu)
        for col in self.features:
            if col not in df.columns:
                df[col] = 0

        return df

    # ── PIPELINE CHÍNH ───────────────────────────────────────────────────────
    def get_tensor_data(self, df: pd.DataFrame = None,
                        fit: bool = False) -> torch.Tensor | None:
        """
        Chuyển DataFrame / raw_data thành Tensor PyTorch.

        Args:
            df   : DataFrame nguồn (nếu None sẽ dùng self.raw_data)
            fit  : True khi gọi lần đầu (training) để fit scaler

        Returns:
            Tensor shape [N, seq_length, feature_size] hoặc None nếu rỗng
        """
        working_df = df if df is not None else self.raw_data
        if working_df is None or len(working_df) == 0:
            return None

        working_df = self._extract_features(working_df)

        # Lấy ma trận số
        numeric_data = working_df[self.features].fillna(0).values.astype(np.float32)

        # Chuẩn hoá
        if fit:
            self.fit_scaler(numeric_data)
        normalized = self.normalize(numeric_data)

        # Sliding window
        windows = self.create_sliding_windows(normalized)
        if len(windows) == 0:
            return None

        # Lưu metadata gốc (IP, timestamp) để map kết quả về sau
        if 'ip' in working_df.columns and 'timestamp' in working_df.columns:
            self.original_df_subset = (
                working_df[['ip', 'timestamp']]
                .iloc[self.seq_length - 1:]
                .reset_index(drop=True)
            )

        return torch.tensor(windows, dtype=torch.float32)
