# chuẩn hóa dữ liệu về khoảng [0,1]

# chia nhỏ các chuỗi log bằng sliding windowclass LogDataLoader:

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch

class LogDataLoader:
    def __init__(self, data_path, seq_length=10):
        # load dữ liệu từ CSV hoặc Elasticsearch JSON
        # Dữ liệu mạng thường là các con số (packet size, rate, entropy)

        self.seq_length = seq_length
        self.data_path = data_path
        self.raw_data = None
        self.scaler = MinMaxScaler()
        self.original_df_subset = None

        if data_path:
          self.raw_data = pd.read_csv(data_path) # gọi API từ Elasticsearch thay vì pd.read_csv

    def normalize(self, data):
        return self.scaler.fit_transform(data)

    def create_sliding_windows(self, data):
        # mảng 1D thành các windows (ví dụ: [Batch, seq_length, Features])
        # để Transformer hiểu được "ngữ cảnh" theo thời gian

        windows = []
        for i in range(len(data) - self.seq_length + 1):
          window = data[i : i + self.seq_length]
          windows.append(window)

        return np.array(windows)

    def get_tensor_data(self, df=None):
      working_df = df if df is not None else self.raw_data

      if 'method' in working_df.columns:
        working_df['method_cat'] = working_df['method'].astype('category').cat.codes

      if 'url' in working_df.columns:
        working_df['url_length'] = working_df['url'].apply(len)

      if 'timestamp' in working_df.columns:
        # Định dạng timestamp Apache: [11/Mar/2026:19:10:42 +0700]
        working_df['hour'] = pd.to_datetime(
            working_df['timestamp'],
            format='%d/%b/%Y:%H:%M:%S %z',
            exact=False
        ).dt.hour

      selected_features = ['status', 'bytes', 'method_cat', 'url_length', 'hour']

      for col in selected_features:
        if col not in working_df.columns:
          working_df[col] = 0

      numeric_data = working_df[selected_features].fillna(0).values

      normalized_data = self.normalize(numeric_data)

      sliding_windows = self.create_sliding_windows(normalized_data)
      # ĐỒNG BỘ HÓA: Lưu lại thông tin gốc (IP, Time) khớp với số lượng cửa sổ
      # Do dùng sliding window, dòng i của Tensor sẽ ứng với thông tin ở dòng i + seq_length - 1
      self.original_df_subset = working_df[['ip', 'timestamp']].iloc[self.seq_length -1:].reset_index(drop=True)

      return torch.tensor(sliding_windows, dtype=torch.float32)
