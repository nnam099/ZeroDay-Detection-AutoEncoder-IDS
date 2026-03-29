import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from elasticsearch import Elasticsearch

# 1. Lấy dữ liệu lịch sử từ Elasticsearch
es = Elasticsearch("http://elasticsearch:9200")
res = es.search(index="pfsense-logs-*", size=10000, body={"query": {"match_all": {}}})
data = [hit["_source"] for hit in res["hits"]["hits"]]
df = pd.DataFrame(data)

# 2. Feature Engineering (Chỉ lấy các đặc trưng số học)
# Giả sử trong log đã có length, protocol_id. Ta điền 0 nếu thiếu (fillna).
df['length'] = pd.to_numeric(df['length'], errors='coerce').fillna(0)
df['protocol_id'] = pd.to_numeric(df['protocol_id'], errors='coerce').fillna(0)

# Chuyển đổi nhãn (action: pass -> 0, block -> 1) để làm nhãn DDoS
df['label'] = df['action'].apply(lambda x: 1 if x == 'block' else 0)

# Biến đổi thành Tensor cho PyTorch
X = torch.tensor(df[['length', 'protocol_id']].values, dtype=torch.float32)
y = torch.tensor(df['label'].values, dtype=torch.float32).view(-1, 1)

# 3. Định nghĩa mô hình Deep Learning (Placeholder cho Transformer/Mạng Neural)
class DDoSDetector(nn.Module):
    def __init__(self):
        super(DDoSDetector, self).__init__()
        # Kiến trúc ví dụ, bạn có thể thay bằng khối nn.Transformer cho chuỗi thời gian
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = DDoSDetector()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Vòng lặp huấn luyện (Training Loop)
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. Lưu mô hình định dạng PyTorch
torch.save(model.state_dict(), "model.pth")
print("Đã lưu mô hình tại model.pth")