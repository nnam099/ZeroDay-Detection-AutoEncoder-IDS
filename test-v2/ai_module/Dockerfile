# Sử dụng image Python chuẩn, phiên bản slim để nhẹ máy
FROM python:3.9-slim

# Cài đặt một số công cụ mạng cơ bản để hỗ trợ paramiko SSH
RUN apt-get update && apt-get install -y iputils-ping && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc trong container
WORKDIR /app

# Copy file requirements vào trước để cài thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code và thư mục model/ vào container
COPY . .

# Chạy lệnh khởi động não bộ AI
CMD ["python", "-u", "main.py"]