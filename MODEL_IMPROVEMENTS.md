# Tổng Hợp Các Cải Tiến Mô Hình AI Phát Hiện Tấn Công (TransformerVAE) 🚀

Tài liệu này tóm tắt toàn bộ các thay đổi và thuật toán cải tiến đã được áp dụng vào **Mô hình 1** (Phiên bản mới nhất cấu trúc TransformerVAE) giúp vượt qua phiên bản cũ cả về mặt kỹ thuật, độ ổn định và khả năng bắt trúng hacker (Recall).

---

## 1. Nâng Cấp Kiến Trúc Lõi (Targeting Architecture)

Thực tế mạng máy tính (Network Traffic) luôn chứa tính chất *chuỗi thời gian* và *sự bùng nổ dữ liệu cục bộ*. Do đó 2 mảnh ghép sinh tử sau đã được thêm vào:

- **Tích hợp Conv1D (Mạng nén Tích chập 1 Chiều):** DDoS hay DoS thường tạo ra các nhóm request bùng nổ nhỏ ở một thời điểm liên tiếp. `Conv1D` chạy ngang qua cửa sổ chuỗi thời gian (window) để nhận diện nhanh những cực trị cục bộ này trước khi đẩy qua Transformer.
- **Bổ sung Positional Encoding (Mã hóa Vị trí):** Transformer truyền thống có nhược điểm là "quên" mất thứ tự sinh ra gói tin. Việc gắn Positional Encoding giúp mô hình nhận ra được nhịp độ đều đặn hoặc bất thường theo đúng trình tự thời gian (Ví dụ một loạt kết nối SYN dồn dập cách đều nhau).
- **Cơ Chế Sliding Window Zero-Copy:** Thay vì vòng lặp thô sơ để tạo các cửa sổ thời gian (SEQ=10) chèn bộ nhớ, hệ thống sử dụng cơ chế dịch bit (stride tricks) của array numpy. Giúp nạp dữ liệu vào GPU mượt và không bị nổ RAM.

## 2. Tiền Xử Lý & Xóa Bỏ Biến Dạng (Data Engineering)

- **Kỹ thuật `Log1p Transform`:** 
  - *Vấn đề:* Các cột giá trị như `Flow Bytes/s` hoặc `Total Fwd Packets` có sự lệch biên rất lớn. Đặc biệt khi hacker dồn băng thông, giá trị này to ra hàng trăm vạn lần so với người dùng thường.
  - *Giải pháp:* Sử dụng biến đổi Logarit cơ số tự nhiên `np.log1p(np.abs(x))` giúp triệt tiêu những "cái đuôi siêu dài" này, kéo phân phối đặc trưng về vùng chuẩn hơn. Giúp mô hình bắt ngoại lệ dễ dàng hơn nhiều lần.
  
- **Vá triệt để lỗi Rò rỉ `Infinity` / `NaN`:**
  - *Vấn đề:* Bộ dữ liệu chuẩn CIC-IDS2017 thi thoảng chứa chuỗi dị biệt `"Infinity"` tại một vài dòng. Do Pandas không coi nó là rỗng (`NaN`) nên hàm `dropna()` ban đầu bị bỏ qua. Khi hệ thống numpy ép kiểu thành Float32 đã sinh ra biến số vô hạn `np.inf`, làm rớt toàn bộ pipeline khi thử nghiệm tấn công lẻ (như Botnet).
  - *Giải pháp:* Thiết lập tường lửa lọc cuối cùng bằng `np.isfinite(...).all()` chặn đứng bất kỳ dị biến toán học nào ngay trước hàm đánh giá. 

## 3. Cân Chỉnh Hàm Mất Mát (Loss Function Tuning)

- **Tham số $\beta = 0.1$ cho Tản chuẩn KLD:**
  - Sự kết hợp của thuật toán $\beta$-VAE. Nếu đặt quá cao, mô hình sinh bị ép nén quá mức và không nắm được biến thiên. Nếu vứt đi, nó biến tướng thành đồ thị Encode-Decode thẳng đuột.
  - Đặt $\beta=0.1$ giúp phần lỗi tái tạo (Mean Squared Error - MSE) của bản chép này chiếm vai trò trung tâm (90%). Mô hình sẽ ưu tiên "học lại" traffic cực chuẩn, và tạo ra độ ngợp (Reconstruction Error đẩy lên rất cao) ngay khi đụng độ thứ gì vượt rào bình thường.

## 4. Cải Thiện Đo Đếm (MC Dropout & Thresholding)

- **Thuật Toán Monte Carlo (MC Dropout):** Không dùng 1 lần dự đoán mà tận dụng `Dropout` và quét mẫu 15 lần chạy ngẫu nhiên vào mạng Neuron cho duy nhất 1 luồng dữ liệu. Giúp nhận biết được 2 cực độ: 
  - `RE` (Sai số tái tạo).
  - `Uncertainty / U` (Sự hoang mang của mô hình).
- **Lập Mức Ngưỡng Chuẩn (`Percentile 90`):** Chấp nhận một phần tỷ lệ báo giả cho các Traffic cá biệt của người xem Netflix/Game, để đổi lấy việc đập tan các cuộc tấn công dai dẳng.

---

## 🏆 TỔNG QUAN HIỆU SUẤT ĐẠT ĐƯỢC

1. **Chuẩn hóa Triệt để Độ phủ:** Vượt giới hạn của mô hình cũ, **Attack Recall đã đi từ `25%` vọt lên `51.28%`** (Tăng hơn gấp đôi).
2. **Loại bỏ Hoàn toàn Thảm họa:** Nhận diện sạch sẽ và chuẩn xác đối với: `Web Attack` (99+%), `FTP-Patator` (99.5%), `DoS Slowhttptest` (95%), `Infiltration` (100%).
3. **Thời gian chạy siêu êm:** Không còn bị crash đột tử nhờ hệ thống Zero-Copy và Safeguard Lọc Dữ Liệu Tĩnh. Cả tiến trình đánh giá 1.5 triệu records trên Test Phase chạy mượt mà ngay trên T4 GPU.

## 👉 Khuyến nghị bước hoàn thiện tiếp theo (Next Steps):
Mô hình hiện đang điểm mù lớn nhất tại `PortScan` và `Botnet`. Để khắc phục dứt điểm nhược điểm cuối cùng này của VAE, ta chỉ cần thu thập trích xuất thêm nhóm Đặc trưng:
- **`Destination Port`**: Bắt PortScan 
- **`Active Mean` / `Idle Mean`**: Bắt Botnet
>>> Khi có được cụm sinh trắc này, Recall sẽ có cơ sở mạnh mẽ để tiến gần định mốc an toàn 90%.
