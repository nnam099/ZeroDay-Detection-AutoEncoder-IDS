# Project Audit

Ngày cập nhật: 2026-05-13

## Đã xác minh

- Python compile pass cho `src/`, `dashboard/`, `export_model.py`, `patch_checkpoint.py` và `tests/`.
- Checkpoint `ids_v14_model.pth` load được vào `IDSModel` v14, không có missing/unexpected weights.
- Pipeline v14 có 61 features, khớp `n_features` trong checkpoint.
- `log_normalizer.py` có thể map CSV firewall/flow có cột phổ biến sang schema flow gần UNSW.
- Dashboard hiện validate contract giữa checkpoint và pipeline trước khi inference.
- Có `scripts/smoke_check.py` để chạy compile + unittest bằng một lệnh.

## Điểm mạnh

- Có tách rõ train/inference/dashboard theo module.
- Artifact v14 lưu đủ metadata cần thiết: scaler, label encoder, feature list, thresholds.
- Dashboard có fallback demo mode khi thiếu artifact và hỗ trợ CSV log thực tế qua normalizer.
- MITRE mapping được đóng gói riêng, dễ mở rộng rule/evidence sau này.

## Rủi ro hiện tại

- Nhiều file code đang có comment/docstring hiển thị mojibake trong một số terminal Windows, gây khó đọc khi báo cáo hoặc bảo trì.
- Đã có artifact contract validator cơ bản cho checkpoint/pipeline; vẫn cần mở rộng sang validation dữ liệu upload theo từng schema.
- `v15` là nhánh thử nghiệm nhưng dashboard mặc định v14; nếu chọn v15 khi chưa train artifact sẽ fallback/demo.
- LLM provider là optional nhưng dependency provider chưa nằm trong requirements mặc định; cần cài đúng thư viện theo provider.

## Ưu tiên tiếp theo

1. Sửa mojibake trong comment/docstring chính nếu cần trình bày code trong báo cáo.
2. Tách inference logic khỏi `dashboard/app.py` sang module riêng để test được mà không phụ thuộc Streamlit.
3. Thêm validation dữ liệu upload: cảnh báo coverage thấp, thiếu directional counters hoặc schema không rõ.
4. Thêm CI job chạy `scripts/smoke_check.py`.
5. Nếu tiếp tục v15, train/export artifact v15 riêng và thêm smoke test load v15 tương tự v14.
