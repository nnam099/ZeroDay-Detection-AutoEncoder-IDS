# Zero-Day Detection AutoEncoder IDS

Hệ thống phát hiện xâm nhập mạng dựa trên học sâu cho bộ dữ liệu UNSW-NB15. Project kết hợp phân loại tấn công đã biết, phát hiện bất thường/zero-day bằng reconstruction error, dashboard SOC để phân tích alert, SHAP explainability, MITRE ATT&CK mapping và LLM triage.

> Mục tiêu của project là nghiên cứu, demo và hỗ trợ phân tích SOC ở mức prototype. Đây không phải IDS production-ready và không thay thế quy trình điều tra thủ công của analyst.

## Tổng Quan

Project tập trung vào bài toán phát hiện traffic bất thường khi mô hình chỉ được huấn luyện trên một nhóm lớp tấn công đã biết. Các lớp còn lại được giữ lại để mô phỏng zero-day/OOD traffic.

Luồng chính:

1. Tiền xử lý dữ liệu UNSW-NB15 và chuẩn hóa feature.
2. Huấn luyện mô hình hybrid gồm supervised classifier, contrastive representation và autoencoder/VAE.
3. Hiệu chỉnh ngưỡng phát hiện zero-day trên validation set.
4. Lưu artifact gồm model weights, scaler, label encoder, feature list và thresholds.
5. Dashboard Streamlit dùng artifact để phân tích single alert hoặc CSV batch.
6. SHAP, MITRE mapping và LLM được dùng như lớp hỗ trợ giải thích, không phải nguồn quyết định cuối cùng.

## Tính Năng Chính

- **Known-attack classification**: phân loại các lớp đã biết như `Normal`, `DoS`, `Exploits`, `Reconnaissance`, `Generic`.
- **Zero-day/OOD detection**: dùng reconstruction error, confidence score và hybrid threshold để đánh dấu traffic bất thường.
- **SOC dashboard**: giao diện Streamlit cho phân tích alert, upload CSV, xem score, verdict, risk và lịch sử alert.
- **Real-world CSV normalization**: hỗ trợ chuẩn hóa một số CSV flow/firewall/Zeek/Suricata về schema gần UNSW-NB15.
- **Explainability**: SHAP top features giúp analyst xem yếu tố nào ảnh hưởng đến alert.
- **MITRE ATT&CK mapping**: ánh xạ heuristic từ class/feature sang kỹ thuật ATT&CK để hỗ trợ triage.
- **LLM triage**: tích hợp tùy chọn với Groq, Gemini, OpenAI hoặc Anthropic để tạo nhận định dạng SOC.
- **Smoke tests**: kiểm tra nhanh normalizer, MITRE mapper và khả năng load artifact v14.

## Kiến Trúc Mô Hình

### v14 - Bản vận hành mặc định

`v14` là phiên bản mặc định vì repo hiện có sẵn artifact tương ứng trong `checkpoints/`.

Thành phần chính:

- `IDSBackbone`: Linear projection, LayerNorm, GELU và residual blocks.
- `classifier`: supervised head cho known classes.
- `proj_head`: projection head phục vụ contrastive representation.
- `autoencoder`: học tái tạo feature để tính anomaly/reconstruction error.
- `hybrid detector`: kết hợp classifier confidence và autoencoder score để phát hiện zero-day.

Loss tổng:

```text
FocalLoss + lambda_con * SupConLoss + lambda_ae * AE_MSE
```

### v15 - Bản thử nghiệm

`v15` mở rộng v14 với VAE, Attention Gate, KNN/OOD ensemble và YAML config. Muốn dùng dashboard với v15 cần train/export artifact v15 trước.

## Cấu Trúc Thư Mục

```text
src/
  ids_v14_unswnb15.py      # train/evaluate/export pipeline v14
  ids_v15_unswnb15.py      # train/evaluate/export pipeline v15
  explainer.py             # SHAP explainer
  mitre_mapper.py          # heuristic MITRE ATT&CK mapping
  log_normalizer.py        # normalize CSV log thực tế về schema flow gần UNSW
  llm_agent.py             # wrapper cho các LLM provider
dashboard/
  app.py                   # Streamlit SOC dashboard
configs/
  config_default.yaml      # cấu hình mặc định cho v15
docs/
  architecture.md          # ghi chú kiến trúc
  real_world_csv.md        # hướng dẫn CSV log thực tế
  project_audit.md         # audit kỹ thuật hiện tại
tests/
  test_smoke.py            # smoke tests
scripts/
  train.sh                 # train v14
  train_v15.sh             # train v15
data/                      # dữ liệu local, không commit
checkpoints/               # model/pipeline artifacts, không commit
plots/                     # biểu đồ đánh giá
results/                   # metric summary
```

## Cài Đặt

Yêu cầu khuyến nghị:

- Python 3.9+
- PyTorch
- scikit-learn, pandas, numpy, matplotlib
- Streamlit nếu chạy dashboard
- SHAP và LLM provider SDK nếu dùng các tính năng tùy chọn

Cài dependencies:

```bash
pip install -r requirements.txt
```

Nếu dùng Windows PowerShell với virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dữ Liệu

Project dùng UNSW-NB15. Đặt các file CSV trong thư mục `data/`, ví dụ:

```text
data/
  UNSW-NB15_1.csv
  UNSW-NB15_2.csv
  UNSW-NB15_3.csv
  UNSW-NB15_4.csv
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv
```

`data/` được gitignore vì kích thước lớn và thường chứa dữ liệu local.

Dashboard cũng hỗ trợ upload một số CSV flow/firewall phổ biến. Xem thêm [docs/real_world_csv.md](docs/real_world_csv.md).

## Chạy Dashboard

Mặc định dashboard dùng `v14`:

```bash
streamlit run dashboard/app.py
```

PowerShell:

```powershell
$env:IDS_MODEL_VERSION="v14"
streamlit run dashboard/app.py
```

Các biến môi trường quan trọng:

| Biến | Ý nghĩa |
|------|---------|
| `IDS_MODEL_VERSION` | `v14` hoặc `v15` |
| `IDS_MODEL_PATH` | Đường dẫn file `.pth` |
| `IDS_PIPELINE_PATH` | Đường dẫn file `.pkl` |
| `IDS_DATA_DIR` | Thư mục chứa dữ liệu |
| `IDS_SAMPLE_DATA_PATH` | CSV mẫu dùng trong dashboard |
| `LLM_PROVIDER` | `groq`, `gemini`, `openai` hoặc `anthropic` |

Nếu thiếu model hoặc pipeline, dashboard sẽ chuyển sang demo mode.

## Huấn Luyện

Train v14:

```bash
python src/ids_v14_unswnb15.py --data_dir data/ --save_dir checkpoints/ --plot_dir plots/
```

Train v15:

```bash
python src/ids_v15_unswnb15.py --data_dir data/ --save_dir checkpoints/ --plot_dir plots/
```

Train v15 với config YAML:

```bash
python src/ids_v15_unswnb15.py --config configs/config_default.yaml
```

Sau khi train, project lưu:

- model weights `.pth`
- pipeline `.pkl` gồm scaler, label encoder, feature names, categorical maps và thresholds
- plots đánh giá trong `plots/`
- metric summary trong `results/`

## Kiểm Tra Nhanh

```bash
python -m compileall src dashboard export_model.py patch_checkpoint.py tests
python -m unittest discover -s tests
```

Hoặc chạy gộp:

```bash
python scripts/smoke_check.py
```

Smoke tests sẽ skip phần load artifact nếu checkpoint/pipeline không tồn tại trên máy hiện tại.

## LLM Triage

LLM là tùy chọn. Tạo file `.env` ở thư mục gốc và khai báo provider/key tương ứng:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key
```

Provider hỗ trợ trong code:

- `groq`
- `gemini`
- `openai`
- `anthropic`

LLM output chỉ dùng để hỗ trợ giải thích alert. Analyst vẫn cần xác minh bằng log, endpoint telemetry, SIEM/firewall evidence và context vận hành thực tế.

## Giới Hạn

- Mapping MITRE hiện là heuristic, chưa phải threat intelligence đầy đủ.
- Kết quả zero-day phụ thuộc mạnh vào chất lượng feature, scaler và threshold calibration.
- CSV thực tế được normalizer chuyển đổi xấp xỉ về schema UNSW; nếu thiếu directional counters hoặc timing fields, độ tin cậy inference sẽ thấp hơn.
- Artifact v14 là bản vận hành mặc định; v15 cần được train/export riêng trước khi dùng ổn định.
- Project chưa bao gồm CI/CD, deployment hardening, authentication dashboard hoặc realtime packet capture.

## Bảo Mật Và Git Hygiene

Không commit các file/thư mục sau:

- `.env`
- `data/`
- `checkpoints/`
- `.venv/`
- `__pycache__/`
- file `.pyc`

## Trạng Thái Hiện Tại

- Branch chính đã có dashboard chạy được với v14 artifact.
- Smoke tests hiện pass cho normalizer, MITRE mapper và v14 artifact loading.
- Dashboard đã cập nhật API Streamlit mới: dùng `width="stretch"` thay cho `use_container_width`.

## License

Xem [LICENSE](LICENSE).
