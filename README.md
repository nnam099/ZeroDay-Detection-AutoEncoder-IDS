# Zero-Day Detection AutoEncoder IDS

Hệ thống phát hiện xâm nhập trên UNSW-NB15, gồm mô hình IDS, phát hiện bất thường/zero-day, dashboard SOC, SHAP explainability, MITRE mapping và LLM triage.

## Trạng Thái Hiện Tại

- `v14` là bản vận hành mặc định vì repo hiện có sẵn checkpoint:
  - `checkpoints/ids_v14_model.pth`
  - `checkpoints/ids_v14_pipeline.pkl`
- `v15` là bản thử nghiệm/nâng cấp với VAE, Attention và OOD ensemble. Muốn dùng v15 cần train/export artifact v15 trước.
- Dashboard có thể chọn version bằng biến môi trường `IDS_MODEL_VERSION`.

## Cấu Trúc Chính

```text
src/
  ids_v14_unswnb15.py      # pipeline train/evaluate/export v14
  ids_v15_unswnb15.py      # pipeline train/evaluate/export v15
  explainer.py             # SHAP explainer
  mitre_mapper.py          # UNSW class -> MITRE ATT&CK heuristic mapping
  llm_agent.py             # SOC triage LLM wrapper
dashboard/
  app.py                   # Streamlit SOC dashboard
configs/
  config_default.yaml      # config v15
scripts/
  train.sh                 # train v14
  train_v15.sh             # train v15
data/                      # UNSW-NB15 CSV files, gitignored
checkpoints/               # model/pipeline artifacts, gitignored
results/                   # metrics summary
```

## Cài Đặt

```bash
pip install -r requirements.txt
```

Nếu chạy v15 với YAML config, cần `pyyaml`. File `requirements.txt` đã khai báo dependency này.

## Chạy Dashboard

Mặc định dùng v14:

```bash
streamlit run dashboard/app.py
```

Chọn version hoặc đường dẫn artifact:

```bash
set IDS_MODEL_VERSION=v14
set IDS_MODEL_PATH=checkpoints/ids_v14_model.pth
set IDS_PIPELINE_PATH=checkpoints/ids_v14_pipeline.pkl
streamlit run dashboard/app.py
```

Với PowerShell:

```powershell
$env:IDS_MODEL_VERSION="v14"
streamlit run dashboard/app.py
```

Các biến môi trường hỗ trợ:

- `IDS_MODEL_VERSION`: `v14` hoặc `v15`
- `IDS_MODEL_PATH`: đường dẫn file `.pth`
- `IDS_PIPELINE_PATH`: đường dẫn file `.pkl`
- `IDS_DATA_DIR`: thư mục data
- `IDS_SAMPLE_DATA_PATH`: CSV dùng để lấy sample trong dashboard
- `LLM_PROVIDER`: `groq`, `gemini`, `openai`, hoặc `anthropic`

## Train

Train v14:

```bash
python src/ids_v14_unswnb15.py --data_dir data/ --save_dir checkpoints/ --plot_dir plots/
```

Train v15:

```bash
python src/ids_v15_unswnb15.py --data_dir data/ --save_dir checkpoints/ --plot_dir plots/
```

Sau khi train, pipeline sẽ lưu thêm metadata phục vụ inference thực tế, gồm feature list, scaler, label encoder, categorical mappings, thresholds và version.

## Ghi Chú Vận Hành

- Không commit `.env`, `data/`, `checkpoints/` hoặc file `.pyc`.
- Dashboard sẽ vào demo mode nếu thiếu model hoặc pipeline.
- MITRE mapping hiện là heuristic, phù hợp demo/triage sơ bộ, chưa thay thế phân tích SOC thủ công.
- LLM chỉ dùng để hỗ trợ diễn giải; quyết định xử lý incident vẫn cần analyst xác nhận.
