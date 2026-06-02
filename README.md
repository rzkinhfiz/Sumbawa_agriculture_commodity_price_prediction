# 🌾 Prediksi Harga Komoditas Pertanian Kabupaten Sumbawa

**Status**: ✅ Production Ready | **Last Updated**: June 2026 | **Forecast Range**: 1-365 Hari

Sistem prediksi harga komoditas pertanian menggunakan LSTM neural networks dengan akselerasi GPU RAPIDS CUDA. Mendukung 4 komoditas utama dengan akurasi tinggi untuk short-term forecasting dan capability untuk long-term trend analysis.

---

## 📋 Daftar Isi

- [Ringkasan Singkat](#ringkasan-singkat)
- [📚 Dokumentasi Lengkap](#-dokumentasi-lengkap)
- [🚀 Quick Start](#-quick-start)
- [Teknologi Stack](#teknologi-stack)
- [Fitur Utama](#fitur-utama)
- [Project Status](#project-status)

---

## Ringkasan Singkat

### Business Case
Indonesia adalah negara agraris yang sangat bergantung pada sektor pertanian. Nusa Tenggara Barat, terutama Kabupaten Sumbawa, merupakan daerah yang penting dalam produksi komoditas pertanian nasional. Pada tahun 2023, jumlah rumah tangga usaha pertanian di Provinsi NTB mencapai 77.496 unit.

Fluktuasi harga komoditas seperti beras, jagung, dan kacang hijau berdampak langsung pada pendapatan petani, kelancaran distribusi, dan stabilitas ekonomi lokal. Sistem prediksi ini membantu mengurangi risiko dan meningkatkan perencanaan.

### Objective
Membangun sistem prediksi harga komoditas pertanian berkualitas tinggi untuk membantu:
- ✅ Petani merencanakan produksi dan penjualan
- ✅ Pedagang mengelola stok dan distribusi  
- ✅ Pemerintah merancang kebijakan stabilisasi harga

### Komoditas & Cakupan Data
- **4 Komoditas**: Beras Premium, Beras Medium, Jagung Pipil Kering, Kacang Hijau
- **Periode**: 1 Januari 2022 - 30 November 2024
- **Granularity**: Harga harian, Produktivitas triwulanan
- **Forecast Range**: 1-365 hari

---

## 📚 Dokumentasi Lengkap

Proyek ini memiliki dokumentasi komprehensif dalam 6 bagian. **Mulai dari sini untuk pemahaman mendalam**:

| File | Deskripsi | Untuk Siapa |
|------|-----------|-----------|
| [**01_Architecture.md**](docs/01_Architecture.md) | Desain sistem, data flow, technology stack | Developers, DevOps |
| [**02_Data_Dictionary.md**](docs/02_Data_Dictionary.md) | Spesifikasi data lengkap, feature engineering | Data Scientists, Analysts |
| [**03_Model.md**](docs/03_Model.md) | LSTM architecture, hyperparameters, GPU acceleration | ML Engineers, Researchers |
| [**04_Results.md**](docs/04_Results.md) | Performance metrics, model comparison, business insights | Decision Makers, Stakeholders |
| [**05_Setup.md**](docs/05_Setup.md) | Installation, environment, local & cloud deployment | DevOps, System Admin |
| [**06_Usage.md**](docs/06_Usage.md) | Web app tutorial, feature interpretation, FAQ | End Users, Business Users |

**Akses dokumentasi**: Lihat folder `docs/` atau klik link di atas.

---

## 🚀 Quick Start

### Instalasi & Setup (5 menit)

```bash
# 1. Clone repository
git clone <repo-url>
cd dsproject_agriculturepredict

# 2. Create conda environment
conda env create -f environment-local.yml
conda activate rapids-24.10

# 3. Verifikasi instalasi
python -c "import torch, cudf; print('✅ Setup OK')"
```

### Jalankan Aplikasi Streamlit (2 menit)

```bash
# Terminal 1: Activate environment
conda activate rapids-24.10

# Terminal 2: Run app
streamlit run streamlit_app/app.py

# Buka browser: http://localhost:8501
```

**Fitur Aplikasi**:
- 📊 **Dashboard**: Quick price forecast (default 30 hari)
- 🔍 **Prediksi Lanjutan**: Advanced analysis, model comparison, export data

### Jalankan Jupyter Notebooks

```bash
jupyter lab
# Buka: notebooks/01_Business_Understanding.ipynb
# Ikuti alur CRISP-DM (01 → 02 → 03 → 04 → 05 → 06 → 07)
```

Lihat [Setup Guide](docs/05_Setup.md) untuk detail instalasi lengkap.

### Deployment ke Cloud

```bash
# Streamlit Cloud (free, recommended)
1. Push ke GitHub
2. Go to https://share.streamlit.io/
3. Connect repo & deploy

# Docker (production)
docker build -t agriculture-predict:latest .
docker run -p 8501:8501 agriculture-predict:latest
```

Lihat [Setup Guide § Cloud Deployment](docs/05_Setup.md#deployment-ke-cloud) untuk detail.

---

## Teknologi Stack

| Kategori | Tools |
|----------|-------|
| **Data Processing** | RAPIDS cuDF 24.10, Pandas, NumPy |
| **ML Framework** | PyTorch 2.0+ (LSTM) |
| **GPU** | CUDA 11.8, NVIDIA GPUs |
| **Web App** | Streamlit 1.28, Plotly 5.17 |
| **Optimization** | Scikit-learn, Optuna hyperparameter tuning |
| **Environment** | Conda 4.10+, Python 3.10+ |

---

## Fitur Utama

### ✨ Model Capabilities

| Aspek | Kemampuan |
|-------|-----------|
| **Komoditas** | 4 komoditas pertanian utama |
| **Mode Prediksi** | Univariate (price only) + Multivariate (with production) |
| **Forecast Range** | 1-365 hari (dengan adaptive confidence intervals) |
| **Accuracy** | ~99% (7 hari), ~95% (30 hari), ~50% (365 hari) |
| **GPU Support** | RAPIDS CUDA 24.10 (45ms inference) |

### 📈 Prediksi Output

```
Input:  Komoditas (Beras Premium) + Horizon (30 hari)
Output: 
  - Harga prediksi harian
  - Confidence interval (adaptive widening)
  - MAE metric (actual accuracy)
  - Historical reference
```

### 📊 Visualisasi

- Interactive Plotly charts dengan hover details
- Confidence bands yang auto-adjust per horizon
- Monthly summaries & trend analysis
- CSV export untuk further analysis

---

## Project Status

### ✅ Completed

- [x] 4 Univariate LSTM models (1 per commodity)
- [x] 1 Multivariate LSTM model (all commodities)
- [x] Extended forecasting: 90 → 365 hari
- [x] Adaptive confidence intervals
- [x] Streamlit dashboard + Advanced page
- [x] GPU acceleration (RAPIDS CUDA)
- [x] Hyperparameter tuning (Optuna)
- [x] **Comprehensive documentation (6 files)**

### 📊 Model Performance (Best Models)

| Commodity | Mode | MAE | MAPE | Accuracy |
|-----------|------|-----|------|----------|
| Beras Medium | Multivariate | 44.18 Rp/kg | 0.88% | ⭐⭐⭐⭐⭐ |
| Beras Medium | Univariate | 70.43 Rp/kg | 1.12% | ⭐⭐⭐⭐⭐ |
| Jagung Pipil K. | Univariate | 128.54 Rp/kg | 2.18% | ⭐⭐⭐⭐ |
| Beras Premium | Univariate | 183.99 Rp/kg | 1.35% | ⭐⭐⭐⭐ |

📝 Full results in [Results Documentation](docs/04_Results.md)

### 📚 Key Files

- `src/train.py` — Model training orchestration
- `streamlit_app/app.py` — Main web interface
- `streamlit_app/pages/2_Prediksi_Lanjutan.py` — Advanced analysis page
- `runs/recomputed_compare.csv` — Latest baseline metrics
- `models/best/` — Best checkpoints per commodity
- `notebooks/` — CRISP-DM workflow (01-07)

---

## 🔧 Rekayasa & Development

### CRISP-DM Workflow

1. **Business Understanding** — Mendefinisikan tujuan bisnis & metrik sukses
2. **Data Understanding** — Eksplorasi harga & data produktivitas  
3. **Data Preparation** — Cleaning, transformasi, feature engineering
4. **Modeling** — Training LSTM univariat & multivariat dengan Optuna tuning
5. **Evaluation** — Validasi model, evaluasi metrik (MAE, RMSE, MAPE, sMAPE)
6. **Deployment** — Streamlit web app, cloud deployment
7. **Monitoring** — Performance tracking, automated retraining

### Notebooks (CRISP-DM Sequence)

- `notebooks/01_Business_Understanding.ipynb` — Objectives & success metrics
- `notebooks/02_Data_Understanding.ipynb` — EDA & data quality analysis
- `notebooks/03_Data_Preparation.ipynb` — Feature engineering & preprocessing
- `notebooks/04_Exploratory_Data_Analysis.ipynb` — Advanced EDA & visualization
- `notebooks/05_Modeling.ipynb` — Model training & hyperparameter tuning
- `notebooks/06_Evaluation.ipynb` — Results analysis & recommendations
- `notebooks/07_Deployment.ipynb` — Streamlit app deployment guide

---

## 🎯 Reproducing Experiments

### 1. Recompute Baseline (All Commodities)

```bash
conda activate rapids-24.10

python - <<'PY'
from pathlib import Path
import yaml, subprocess, sys
root = Path('.').resolve()
cfg = yaml.safe_load((root/'config.yaml').read_text())
commodities = [c['name'] for c in cfg.get('commodities',[])]

out_compare = root/'runs'/'recomputed_compare.csv'
if out_compare.exists():
    out_compare.unlink()

for comm in commodities:
    print(f'\n=== Recomputing for {comm} ===')
    log_csv = root/'runs'/f'recomputed_metrics_{comm.replace(" ","_")}.csv'
    cmd = [sys.executable, str(root/'src'/'train.py'), 
           '--commodity', comm, '--mode', 'both', 
           '--compare-csv', str(out_compare), '--log-csv', str(log_csv),
           '--epochs', '60', '--device', 'cpu', '--early-stopping-patience', '5']
    subprocess.run(cmd, cwd=str(root), check=True)
    
print(f'\n✅ Baseline complete: {out_compare}')
PY
```

### 2. Hyperparameter Tuning with Optuna

```bash
# Example: Tune Beras Premium multivariate model
python src/train.py \
  --commodity "Beras Premium" \
  --mode multivariate \
  --optuna \
  --optuna-trials 50 \
  --device cpu \
  --log-dir runs/optuna/Beras_Premium \
  --log-csv runs/optuna_metrics.csv
```

### 3. Select & Save Best Models

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd, shutil

root = Path('.').resolve()
runs = root/'runs'
out_dir = root/'models'/'best'
out_dir.mkdir(parents=True, exist_ok=True)

# Read baseline comparison
base = pd.read_csv(runs/'recomputed_compare.csv')

# Pick best mode per commodity by MAE
best = base.loc[base.groupby('commodity')['mae'].idxmin()].reset_index(drop=True)

rows = []
for _, r in best.iterrows():
    comm = r['commodity']
    mode = r['mode']
    mae = float(r['mae'])
    
    # Find checkpoint
    comm_safe = comm.replace(' ','_')
    src_candidates = [
        runs/comm_safe/mode/f'lstm_{comm_safe}_{mode}.pth',
        runs/comm_safe/mode/f'lstm_{comm_safe}.pth',
        runs/comm_safe/f'lstm_{comm_safe}_{mode}.pth',
    ]
    
    src_path = None
    for p in src_candidates:
        if p.exists():
            src_path = p
            break
    
    if not src_path and (runs/comm_safe).exists():
        cand = list((runs/comm_safe).rglob('*.pth'))
        src_path = cand[0] if cand else None
    
    dest_path = None
    if src_path:
        dest_path = out_dir / f"{comm_safe}_{mode}_mae{mae:.3f}.pth"
        shutil.copy2(src_path, dest_path)
        print(f'✅ Copied {src_path} → {dest_path}')
    else:
        print(f'❌ No checkpoint found for {comm} {mode}')
    
    rows.append({'commodity':comm, 'mode':mode, 'mae':mae, 
                 'src':str(src_path) if src_path else '', 
                 'dest':str(dest_path) if dest_path else ''})

import csv
summary_file = out_dir/'selected_models.csv'
with open(summary_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['commodity','mode','mae','src','dest'])
    writer.writeheader()
    writer.writerows(rows)
print(f'\n✅ Summary: {summary_file}')
PY
```

---

## 📁 Project Structure

```
dsproject_agriculturepredict/
├── docs/                          # 📚 Dokumentasi lengkap
│   ├── 01_Architecture.md         # System design & data flow
│   ├── 02_Data_Dictionary.md      # Data specifications
│   ├── 03_Model.md                # Model architecture & training
│   ├── 04_Results.md              # Performance & insights
│   ├── 05_Setup.md                # Installation & deployment
│   └── 06_Usage.md                # User guide & FAQ
│
├── streamlit_app/                 # 🌐 Web Application
│   ├── app.py                     # Main dashboard
│   ├── pages/
│   │   └── 2_Prediksi_Lanjutan.py # Advanced analysis page
│   ├── utils/
│   │   ├── data_processor.py      # Data pipeline
│   │   ├── model_loader.py        # Model loading
│   │   └── visualization.py       # Chart generation
│   └── assets/                    # Static files
│
├── src/                           # 🔧 Core modules
│   ├── train.py                   # Training orchestration
│   ├── data/
│   │   ├── prepare.py             # Data preparation
│   │   └── datasets.py            # PyTorch datasets
│   ├── models/
│   │   ├── lstm.py                # LSTM architecture
│   │   └── trainer.py             # Training loops
│   ├── features/                  # Feature engineering
│   ├── utils/                     # Utilities
│   └── visualization/             # Plotting functions
│
├── notebooks/                     # 📓 Jupyter Notebooks
│   ├── 01_Business_Understanding.ipynb
│   ├── 02_Data_Understanding.ipynb
│   ├── 03_Data_Preparation.ipynb
│   ├── 04_Exploratory_Data_Analysis.ipynb
│   ├── 05_Modeling.ipynb
│   ├── 06_Evaluation.ipynb
│   └── 07_Deployment.ipynb
│
├── data/                          # 📊 Data directory
│   ├── raw/                       # Original CSV files
│   ├── processed/                 # Cleaned & transformed
│   └── external/                  # External data
│
├── models/                        # 🧠 Trained Models
│   ├── *.pth                      # Individual commodity models
│   ├── best/                      # Best performing models
│   │   ├── selected_models.csv
│   │   └── *.pth
│   └── modelscaler/               # Feature scalers
│
├── runs/                          # 📈 Training Logs
│   ├── recomputed_compare.csv     # Latest baseline metrics
│   ├── all_commodity_optuna_summary.csv
│   ├── optuna_metrics.csv
│   └── [commodity]/               # Per-commodity runs
│
├── tests/                         # ✅ Unit tests
├── config.yaml                    # Configuration file
├── environment-local.yml          # Conda environment (local dev)
├── requirements.txt               # Python packages (cloud)
└── README.md                      # This file
```

---

## 📖 Recommended Reading Order

**Baru dalam project?** Ikuti urutan ini:

1. **[Quick Start](#-quick-start)** (5 menit) — Setup & jalankan app
2. **[Architecture](docs/01_Architecture.md)** (10 menit) — Pahami system design
3. **[Data Dictionary](docs/02_Data_Dictionary.md)** (15 menit) — Kenali data
4. **[Results](docs/04_Results.md)** (10 menit) — Lihat performance
5. **[Usage Guide](docs/06_Usage.md)** (15 menit) — Gunakan aplikasi

**Untuk development?**:

1. **[Setup Guide](docs/05_Setup.md)** — Instalasi lengkap
2. **[Model Documentation](docs/03_Model.md)** — Architecture & training
3. **[Notebooks](notebooks/)** — CRISP-DM workflow (01-07)

---

## 🌐 Aplikasi Online

**Streamlit Cloud** (Recommended):
```
https://app-username-projectname.streamlit.app
```

**Local Development**:
```bash
streamlit run streamlit_app/app.py
# Access: http://localhost:8501
```

---

## 💡 Tips & Troubleshooting

### Akurasi Rendah?
- Gunakan **univariate** untuk short-term (< 30 hari)
- Gunakan **multivariate** untuk long-term (> 90 hari)
- Lihat [Model Results](docs/04_Results.md) untuk rekomendasi per komoditas

### GPU Issues?
```bash
# Check GPU
nvidia-smi

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

Lihat [Setup Troubleshooting](docs/05_Setup.md#troubleshooting) untuk solusi lengkap.

### Pertanyaan Umum?

Lihat [FAQ di Usage Guide](docs/06_Usage.md#faq) untuk jawaban detail.

---

## 📊 Key Metrics

```
┌─────────────────────────────────────────────────────┐
│              MODEL PERFORMANCE SUMMARY               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Best Accuracy:    Beras Medium (MAE 44 Rp/kg)     │
│  Most Volatile:    Kacang Hijau (MAE 276 Rp/kg)    │
│  Forecast Range:   1 - 365 hari                     │
│  GPU Inference:    45 ms / prediction               │
│  Training Time:    4-8 minutes per model            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 📚 Data Sources

- **Harga Komoditi**: Badan Pusat Statistik Provinsi NTB
- **Produktivitas**: BPS NTB (Quarterly data)
- **Period**: 1 Januari 2022 - 30 November 2024

Files:
- `data/raw/DATASET HARGA KOMODITI.csv` (2,000+ records)
- `data/raw/produksi.csv` (60+ records)

---

## 🚀 Next Steps

- [ ] Deploy ke Streamlit Cloud
- [ ] Set up automated retraining (monthly)
- [ ] Monitor production predictions weekly
- [ ] Gather user feedback & iterate
- [ ] Consider adding weather data untuk improved accuracy
- [ ] Explore ensemble methods with statistical forecasts

---

## 📝 License & Attribution

**Project Owner**: Rizki Nurhafizd Achmad  
**Last Updated**: June 2026  
**Status**: ✅ Production Ready

---

## 📞 Contact & Support

- **Issues/Bugs**: Create GitHub issue
- **Feature Requests**: Open GitHub discussion
- **Documentation**: See `docs/` folder

---

## 📖 References

- Badan Pusat Statistik Provinsi Nusa Tenggara Barat, 2023
- PyTorch LSTM Documentation: https://pytorch.org/docs/stable/nn.html#lstm
- RAPIDS cuDF Guide: https://docs.rapids.ai/
- Streamlit Documentation: https://docs.streamlit.io/

---

**Selamat datang di sistem prediksi harga komoditas pertanian! 🌾**

Untuk memulai, ikuti [Quick Start](#-quick-start) atau baca [Dokumentasi Lengkap](docs/).

