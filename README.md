# Prediksi Harga Komoditas Pertanian Kabupaten Sumbawa

## Business Case
Indonesia adalah negara agraris yang sangat bergantung pada sektor pertanian. Nusa Tenggara Barat, terutama Kabupaten Sumbawa, merupakan daerah yang penting dalam produksi komoditas pertanian nasional. Pada tahun 2023, jumlah rumah tangga usaha pertanian di Provinsi NTB mencapai 77.496 unit (Badan Pusat Statistik Provinsi Nusa Tenggara Barat, 2023).

Fluktuasi harga komoditas seperti beras, jagung, dan kacang hijau berdampak langsung pada pendapatan petani, kelancaran distribusi, dan stabilitas ekonomi lokal. Contoh konkret adalah penurunan harga jagung hingga Rp 3.800/kg pada 29 April 2024, yang menyebabkan kerugian bagi petani dan memicu aksi protes publik (Gustiana, 2024).

## Objective
Tujuan utama proyek ini adalah membangun sistem prediksi harga komoditas pertanian di Kabupaten Sumbawa menggunakan model LSTM terakselerasi GPU. Sistem ini ditujukan untuk membantu:
- Petani dalam merencanakan produksi dan penjualan,
- Pedagang dalam mengelola stok dan distribusi,
- Pemerintah daerah dalam merancang kebijakan stabilisasi harga.

## Scope
Proyek awal akan fokus pada empat seri komoditas utama yang tersedia dalam data:
- Beras Premium
- Beras Medium
- Jagung Pipil Kering
- Kacang Hijau

Dataset ini mencakup harga komoditas harian dari 1 Januari 2022 hingga 30 November 2024, dan data produktivitas triwulanan.

## Model Strategy
Strategi modeling mencakup:
- Empat model LSTM univariat terpisah, satu model untuk masing-masing komoditas.
- Satu model LSTM multivariat sebagai benchmark.
- Penggunaan variabel eksogen seperti produktivitas, musim, dan kalender libur.
- Validasi dengan Time Series Cross Validation.
- Evaluasi menggunakan MAE, RMSE, MAPE, sMAPE, dan Directional Accuracy.

## Teknologi Stack
- RAPIDS 24.10: cuDF, cuPY, cuML
- PyTorch + CUDA
- Python 3.12.7
- Pandas, NumPy, Scikit-Learn, Optuna
- Matplotlib, Seaborn, Plotly
- JupyterLab

## CRISP-DM Workflow
1. Business Understanding: mendefinisikan tujuan bisnis dan sasaran model.
2. Data Understanding: eksplorasi data harga dan produktivitas.
3. Data Preparation: pembersihan, transformasi, dan feature engineering.
4. Modeling: pelatihan LSTM univariat dan multivariat.
5. Evaluation: validasi model dan evaluasi metrik.
6. Deployment: rencana penerapan model untuk inferensi.

## Cara Menjalankan Proyek
1. Aktifkan conda environment RAPIDS untuk pengembangan lokal:
   ```bash
   conda env create -f environment-local.yml
   conda activate rapids-24.10
   ```
   > Untuk deploy Streamlit, repository akan menggunakan `requirements.txt` karena Streamlit Cloud akan memilih environment `requirements.txt` jika file `environment.yml` tidak ada.
2. Periksa instalasi RAPIDS dan CUDA:
   ```bash
   python -c "import cudf, cupy, torch; print('RAPIDS', cudf.__version__, 'CuPy', cupy.__version__, 'Torch', torch.__version__)"
   ```
3. Jalankan JupyterLab:
   ```bash
   jupyter lab
   ```
4. Buka notebook `notebooks/01_Business_Understanding.ipynb` dan ikuti alur CRISP-DM.

## Project Status & Recent Findings

- `src/train.py` was fixed to restore the best validation checkpoint before final test evaluation. This prevents stale final metrics from being computed on non-optimal weights.
- A fresh baseline comparison was recomputed and saved to `runs/recomputed_compare.csv`. Prefer this file when evaluating current baseline performance.
- Optuna hyperparameter search was executed; final Optuna results are in `runs/all_commodity_optuna_summary.csv` and detailed Optuna trial metrics are in `runs/optuna_metrics.csv`.
- I collected the best checkpoint per commodity (based on the recomputed baseline MAE) and copied them to `models/best/` with a summary file `models/best/selected_models.csv`.

Files of interest (workspace-relative):
- `runs/recomputed_compare.csv` — recomputed baseline comparison (preferred)
- `runs/all_commodity_optuna_summary.csv` — Optuna final multivariate retrain results
- `runs/optuna_metrics.csv` — per-trial Optuna metrics
- `models/best/` — saved best checkpoints and `selected_models.csv` summary

## Reproduce Key Steps

1) Recompute baseline comparisons for all commodities (fresh overwrite):

```bash
conda activate rapids-24.10
python - <<'PY'
from pathlib import Path
import yaml, subprocess, sys
root=Path('.').resolve()
cfg=yaml.safe_load((root/'config.yaml').read_text())
out_compare=root/'runs'/'recomputed_compare.csv'
if out_compare.exists():
   out_compare.unlink()
for comm in [c['name'] for c in cfg['commodities']]:
   log_csv = root/'runs'/f'recomputed_metrics_{comm.replace(" ","_")}.csv'
   cmd=[sys.executable, str(root/'src'/'train.py'), '--commodity', comm, '--mode', 'both', '--compare-csv', str(out_compare), '--log-csv', str(log_csv), '--epochs', '60', '--device', 'cpu', '--early-stopping-patience', '5']
   subprocess.run(cmd, cwd=str(root), check=True)
PY
```

2) Run Optuna tuning for a commodity (example):

```bash
conda activate rapids-24.10
python src/train.py --commodity "Beras Premium" --mode multivariate --optuna --optuna-trials 50 --device cpu --log-dir runs/optuna/Beras_Premium --log-csv runs/optuna_metrics.csv
```

3) Save best models (scripted): the repo contains a helper that copies selected checkpoints into `models/best/` and writes `models/best/selected_models.csv`.

## Notebooks (updated)
- `notebooks/05_Modeling_v2.ipynb` — added training orchestration cell and a comparison summary cell that prefers `runs/recomputed_compare.csv`.
- `notebooks/06_Evaluation.ipynb` — added an Optuna insight cell and logic to load the preferred comparison CSV; contains recommendations per commodity.

## Recommendations & Next Steps
- Use `runs/recomputed_compare.csv` as the canonical baseline for evaluation and dashboards.
- Promote Optuna checkpoints for `Beras Premium` and `Jagung Pipil Kering` to further validation and holdout testing before deployment.
- Investigate `Beras Medium` Optuna runs (Optuna result was worse than baseline); check data joins, scalers, and Optuna search space.
- Add a lightweight CI check that ensures `src/train.py` always loads the best checkpoint before final metrics are computed.

## Contact / Maintainers
- Repo owner: Rizki Nurhafizd Achmad
- For reproducing long-running experiments (Optuna / full retrain), run on a machine with sufficient CPU/GPU and adjust `--device` accordingly.

## Data Mentah Tersedia
- data/raw/DATASET HARGA KOMODITI.csv (harga komoditas harian)
- data/raw/produksi.csv (data produktivitas triwulanan)

## Referensi
- Badan Pusat Statistik Provinsi Nusa Tenggara Barat, 2023.
- Gustiana, 2024.
