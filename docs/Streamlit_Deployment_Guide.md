# Streamlit Deployment Guide

## Pendahuluan

Dokumen ini adalah panduan lengkap untuk membuat, mengembangkan, dan men-deploy Web App Streamlit untuk project "Prediksi Harga Komoditas Pertanian Kabupaten Sumbawa".

Tujuan utama web app:

- Menyajikan prediksi harga komoditas pertanian secara interaktif dan visual.
- Menampilkan hasil model LSTM yang sudah dilatih untuk 4 komoditas univariate dan 1 model multivariate.
- Memfasilitasi pengguna non-teknis seperti petani, pedagang, dan pembuat kebijakan untuk melihat tren harga, evaluasi model, dan insight bisnis.
- Menyediakan analisis perbandingan univariate vs multivariate.

Konsep desain:

- Tema Glassmorphism: panel semi-transparent, efek blur, dan aksen warna modern.
- Tampilan elegan, minimalis, dan responsif dengan latar belakang gelap/gradien.
- Fokus pada keterbacaan dan pengalaman pengguna yang nyaman.

---

## 1. Fitur Lengkap Web App

Fitur yang direkomendasikan untuk aplikasi:

- Pilih komoditas melalui dropdown.
- Pilih periode prediksi: 7, 14, 30 hari ke depan atau custom date range.
- Input data tambahan secara manual untuk what-if analysis.
- Grafik line chart historis dan forecast dengan shading confidence interval.
- Perbandingan metrik univariate vs multivariate.
- Dashboard metrik evaluasi (MAE, RMSE, MAPE, sMAPE, Directional Accuracy).
- Halaman terpisah untuk Analisis Model dan Insights Otomatis.
- Rekomendasi bisnis berdasarkan hasil prediksi.
- Download hasil prediksi dalam format CSV.
- Model selector antara univariate dan multivariate.
- Informasi metadata model dan sumber data.
- Dukungan GPU (PyTorch CUDA) jika tersedia, dan fallback CPU jika tidak.

---

## 2. Struktur Folder yang Direkomendasikan

Struktur yang rapi dan profesional:

```
streamlit_app/
├── app.py
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Prediksi_Lanjutan.py
│   ├── 3_Analisis_Model.py
│   └── 4_Insights.py
├── utils/
│   ├── model_loader.py
│   ├── data_processor.py
│   ├── visualization.py
│   └── insights_generator.py
├── assets/
│   ├── css/
│   │   └── glassmorphism.css
│   └── images/
├── models/                     # symlink atau copy dari root models/
│   └── best/
├── config.yaml
├── requirements.txt
└── .streamlit/config.toml
```

Detail:

- `app.py`: entry point utama Streamlit.
- `pages/`: modul halaman Streamlit agar UI tersegmentasi.
- `utils/`: helper untuk load model, preprocessing, visualisasi, dan insight.
- `assets/css/glassmorphism.css`: styling tema glassmorphism.
- `models/best/`: checkpoint terbaik model.
- `config.yaml`: konfigurasi komoditas dan parameter model.
- `.streamlit/config.toml`: konfigurasi server Streamlit.

---

## 3. Custom CSS untuk Glassmorphism

Gunakan CSS berikut untuk efek glassmorphism:

```css
/* streamlit_app/assets/css/glassmorphism.css */
:root {
  --bg-gradient: linear-gradient(135deg, rgba(10, 18, 40, 0.95), rgba(24, 45, 84, 0.95));
  --panel-bg: rgba(255, 255, 255, 0.10);
  --panel-border: rgba(255, 255, 255, 0.16);
  --panel-shadow: 0 20px 60px rgba(0, 0, 0, 0.22);
  --text-color: #eef5ff;
  --accent: #7fd1b9;
  --accent-alt: #f8d57e;
}

body {
  background: var(--bg-gradient);
  color: var(--text-color);
  font-family: "Inter", "Segoe UI", sans-serif;
}

section.main {
  background: rgba(10, 18, 40, 0.72) !important;
  backdrop-filter: blur(20px);
  border: 1px solid var(--panel-border) !important;
  border-radius: 28px !important;
  box-shadow: var(--panel-shadow) !important;
}

[data-testid="stSidebar"] {
  background: rgba(10, 18, 40, 0.82) !important;
  border: 1px solid rgba(255, 255, 255, 0.14) !important;
  backdrop-filter: blur(18px) !important;
}

[data-testid="stHeader"] {
  background: transparent !important;
}

.stButton>button {
  background: rgba(127, 209, 185, 0.22) !important;
  border: 1px solid rgba(127, 209, 185, 0.40) !important;
  color: #eef5ff !important;
  backdrop-filter: blur(10px) !important;
}

.stButton>button:hover {
  background: rgba(127, 209, 185, 0.30) !important;
}

div[data-testid="metric-container"] {
  background: rgba(255, 255, 255, 0.08) !important;
  border: 1px solid rgba(255, 255, 255, 0.14) !important;
  border-radius: 20px !important;
  padding: 1rem !important;
}

.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4,
.stMarkdown h5,
.stMarkdown h6 {
  color: #eef5ff !important;
}

.css-1v0mbdj, .css-1ubnkh, .css-1mkc9eh {
  background: rgba(255, 255, 255, 0.08) !important;
}

.stProgress>div>div {
  background: #7fd1b9 !important;
}

div[data-testid="stMetricValue"] {
  color: #eef5ff !important;
}
```

> Catatan: nama kelas Streamlit dapat berubah antar versi. Jika ada perbedaan, sesuaikan selektor CSS dengan inspeksi elemen.

---

## 4. Panduan Step-by-Step Pembuatan

### 4.1 Setup Streamlit dengan Custom CSS

1. Buat virtual environment baru.
2. Install dependencies dengan `pip install -r requirements.txt`.
3. Simpan CSS di `streamlit_app/assets/css/glassmorphism.css`.
4. Di `app.py`, muat CSS menggunakan `st.markdown(..., unsafe_allow_html=True)`.
5. Pastikan `config.yaml` tersedia di folder root atau `streamlit_app/`.

### 4.2 Load Model PyTorch `.pth` dengan GPU support

- Deteksi device CUDA dengan `torch.cuda.is_available()`.
- Muat checkpoint menggunakan `torch.load(..., map_location=device)`.
- Kedua model univariate dan multivariate harus didukung.
- Gunakan fallback CPU di lingkungan tanpa GPU.

Contoh:

```python
import torch


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pytorch_model(checkpoint_path, model_class, device=None):
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_class()
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
```

### 4.3 Implementasi Dashboard Interaktif

Komponen yang harus tersedia:

- Sidebar untuk kontrol komoditas, periode prediksi, jenis model, dan input manual.
- Halaman dashboard utama dengan grafik harga historis dan forecast.
- Halaman prediksi lanjutan untuk what-if analysis.
- Halaman analisis model untuk membandingkan error dan metrik.
- Halaman insights untuk rekomendasi bisnis.

### 4.4 Pembuatan Insight Otomatis

- Buat logika sederhana berdasarkan tren persentase prediksi dan error model.
- Tampilkan teks insight di bawah grafik.
- Sertakan rekomendasi tindakan untuk petani dan pembuat kebijakan.
- Gunakan template message sehingga insight mudah dibaca.

### 4.5 Caching di Streamlit

- `@st.cache_resource` untuk memuat model dan objek besar.
- `@st.cache_data` untuk memuat dataset, scaler, dan hasil preprocessing.
- Cache hasil prediksi jika input tidak berubah.
- Cache membantu performa interaksi dan mengurangi latency.

### 4.6 Input Manual dan Custom Prediction Horizon

- Sediakan input manual untuk variabel eksogen dan lag.
- Gunakan `st.number_input` atau `st.date_input`.
- Validasi data segera setelah input.
- Jika user memilih `Custom`, tampilkan slider atau field input `1-90 hari`.

---

## 5. Contoh Kode Penting

Berikut contoh kode yang bisa dijadikan template.

### 5.1 `app.py`

```python
# streamlit_app/app.py
import streamlit as st
from pathlib import Path
from utils.data_processor import load_config, load_commodity_data
from utils.model_loader import get_model_bundle
from utils.visualization import build_line_chart
from utils.insights_generator import generate_insights

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title='Prediksi Harga Komoditas Sumbawa',
    layout='wide',
)

with open(ROOT / 'assets' / 'css' / 'glassmorphism.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

config = load_config(ROOT / 'config.yaml')
commodities = [item['name'] for item in config['commodities']]

with st.sidebar:
    st.title('Kontrol Prediksi')
    commodity = st.selectbox('Pilih Komoditas', commodities)
    model_mode = st.radio('Pilih Model', ['Univariate', 'Multivariate'])
    horizon_option = st.selectbox('Horizon Prediksi', [7, 14, 30, 'Custom'])
    if horizon_option == 'Custom':
        horizon = st.number_input('Hari ke Depan', min_value=1, max_value=90, value=30)
    else:
        horizon = int(horizon_option)
    st.markdown('---')
    st.write('Input manual (opsional)')
    manual_feature = st.number_input('Produksi tambahan (ton)', min_value=0.0, value=0.0)

st.title('Prediksi Harga Komoditas Pertanian Kabupaten Sumbawa')
st.markdown('Grafik historis harga dan forecast prediksi LSTM.')

commodity_data = load_commodity_data(commodity)
model_bundle = get_model_bundle(commodity, model_mode)

forecast, lower, upper = model_bundle.predict_with_interval(
    commodity_data['features'], horizon=horizon, manual_feature=manual_feature)

fig = build_line_chart(
    history=commodity_data['history'],
    forecast=forecast,
    lower=lower,
    upper=upper,
    title=f'Forecast {commodity} ({model_mode})',
)
st.plotly_chart(fig, use_container_width=True)

metrics = commodity_data['metrics'][model_mode.lower()]
cols = st.columns(4)
cols[0].metric('MAE', f"{metrics['mae']:.2f}")
cols[1].metric('RMSE', f"{metrics['rmse']:.2f}")
cols[2].metric('MAPE', f"{metrics['mape']:.2f}%")
cols[3].metric('sMAPE', f"{metrics['smape']:.2f}%")

insight_text = generate_insights(
    history=commodity_data['history'],
    forecast=forecast,
    metrics=commodity_data['metrics'],
    commodity_name=commodity,
)
st.info(insight_text)
```

### 5.2 `pages/1_Dashboard.py`

```python
# streamlit_app/pages/1_Dashboard.py
import streamlit as st
from utils.data_processor import load_commodity_data, get_commodity_names
from utils.model_loader import get_model_bundle
from utils.visualization import build_line_chart

st.header('Dashboard Harga Historis dan Forecast')
commodity_options = get_commodity_names()
selected_commodity = st.selectbox('Pilih Komoditas', commodity_options)
model_mode = st.radio('Model', ['Univariate', 'Multivariate'])
forecast_horizon = st.selectbox('Horizon Prediksi', [7, 14, 30, 60])

data = load_commodity_data(selected_commodity)
model_bundle = get_model_bundle(selected_commodity, model_mode)
forecast, lower, upper = model_bundle.predict_with_interval(data['features'], forecast_horizon)
fig = build_line_chart(data['history'], forecast, lower, upper)
st.plotly_chart(fig, use_container_width=True)

st.subheader('Evaluasi Model')
cols = st.columns(3)
metrics = data['metrics'][model_mode.lower()]
cols[0].metric('MAE', f"{metrics['mae']:.2f}")
cols[1].metric('RMSE', f"{metrics['rmse']:.2f}")
cols[2].metric('MAPE', f"{metrics['mape']:.2f}%")
```

### 5.3 `utils/model_loader.py`

```python
# streamlit_app/utils/model_loader.py
from pathlib import Path
import torch
import joblib
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'best'


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def load_model(checkpoint_path: str, model_class):
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_class()
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


@st.cache_data
def load_scaler(scaler_path: str):
    if Path(scaler_path).exists():
        return joblib.load(scaler_path)
    return None


class ModelBundle:
    def __init__(self, model, scaler=None, device=None):
        self.model = model
        self.scaler = scaler
        self.device = device or get_device()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_pred = self.model(tensor).cpu().numpy().ravel()

        if self.scaler is not None:
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred
```

### 5.4 `utils/insights_generator.py`

```python
# streamlit_app/utils/insights_generator.py
from typing import Dict


def generate_insights(history, forecast, metrics: Dict, commodity_name: str) -> str:
    last_price = history['target'].iloc[-1]
    forecast_mean = float(forecast['mean'].mean())
    change_pct = (forecast_mean - last_price) / last_price * 100
    trend = 'naik' if change_pct > 0 else 'turun' if change_pct < 0 else 'stabil'

    insights = [
        f'Prediksi harga {commodity_name} menunjukkan tren {trend} dalam horizon yang dipilih.',
        f'Rata-rata prediksi berada di kisaran Rp {forecast_mean:,.0f}.',
    ]

    if change_pct >= 5:
        insights.append('Harga diperkirakan naik signifikan; direkomendasikan perencanaan panen dan distribusi yang matang.')
    elif change_pct <= -5:
        insights.append('Harga diperkirakan turun; disarankan mitigasi stok dan dukungan pasar agar pendapatan petani stabil.')
    else:
        insights.append('Pergerakan harga relatif stabil; manfaatkan periode ini untuk perencanaan produksi.')

    if metrics.get('univariate') and metrics.get('multivariate'):
        diff = abs(metrics['univariate']['mae'] - metrics['multivariate']['mae'])
        if diff >= 5:
            insights.append('Perbedaan performa antara univariate dan multivariate besar; analisis fitur eksogen tambahan direkomendasikan.')

    if metrics.get('univariate', {}).get('mape', 0) > 15:
        insights.append('Model memiliki error relatif tinggi; lakukan validasi data input dan evaluasi ulang preprocessing.')

    return ' '.join(insights)
```

### 5.5 `utils/visualization.py`

```python
# streamlit_app/utils/visualization.py
import plotly.graph_objects as go


def build_line_chart(history, forecast, lower, upper, title='Harga Historis dan Prediksi'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['target'],
        name='Harga Historis',
        mode='lines',
        line=dict(color='#7fd1b9', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=forecast['mean'],
        name='Prediksi',
        mode='lines',
        line=dict(color='#f8d57e', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=upper,
        name='Confidence Upper',
        mode='lines',
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=forecast['date'],
        y=lower,
        name='Confidence Lower',
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(248, 213, 126, 0.2)',
        line=dict(width=0),
        showlegend=False,
    ))
    fig.update_layout(
        title=title,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eef5ff'),
        legend=dict(bgcolor='rgba(255,255,255,0.08)', bordercolor='rgba(255,255,255,0.12)'),
        margin=dict(t=30, b=20, l=20, r=20),
    )
    return fig
```

### 5.6 `utils/data_processor.py`

```python
# streamlit_app/utils/data_processor.py
from pathlib import Path
import yaml
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_config(path: Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['Tanggal'])


def get_commodity_names():
    config = load_config(ROOT / 'config.yaml')
    return [item['name'] for item in config['commodities']]


@st.cache_data
def load_commodity_data(commodity_name: str):
    config = load_config(ROOT / 'config.yaml')
    commodity_info = next(item for item in config['commodities'] if item['name'] == commodity_name)
    raw = load_csv(ROOT / 'data/raw/DATASET HARGA KOMODITI.csv')
    df = raw[raw['Komoditas'] == commodity_info['series_id']].copy()
    df = df.sort_values('Tanggal')
    return {
        'history': df[['Tanggal', commodity_info['target_column']]].rename(columns={commodity_info['target_column']: 'target'}),
        'features': df,
        'metrics': {
            'univariate': {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'smape': 0.0},
            'multivariate': {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'smape': 0.0},
        },
    }
```

---

## 6. Deployment Options

### 6.1 Local

```bash
pip install -r requirements.txt
cd streamlit_app
streamlit run app.py
```

Lokal cocok untuk demo internal dan pengujian.

### 6.2 Streamlit Community Cloud

1. Push `streamlit_app/` dan `requirements.txt` ke GitHub.
2. Hubungkan repository ke Streamlit Cloud.
3. Set entrypoint `app.py`.
4. Gunakan `Settings -> Secrets` untuk variabel environment.

Keterbatasan:
- Tidak ada akses GPU.
- Tidak cocok untuk RAPIDS.

### 6.3 Hugging Face Spaces

1. Buat Space baru bertipe Streamlit.
2. Push `streamlit_app/`, `requirements.txt`, dan asset CSS.
3. Jalankan dan periksa log build.

### 6.4 Docker

- Buat `Dockerfile` berbasis Python atau PyTorch.
- Pasang dependencies dan salin `streamlit_app`.
- Jalankan `streamlit run app.py --server.port $PORT`.
- Gunakan `docker-compose` untuk volume `models/` jika perlu.

---

## 7. Best Practices

### Responsiveness

- Gunakan `use_container_width=True` untuk grafik.
- Gunakan `st.columns` untuk tampilan metrik.
- Gunakan `st.expander` untuk detail tambahan.

### Error handling

- Tangkap exception saat model dan data tidak ditemukan.
- Tampilkan pesan jelas dengan `st.error`.
- Validasi input manual.

### Performance

- Cache model dan data.
- Hindari reload model setiap interaksi.
- Gunakan sampling untuk dataset historis besar.

### User experience

- Berikan deskripsi singkat tiap halaman.
- Tambahkan penjelasan istilah metrik.
- Tampilkan insight otomatis di area terpisah.
- Gunakan elemen visual yang konsisten.

---

## 8. Sinkronisasi dengan Proyek CRISP-DM

Panduan ini selaras dengan kondisi proyek:

- Model sudah pada tahap Evaluasi dan menggunakan LSTM.
- Terdapat 4 model univariate untuk 4 komoditas, dan 1 model multivariate.
- Data preprocessing menggunakan `config.yaml` dan `data/raw/DATASET HARGA KOMODITI.csv`.
- Evaluasi model diselaraskan dengan `notebooks/06_Evaluation.ipynb`.
- Gunakan `runs/recomputed_compare.csv` untuk menentukan model terbaik.
- Simpan checkpoint terbaik di `models/best/` dengan nama yang mudah dikenali.

---

## 9. Checklist Implementasi

- [ ] Buat `streamlit_app/app.py` sebagai entrypoint.
- [ ] Buat halaman modular di `streamlit_app/pages/`.
- [ ] Buat helper di `streamlit_app/utils/`.
- [ ] Tambahkan `streamlit_app/assets/css/glassmorphism.css`.
- [ ] Pastikan `config.yaml` berisi daftar komoditas.
- [ ] Validasi input custom horizon dan manual features.
- [ ] Terapkan caching model/data.
- [ ] Sediakan fallback GPU/CPU.
- [ ] Uji aplikasi lokal dengan `streamlit run app.py`.

---

## 10. Referensi Tambahan

- Notebook evaluasi: `notebooks/06_Evaluation.ipynb`
- Rekap model terbaik: `models/best/selected_models.csv`
- Hasil training dan perbandingan: `runs/recomputed_compare.csv`
- Data produksi dan fitur eksogen: `data/raw/produksi.csv`

Dokumen ini dirancang untuk membantu penerapan aplikasi prediksi harga komoditas yang profesional, modern, dan mudah dioperasikan oleh tim data dan stakeholder bisnis.
END_OF_FILE