import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from utils.data_processor import (
    create_forecast,
    evaluate_model,
    get_commodity_names,
    prepare_commodity_dataset,
)
from utils.model_loader import get_device, load_model
from utils.visualization import build_line_chart
from utils.insights_generator import generate_insights

st.set_page_config(
    page_title='Prediksi Harga Komoditas Sumbawa',
    layout='wide',
)

with open(ROOT / 'assets' / 'css' / 'glassmorphism.css', 'r') as css_file:
    st.markdown(f'<style>{css_file.read()}</style>', unsafe_allow_html=True)

st.sidebar.title('Kontrol Prediksi')
commodity = st.sidebar.selectbox('Pilih Komoditas', get_commodity_names())
model_mode = st.sidebar.radio('Pilih Model', ['Univariate', 'Multivariate'])
horizon_option = st.sidebar.selectbox('Horizon Prediksi', [7, 14, 30, 'Custom'])
if horizon_option == 'Custom':
    horizon = st.sidebar.number_input('Hari ke Depan', min_value=1, max_value=90, value=30)
else:
    horizon = int(horizon_option)

st.sidebar.markdown('---')
st.sidebar.write('Input manual (opsional)')
manual_feature = st.sidebar.number_input('Produksi tambahan (ton)', min_value=0.0, value=0.0, step=0.1)

device = get_device()
st.sidebar.caption(f'Device inference: {device}')

st.title('Prediksi Harga Komoditas Pertanian Kabupaten Sumbawa')
st.markdown(
    'Aplikasi ini menampilkan harga historis dan prediksi LSTM untuk komoditas utama Kabupaten Sumbawa.'
)

try:
    dataset = prepare_commodity_dataset(commodity, model_mode)
    model = load_model(
        commodity,
        model_mode,
        input_size=dataset['input_size'],
        hidden_size=dataset['hidden_size'],
        num_layers=dataset['num_layers'],
        dropout=dataset['dropout'],
    )

    forecast, lower, upper = create_forecast(model, dataset, horizon, manual_feature)
    chart = build_line_chart(
        dataset['history'],
        forecast,
        lower,
        upper,
        title=f'Forecast {commodity} ({model_mode})',
    )
    st.plotly_chart(chart, use_container_width=True)

    metrics = evaluate_model(model, dataset)
    cols = st.columns(4)
    cols[0].metric('MAE', f"{metrics['mae']:.2f}")
    cols[1].metric('RMSE', f"{metrics['rmse']:.2f}")
    cols[2].metric('MAPE', f"{metrics['mape']:.2f}%")
    cols[3].metric('sMAPE', f"{metrics['smape']:.2f}%")

    insight_text = generate_insights(
        history=dataset['history'],
        forecast=forecast,
        metrics={'univariate': metrics, 'multivariate': metrics},
        commodity_name=commodity,
    )
    st.info(insight_text)
except Exception as exc:
    st.error(f'Gagal memuat data atau model: {exc}')
