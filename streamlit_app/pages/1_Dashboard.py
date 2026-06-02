import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_processor import (
    create_forecast,
    evaluate_model,
    get_commodity_names,
    prepare_commodity_dataset,
)
from utils.model_loader import get_device, load_model
from utils.visualization import build_line_chart
from utils.insights_generator import generate_insights

st.header('Dashboard Harga Historis dan Forecast')

commodity = st.selectbox('Pilih Komoditas', get_commodity_names())
model_mode = st.radio('Model', ['Univariate', 'Multivariate'])
forecast_horizon = st.selectbox('Horizon Prediksi', [7, 14, 30, 60])
manual_feature = st.number_input('Produksi tambahan (ton)', min_value=0.0, value=0.0, step=0.1)

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

    forecast, lower, upper = create_forecast(model, dataset, forecast_horizon, manual_feature)
    figure = build_line_chart(dataset['history'], forecast, lower, upper)
    st.plotly_chart(figure, use_container_width=True)

    st.subheader('Evaluasi Model')
    metrics = evaluate_model(model, dataset)
    cols = st.columns(3)
    cols[0].metric('MAE', f"{metrics['mae']:.2f}")
    cols[1].metric('RMSE', f"{metrics['rmse']:.2f}")
    cols[2].metric('MAPE', f"{metrics['mape']:.2f}%")

    insight_text = generate_insights(
        history=dataset['history'],
        forecast=forecast,
        metrics={'univariate': metrics, 'multivariate': metrics},
        commodity_name=commodity,
    )
    st.info(insight_text)
except Exception as exc:
    st.error(f'Gagal memuat data atau model: {exc}')

st.write('Aplikasi ini memanfaatkan model LSTM yang telah dilatih untuk memberikan perkiraan harga dan wawasan operasional.')
