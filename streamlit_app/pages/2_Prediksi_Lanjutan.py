"""
Advanced Prediction Page
Memungkinkan user untuk melakukan prediksi dengan granularity yang lebih tinggi,
perbandingan model, dan analisis mendalam.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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

st.set_page_config(page_title='Prediksi Lanjutan', layout='wide')

st.header('🔮 Prediksi Lanjutan')
st.markdown(
    'Lakukan prediksi dengan horizon yang lebih panjang, perbandingan model, '
    'dan analisis mendalam terhadap akurasi prediksi.'
)

# Sidebar configuration
st.sidebar.subheader('⚙️ Konfigurasi Prediksi')

col_commodity, col_mode = st.sidebar.columns(2)
with col_commodity:
    commodity = st.selectbox('Komoditas', get_commodity_names(), key='commodity_adv')
with col_mode:
    model_mode = st.radio('Model', ['Univariate', 'Multivariate'], key='mode_adv')

# Horizon selection with detailed options
st.sidebar.subheader('📅 Horizon Prediksi')
horizon_preset = st.sidebar.select_slider(
    'Pilih Horizon',
    options=['7 Hari', '30 Hari', '90 Hari', '180 Hari', '365 Hari', 'Custom'],
    value='90 Hari',
    key='horizon_preset_adv',
)

if horizon_preset == 'Custom':
    horizon_days = st.sidebar.slider(
        'Hari ke Depan',
        min_value=1,
        max_value=365,
        value=90,
        step=7,
        key='horizon_days_adv',
    )
else:
    preset_map = {
        '7 Hari': 7,
        '30 Hari': 30,
        '90 Hari': 90,
        '180 Hari': 180,
        '365 Hari': 365,
    }
    horizon_days = preset_map[horizon_preset]

# Show horizon warning
if horizon_days > 90:
    st.sidebar.warning(
        f'⚠️ **Prediksi Jangka Panjang ({horizon_days} hari)**: '
        'Akurasi LSTM cenderung menurun untuk periode yang lebih jauh. '
        'Gunakan hasil ini sebagai panduan umum, bukan acuan definitif.'
    )
    accuracy_note = f'*Catatan: Prediksi {horizon_days} hari memiliki akurasi yang lebih rendah dibanding prediksi < 90 hari*'
else:
    accuracy_note = f'*Prediksi untuk {horizon_days} hari*'

# Additional parameters
st.sidebar.subheader('🎯 Parameter Tambahan')
manual_production = st.sidebar.number_input(
    'Produksi Tambahan (ton)',
    min_value=0.0,
    max_value=10000.0,
    value=0.0,
    step=100.0,
    key='prod_adv',
)

show_metrics = st.sidebar.checkbox('Tampilkan Metrik Detail', value=True, key='metrics_adv')
show_monthly_summary = st.sidebar.checkbox('Ringkasan Bulanan', value=True, key='monthly_adv')
compare_models = st.sidebar.checkbox('Bandingkan Model', value=False, key='compare_adv')

# Main content
try:
    # Load dataset and model
    dataset = prepare_commodity_dataset(commodity, model_mode)
    model = load_model(
        commodity,
        model_mode,
        input_size=dataset['input_size'],
        hidden_size=dataset['hidden_size'],
        num_layers=dataset['num_layers'],
        dropout=dataset['dropout'],
    )

    # Generate forecast
    forecast, lower, upper = create_forecast(model, dataset, horizon_days, manual_production)

    # Main forecast chart
    st.subheader(f'📊 Forecast {commodity} - {horizon_days} Hari')
    st.markdown(accuracy_note)
    
    figure = build_line_chart(
        dataset['history'],
        forecast,
        lower,
        upper,
        title=f'Prediksi {commodity} ({model_mode}) - {horizon_days} Hari',
    )
    st.plotly_chart(figure, use_container_width=True)

    # Forecast data table with monthly summary option
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader('📈 Detail Prediksi')
    
    with col2:
        show_table = st.checkbox('Tampilkan Tabel', value=False, key='table_adv')
    
    if show_table:
        forecast_display = forecast.copy()
        forecast_display['date'] = pd.to_datetime(forecast_display['date']).dt.strftime('%Y-%m-%d')
        forecast_display['mean'] = forecast_display['mean'].round(2)
        st.dataframe(forecast_display, use_container_width=True)

    # Monthly summary if enabled
    if show_monthly_summary and horizon_days >= 30:
        st.subheader('📅 Ringkasan Bulanan')
        
        forecast_with_month = forecast.copy()
        forecast_with_month['year_month'] = pd.to_datetime(forecast_with_month['date']).dt.to_period('M')
        
        monthly_summary = forecast_with_month.groupby('year_month').agg({
            'mean': ['min', 'max', 'mean', 'count']
        }).round(2)
        
        monthly_summary.columns = ['Min Harga', 'Max Harga', 'Rata-rata Harga', 'Jumlah Hari']
        st.dataframe(monthly_summary, use_container_width=True)

    # Model metrics
    if show_metrics:
        st.subheader('📊 Metrik Model')
        metrics = evaluate_model(model, dataset)
        
        metric_cols = st.columns(4)
        metric_cols[0].metric('MAE', f"Rp {metrics['mae']:.2f}")
        metric_cols[1].metric('RMSE', f"Rp {metrics['rmse']:.2f}")
        metric_cols[2].metric('MAPE', f"{metrics['mape']:.2f}%")
        metric_cols[3].metric('sMAPE', f"{metrics['smape']:.2f}%")
        
        st.info(
            '**Interpretasi Metrik**:\n'
            '- **MAE**: Rata-rata kesalahan absolut prediksi\n'
            '- **RMSE**: Root mean squared error (penalti lebih besar untuk error besar)\n'
            '- **MAPE**: Mean Absolute Percentage Error (kesalahan dalam %)\n'
            '- **sMAPE**: Symmetric MAPE (lebih stabil untuk nilai kecil)'
        )

    # Model comparison if enabled
    if compare_models:
        st.subheader('🔄 Perbandingan Model')
        
        alt_model = 'Multivariate' if model_mode == 'Univariate' else 'Univariate'
        
        try:
            alt_dataset = prepare_commodity_dataset(commodity, alt_model)
            alt_model_obj = load_model(
                commodity,
                alt_model,
                input_size=alt_dataset['input_size'],
                hidden_size=alt_dataset['hidden_size'],
                num_layers=alt_dataset['num_layers'],
                dropout=alt_dataset['dropout'],
            )
            
            alt_forecast, alt_lower, alt_upper = create_forecast(
                alt_model_obj, alt_dataset, horizon_days, manual_production
            )
            
            # Create comparison figure
            fig_compare = go.Figure()
            
            fig_compare.add_trace(
                go.Scatter(
                    x=forecast['date'],
                    y=forecast['mean'],
                    name=f'{model_mode}',
                    mode='lines',
                    line=dict(color='#f8d57e', width=2, dash='dot'),
                )
            )
            
            fig_compare.add_trace(
                go.Scatter(
                    x=alt_forecast['date'],
                    y=alt_forecast['mean'],
                    name=f'{alt_model}',
                    mode='lines',
                    line=dict(color='#a8d5ff', width=2, dash='dash'),
                )
            )
            
            fig_compare.update_layout(
                title=f'Perbandingan {model_mode} vs {alt_model}',
                xaxis_title='Tanggal',
                yaxis_title='Harga (Rp/kg)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#eef5ff'),
                legend=dict(bgcolor='rgba(255,255,255,0.04)'),
                hovermode='x unified',
            )
            fig_compare.update_xaxes(showgrid=False)
            fig_compare.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)')
            
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Comparison metrics
            alt_metrics = evaluate_model(alt_model_obj, alt_dataset)
            
            comparison_data = pd.DataFrame({
                model_mode: [
                    f"{metrics['mae']:.2f}",
                    f"{metrics['rmse']:.2f}",
                    f"{metrics['mape']:.2f}%",
                    f"{metrics['smape']:.2f}%",
                ],
                alt_model: [
                    f"{alt_metrics['mae']:.2f}",
                    f"{alt_metrics['rmse']:.2f}",
                    f"{alt_metrics['mape']:.2f}%",
                    f"{alt_metrics['smape']:.2f}%",
                ],
            }, index=['MAE', 'RMSE', 'MAPE', 'sMAPE'])
            
            st.dataframe(comparison_data, use_container_width=True)
            
        except Exception as e:
            st.warning(f'Tidak dapat memuat model alternatif: {e}')

    # Insights
    insight_text = generate_insights(
        history=dataset['history'],
        forecast=forecast,
        metrics={'univariate': evaluate_model(model, dataset)},
        commodity_name=commodity,
    )
    st.info(insight_text)

except Exception as exc:
    st.error(f'❌ Gagal memproses prediksi: {exc}')
    st.stop()

# Footer
st.markdown('---')
st.markdown(
    '**💡 Tips**:\n'
    '- Prediksi jangka panjang (> 90 hari) cocok untuk perencanaan strategis\n'
    '- Perbarui prediksi secara berkala saat data baru tersedia\n'
    '- Gunakan model Multivariate jika ingin mempertimbangkan faktor produksi\n'
    '- Confidence interval akan melebar untuk prediksi jangka panjang (normal)'
)
