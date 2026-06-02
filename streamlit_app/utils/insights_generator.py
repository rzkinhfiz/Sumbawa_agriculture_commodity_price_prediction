from typing import Dict


def generate_insights(history, forecast, metrics: Dict, commodity_name: str) -> str:
    last_price = float(history['target'].iloc[-1])
    forecast_mean = float(forecast['mean'].mean())
    change_pct = 0.0 if last_price == 0 else (forecast_mean - last_price) / last_price * 100
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

    univariate = metrics.get('univariate', {})
    multivariate = metrics.get('multivariate', {})
    if univariate and multivariate:
        diff = abs(univariate.get('mae', 0) - multivariate.get('mae', 0))
        if diff >= 5:
            insights.append('Perbedaan performa antara univariate dan multivariate besar; analisis fitur eksogen tambahan direkomendasikan.')

    if univariate.get('mape', 0) > 15 or multivariate.get('mape', 0) > 15:
        insights.append('Model memiliki error relatif tinggi; lakukan validasi data input dan evaluasi ulang preprocessing.')

    return ' '.join(insights)
