# User Guide & Tutorial

## Daftar Isi
1. [Quick Start](#quick-start)
2. [Dashboard](#dashboard)
3. [Prediksi Lanjutan](#prediksi-lanjutan)
4. [Interpretasi Hasil](#interpretasi-hasil)
5. [Tips & Best Practices](#tips--best-practices)
6. [FAQ](#faq)

---

## Quick Start

### **Akses Aplikasi**

#### **Lokal**
```bash
conda activate rapids-24.10
streamlit run streamlit_app/app.py
# Buka http://localhost:8501
```

#### **Online** (Streamlit Cloud)
```
https://app-username-projectname.streamlit.app
```

### **First Time Setup**

1. ✅ Pilih komoditas di sidebar kiri
2. ✅ Pilih horizon prediksi (default: 30 hari)
3. ✅ Klik "Buat Prediksi" atau tunggu auto-load
4. ✅ Lihat hasil di chart dan metrik

### **Minimal Example**

```
1. Sidebar: Komoditas = "Beras Premium"
2. Sidebar: Horizon = "30 hari"
3. Sidebar: Mode = "Univariate"
4. Hasil: Chart dengan prediksi 30 hari ke depan
```

---

## Dashboard

Main page (app.py) - Untuk quick price forecasting

### **Sidebar Controls**

#### **1. Komoditas Selection**
```
Pilihan:
├─ Beras Premium
├─ Beras Medium
├─ Jagung Pipil Kering
└─ Kacang Hijau
```

**Tips**: 
- Beras Medium paling akurat
- Kacang Hijau paling volatile

#### **2. Horizon Selection**
```
Preset Options:
├─ 7 Hari (1 minggu)
├─ 14 Hari (2 minggu)
├─ 30 Hari (1 bulan)
├─ 90 Hari (3 bulan)
├─ 180 Hari (6 bulan)
├─ 365 Hari (1 tahun)
└─ Custom (1-365 hari)

⚠️ Warning: Horizon > 90 hari accuracy menurun
   Confidence interval melebar signifikan
```

**Rekomendasi**:
- Short-term (< 30 hari): Tinggi akurasi
- Medium-term (30-90 hari): Akurasi sedang
- Long-term (> 90 hari): Gunakan untuk trend saja

#### **3. Mode Selection**
```
├─ Univariate (Default)
│  └─ Hanya historical price data
│  └─ Lebih cepat
│  └─ Lebih stable untuk < 30 hari
│
└─ Multivariate
   └─ Termasuk production data
   └─ Lebih akurat untuk > 90 hari
   └─ Lebih complex
```

### **Main Display**

#### **Historical + Forecast Chart**

```
Harga (Rp/kg)
│     ╱─────── Historis (Blue)
│    ╱         Prediksi (Red)
│───┼─────────────── Confidence Band (Shaded)
│   │
│   └──────────────────> Tanggal
```

**Interpretasi**:
- 🔵 **Blue line**: Historical actual prices
- 🔴 **Red line**: Predicted future prices
- 🟦 **Shaded area**: Confidence interval (typical ±1σ)

#### **Metrik Cards**

```
┌─────────────────────────────────────┐
│ Last Price      │ Predicted Price  │
│ 14,500 Rp/kg    │ 14,625 Rp/kg     │
│ (3 hari lalu)   │ (7 hari ke depan)│
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Expected Change │ Confidence Level │
│ +125 Rp (+0.9%) │ ~95%             │
│ (trend: naik)   │ (typical range)  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Model Info      │ Device           │
│ Univariate LSTM │ GPU (CUDA)       │
│ MAE: 70.43 Rp   │ Inference: 45ms  │
└─────────────────────────────────────┘
```

### **Forecast Table**

```
Tanggal       │ Prediksi   │ Lower Bound │ Upper Bound │ Confidence
────────────────────────────────────────────────────────────────────
2024-07-01    │ 14,625 Rp  │ 14,425 Rp   │ 14,825 Rp   │ 95%
2024-07-02    │ 14,632 Rp  │ 14,418 Rp   │ 14,846 Rp   │ 95%
2024-07-03    │ 14,641 Rp  │ 14,405 Rp   │ 14,877 Rp   │ 95%
...
2024-07-30    │ 14,782 Rp  │ 14,182 Rp   │ 15,382 Rp   │ 95%
```

**Kolom Penting**:
- `Tanggal`: Tanggal forecast
- `Prediksi`: Central prediction
- `Lower/Upper Bound`: Confidence interval range
- `Confidence`: Tingkat kepercayaan (biasanya 95%)

---

## Prediksi Lanjutan

Advanced page (pages/2_Prediksi_Lanjutan.py) - Untuk detailed analysis

### **Fitur Tab 1: Model Comparison**

```
Model 1 (Univariate)  Model 2 (Multivariate)
├─ MAE: 70.43 Rp      ├─ MAE: 44.18 Rp  
├─ MAPE: 1.12%        ├─ MAPE: 0.88%
├─ Chart + Table      └─ Chart + Table
```

**Kegunaan**:
- Compare akurasi 2 model
- Pick model terbaik untuk scenario
- Lihat perbedaan prediksi

### **Fitur Tab 2: Detailed Analysis**

#### **Chart dengan Multiple Metrics**
- Historical + Forecast
- Error bars per hari
- Confidence intervals adaptive

#### **Metrics Breakdown**
```
Metrik             Nilai      Interpretasi
─────────────────────────────────────────────
MAE                44.18 Rp   Rata-rata error
RMSE               55.42 Rp   Penalti error besar
MAPE               0.88%      % error dari actual
sMAPE              0.85%      Symmetric %
Direction Acc.     92%        Prediksi up/down
```

### **Fitur Tab 3: Monthly Summary**

```
Bulan          Min Price  Max Price  Avg Price  Prediksi
──────────────────────────────────────────────────────────
Juli 2024      14,200     14,800     14,500     14,625
Agustus 2024   14,300     14,900     14,600     14,750
September 2024 14,100     14,700     14,400     14,500
```

**Kegunaan**:
- Overview bulanan
- Planning procurement
- Seasonal pattern visibility

### **Fitur Tab 4: Export Data**

```
┌────────────────────────────────┐
│ Download Forecast as CSV       │
│ ┌──────────────────────────┐   │
│ │ Download Button          │   │
│ └──────────────────────────┘   │
└────────────────────────────────┘
```

**Format CSV**:
```
date,prediction,lower_bound,upper_bound,confidence
2024-07-01,14625,14425,14825,0.95
2024-07-02,14632,14418,14846,0.95
...
```

---

## Interpretasi Hasil

### **Reading the Chart**

#### **Normal Forecast** ✅
```
Harga trend naik
│
├─ Blue line smooth (historical data)
├─ Red line smooth (predicted trend)
├─ Confidence band reasonable (< 10% dari price)
└─ No sudden jumps atau gaps

Interpretation: High confidence, dapat digunakan
```

#### **High Uncertainty** ⚠️
```
Harga dengan volatile
│
├─ Red line flatter (less confident)
├─ Confidence band sangat lebar
├─ Multiple possible outcomes
└─ Horizon > 90 hari

Interpretation: Use untuk trend saja, bukan point estimate
```

#### **Potential Issues** ❌
```
Model perlu retraining
│
├─ Red line tidak masuk akal
├─ Sudden jump di prediksi
├─ Model diverging dari historis
└─ Device mismatch atau data error

Action: Restart app, check data quality
```

### **Confidence Interval Interpretation**

#### **Narrow Band** (< 5% dari price)
```
Contoh: 14,500 ± 500 Rp (3% bandwidth)

Interpretasi: 
✅ High confidence
✅ Dapat untuk tactical decisions
✅ Likely short horizon (< 30 hari)
```

#### **Medium Band** (5-10%)
```
Contoh: 14,500 ± 1,200 Rp (8% bandwidth)

Interpretasi:
⚠️ Medium confidence
⚠️ Use untuk direction, not points
⚠️ Likely 30-90 hari horizon
```

#### **Wide Band** (> 10%)
```
Contoh: 14,500 ± 2,200 Rp (15% bandwidth)

Interpretasi:
❌ Low confidence
❌ Use untuk trend/direction saja
❌ Likely > 90 hari horizon
❌ Consider scenario planning instead
```

### **Direction Accuracy**

**Penting**: Model bisa salah dalam magnitude tapi benar direction

```
Actual:    14,500 → 15,000 (naik 3.4%)
Predicted: 14,500 → 14,700 (naik 1.4%)

Evaluation:
✅ Direction: Correct (naik)
❌ Magnitude: Underestimate 50%

Usage: Still valuable untuk "naik/turun" signal
```

---

## Tips & Best Practices

### **Short-term Forecasting (< 30 hari)**

✅ **DO**:
1. Use Univariate model (lebih stable)
2. Trust point estimate (confidence interval narrow)
3. Update prediction setiap minggu
4. Combine dengan market sentiment

❌ **DON'T**:
1. Use extreme confidence levels
2. Ignore structural breaks
3. Over-rely pada single model
4. Forecast > 30 hari dengan short term settings

### **Medium-term Forecasting (30-90 hari)**

✅ **DO**:
1. Ensemble Univariate + Multivariate
2. Focus pada direction, not magnitude
3. Use widened confidence bands
4. Incorporate seasonal patterns

❌ **DON'T**:
1. Trade pada point estimate
2. Ignore confidence widening
3. Update terlalu frequent (weekly OK)
4. Use single model exclusively

### **Long-term Forecasting (> 90 hari)**

✅ **DO**:
1. Use Multivariate model (incorporate supply)
2. Focus pada trend, ignore noise
3. Widen confidence bands manually (×2)
4. Scenario planning approach

❌ **DON'T**:
1. Trust any point estimate
2. Use untuk tactical trading
3. Ignore external factors
4. Forecast > 365 hari

### **Model Monitoring**

**Weekly Checklist**:
```
□ Compare actual vs predicted (last 7 days)
□ Check prediction direction accuracy
□ Monitor confidence interval widths
□ Look for unusual patterns
□ Verify data freshness
```

**Monthly Checklist**:
```
□ Recalculate MAE/RMSE metrics
□ Compare vs baseline (moving average)
□ Retrain jika possible
□ Document performance in log
□ Plan any adjustments
```

---

## FAQ

### **General Questions**

**Q: Berapa akurat model?**
```
A: Tergantung horizon:
   - 7 hari: ~99% accuracy (MAE < 2%)
   - 30 hari: ~95% accuracy (MAE < 2%)
   - 90 hari: ~85% accuracy (MAE ~2-3%)
   - 365 hari: ~50% accuracy (trend only)
```

**Q: Apa maksimal forecast horizon?**
```
A: Technically 365 hari, tapi:
   - < 90 hari: Tinggi confidence
   - 90-180 hari: Confidence sedang
   - > 180 hari: Hanya untuk trend
   - > 365 hari: Tidak recommended
```

**Q: Bagaimana update prediksi?**
```
A: Otomatis setiap kali app diload
   Manual update: F5 atau refresh button
   New training: Monthly atau saat major change
```

### **Technical Questions**

**Q: Model mana yang lebih bagus?**
```
A: Tergantung kasus:
   - Univariate: Short-term (<30 hari)
   - Multivariate: Long-term (>90 hari)
   - Ensemble: Best accuracy (kedua)
```

**Q: Apa CPU vs GPU difference?**
```
A: GPU jauh lebih cepat:
   - GPU: 45ms per inference
   - CPU: 500ms per inference
   
   Trade-off: GPU needs NVIDIA hardware
```

**Q: Bisakah model memprediksi sudden spike?**
```
A: Tidak. Model smooth trend, tidak capture:
   - Sudden supply shock
   - Major policy change
   - Crisis events
   - Extreme outliers
   
   Saran: Kombina dengan market monitoring
```

### **Interpretation Questions**

**Q: Apakah confidence interval = probability?**
```
A: Tidak. Confidence interval adalah:
   - Statistical range (95% confidence)
   - NOT probability distribusi
   - Wider bands = more uncertainty
   
   Interpretation: Price LIKELY dalam range ini
```

**Q: Apa artinya negative predicted price?**
```
A: Tidak terjadi dalam praktik karena:
   - Prices always positive (min 0)
   - Model trained dengan positive prices
   - Confidence lower bound tidak go negative
   
   Jika terjadi: Data quality issue
```

**Q: Bagaimana handle forecast uncertainty?**
```
A: Beberapa strategi:

   1. Conservative: Use upper bound untuk budget
   2. Aggressive: Use lower bound untuk opportunity
   3. Scenario: Multiple forecasts (best/worst/base)
   4. Ensemble: Combine multiple models
```

### **Operational Questions**

**Q: Bagaimana jika model error besar?**
```
A: Debugging steps:
   1. Check data quality (nulls, outliers)
   2. Verify model loading success
   3. Compare vs baseline (simple MA)
   4. Check for structural breaks
   5. Consider retraining
```

**Q: Dapatkah saya modify predictions?**
```
A: Tidak directly, tapi:
   - Override dengan business rules
   - Add safety margin to bounds
   - Adjust confidence untuk conservatism
   - Document overrides untuk audit trail
```

**Q: Bagaimana integrate dengan system lain?**
```
A: API tidak tersedia, tapi:
   1. Export CSV dari advanced page
   2. Schedule daily download
   3. Parse CSV ke database
   4. Build custom integrations
   
   Alternative: Contact team untuk API access
```

### **Performance Questions**

**Q: Kenapa app slow sometimes?**
```
A: Possible causes:
   - GPU busy (other processes)
   - Model not cached
   - Large forecast horizon
   - Network latency (cloud deployment)
   
   Solution:
   - Close other apps
   - Use shorter horizon for speed
   - Check system resources
   - Run locally if possible
```

**Q: Berapa cost untuk cloud deployment?**
```
A: Streamlit Cloud: FREE tier available
   - Limited compute
   - No GPU support (slower inference)
   - Perfect untuk < 100 daily users
   
   Paid tiers:
   - $5-50/month untuk higher performance
   - GPU instance: ~$1-5 per hour additional
```

---

## Support & Resources

### **Documentation**
- [Architecture](01_Architecture.md) - System design
- [Data Dictionary](02_Data_Dictionary.md) - Data specs
- [Model Guide](03_Model.md) - Model details
- [Results](04_Results.md) - Performance metrics
- [Setup](05_Setup.md) - Installation guide

### **Troubleshooting**
- Check error messages di app
- Review browser console (F12)
- Check terminal untuk logs
- Run tests: `pytest tests/`

### **Contributing**
- Report bugs ke project issues
- Suggest improvements
- Submit pull requests
- Share results & feedback

---

**Document Version**: 1.0  
**Last Updated**: June 2026  
**Support Contact**: [project-team-email]
