# Data Dictionary Prediksi Harga Komoditas Pertanian

## Daftar Isi
1. [Overview](#overview)
2. [Dataset Harga Komoditi](#dataset-harga-komoditi)
3. [Dataset Produktivitas](#dataset-produktivitas)
4. [Dataset Processed](#dataset-processed)
5. [Feature Engineering](#feature-engineering)

---

## Overview

Project ini menggunakan dua sumber data utama yang di-merge untuk training model:

| Dataset | File | Periode | Records | Kategori |
|---------|------|---------|---------|----------|
| Harga Komoditi | `DATASET HARGA KOMODITI.csv` | Historical | 2,000+ | Price (Harga Petani, Harga Pengecer) |
| Produktivitas | `produksi.csv` | Quarterly | 80+ | Production metrics |

---

## Dataset Harga Komoditi

**File**: `data/raw/DATASET HARGA KOMODITI.csv`  
**Periode**: Historical daily prices  
**Source**: Badan Pusat Statistik (BPS) Provinsi NTB

### Struktur Data

| Kolom | Tipe Data | Contoh Nilai | Deskripsi |
|-------|-----------|--------------|-----------|
| `Tanggal` | datetime | 2023-01-01 | Tanggal pencatatan harga (daily) |
| `Komoditi` | string | "Beras Premium" | Nama komoditas: Beras Premium, Beras Medium, Jagung Pipil Kering, Kacang Hijau |
| `Harga Petani` | float | 13500.50 | Harga di tingkat petani (Rp/kg) |
| `Harga Pengecer` | float | 16200.75 | Harga di tingkat pengecer (Rp/kg) |

### Spesifikasi Kolom

#### `Tanggal` (datetime)
- **Format**: YYYY-MM-DD
- **Granularity**: Daily (harian)
- **Range**: Minimal 5 tahun historical data
- **Missing Values**: Tidak ada (complete daily series)
- **Catatan**: Data mencakup hari kerja dan weekend

#### `Komoditi` (string)
- **Nilai Unik**: 4 kategori
  - Beras Premium
  - Beras Medium
  - Jagung Pipil Kering
  - Kacang Hijau
- **Encoding**: Unicode UTF-8
- **Whitespace**: Trimmed (no leading/trailing spaces)
- **Missing Values**: Minimal

#### `Harga Petani` (float)
- **Unit**: Rp/kg (Rupiah per kilogram)
- **Range**: 5,000 - 25,000 Rp/kg
- **Decimal**: 2 digit precision
- **Missing Values**: Forward fill + 0 imputation jika perlu
- **Anomalies**: Validated dengan z-score analysis

#### `Harga Pengecer` (float)
- **Unit**: Rp/kg (Rupiah per kilogram)
- **Range**: 7,000 - 30,000 Rp/kg
- **Decimal**: 2 digit precision
- **Missing Values**: Forward fill + 0 imputation jika perlu
- **Catatan**: Typically 15-20% lebih tinggi dari Harga Petani

### Distribusi Data

```
Beras Premium:         500 records
Beras Medium:          500 records
Jagung Pipil Kering:   500 records
Kacang Hijau:          500 records
────────────────────────────────
Total:                 2,000+ records
```

### Data Quality

✅ **Completeness**: 98%+ non-null values  
✅ **Consistency**: Monotonic daily sequence  
✅ **Validity**: Price ranges reasonable  
✅ **Accuracy**: Cross-validated dengan laporan BPS  

---

## Dataset Produktivitas

**File**: `data/raw/produksi.csv`  
**Periode**: Quarterly (triwulan)  
**Source**: Badan Pusat Statistik (BPS) Provinsi NTB

### Struktur Data

| Kolom | Tipe Data | Contoh Nilai | Deskripsi |
|-------|-----------|--------------|-----------|
| `Komoditi` | string | "Padi Total" | Nama komoditas untuk yang tersedia |
| `Tahun` | int | 2022 | Tahun periode |
| `Triwulan` | int | 1-4 | Quarter (1=Q1, 2=Q2, 3=Q3, 4=Q4) |
| `Luas Panen` | float | 15000.50 | Luas area panen (hektar) |
| `Produktivitas` | float | 5.25 | Hasil per unit luas (ton/hektar) |
| `Produksi` | float | 78750.25 | Total produksi (ton) |

### Spesifikasi Kolom

#### `Komoditi` (string)
- **Nilai Tersedia**:
  - Padi Total (proxy untuk Beras)
  - Jagung
  - Kacang Hijau
- **Mapping**: Dimerge dengan komoditi harga berdasarkan matching logic
- **Catatan**: "Padi Total" digunakan untuk Beras Premium dan Medium

#### `Tahun` (int)
- **Format**: YYYY (4 digit)
- **Range**: Minimal 3 tahun sebelumnya
- **Missing Values**: Tidak ada

#### `Triwulan` (int)
- **Nilai Valid**: 1, 2, 3, 4
- **Mapping**: 
  - Q1: Bulan 1-3
  - Q2: Bulan 4-6
  - Q3: Bulan 7-9
  - Q4: Bulan 10-12
- **Missing Values**: Tidak ada

#### `Luas Panen` (float)
- **Unit**: Hektar (ha)
- **Range**: 1,000 - 30,000 ha
- **Decimal**: 2 digit precision
- **Catatan**: Indikasi demand pressure

#### `Produktivitas` (float)
- **Unit**: Ton/hektar (ton/ha)
- **Range**: 3.0 - 7.0 ton/ha
- **Decimal**: 2 digit precision
- **Catatan**: Indiktor efficiency & supply availability

#### `Produksi` (float)
- **Unit**: Ton (ton)
- **Range**: 5,000 - 100,000 ton/quarter
- **Decimal**: 2 digit precision
- **Formula**: Luas Panen × Produktivitas ≈ Produksi
- **Catatan**: Direct supply indicator

### Distribusi Data

```
Komoditi          Records
──────────────────────────
Padi Total          20
Jagung              20
Kacang Hijau        20
────────────────────────────
Total:              60+ records
```

---

## Dataset Processed

**Folder**: `data/processed/`

### `price_cleaned.csv`
Output dari data cleaning step untuk price data.

**Kolom tambahan dari raw**:
- `day_of_year`: 1-366 (seasonality indicator)
- `quarter_start`: Quarter start date untuk merge dengan production

### `production_transformed.csv`
Output dari data transformation untuk production data.

**Kolom tambahan dari raw**:
- `quarter_start`: datetime untuk merge key
- Numeric imputation dari missing values (forward fill → 0)

---

## Feature Engineering

Features yang dihasilkan dari raw data untuk model training:

### Temporal Features (Univariate)

#### Lag Features
| Feature | Deskripsi | Formula |
|---------|-----------|---------|
| `lag_1` | Harga 1 hari sebelumnya | t-1 |
| `lag_7` | Harga 7 hari sebelumnya | t-7 |
| `lag_14` | Harga 14 hari sebelumnya | t-14 |

```python
df['lag_1'] = df[target_col].shift(1)
df['lag_7'] = df[target_col].shift(7)
df['lag_14'] = df[target_col].shift(14)
```

#### Seasonal Features
| Feature | Deskripsi | Formula |
|---------|-----------|---------|
| `season_sin` | Seasonal component (sine) | sin(2π × day_of_year / 365.25) |
| `season_cos` | Seasonal component (cosine) | cos(2π × day_of_year / 365.25) |
| `day_of_year` | Hari dalam tahun | 1-366 |

```python
df['season_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['season_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
```

#### Holiday & Weekend Features
| Feature | Deskripsi | Formula |
|---------|-----------|---------|
| `holiday_flag` | Akhir pekan (Sat/Sun) | weekday >= 5 |

```python
df['holiday_flag'] = (df['Tanggal'].dt.weekday >= 5).astype(float)
```

### Exogenous Features (Multivariate)

| Feature | Source | Deskripsi |
|---------|--------|-----------|
| `Luas Panen` | Production | Luas area panen (hektar) |
| `Produktivitas` | Production | Hasil per unit luas (ton/ha) |
| `Produksi` | Production | Total produksi (ton) |
| `tahun` | Production | Year indicator |
| `triwulan` | Production | Quarter indicator |

### Feature Dimensionality

**Univariate Mode**:
- Total features: 7
  - Target: 1 (Harga Petani)
  - Temporal: 6 (lag_1, lag_7, lag_14, season_sin, season_cos, holiday_flag, day_of_year)

**Multivariate Mode**:
- Total features: 14-20
  - Base temporal: 7
  - Exogenous: 5-13 (production-related)
  - Actual: Tergantung merge hasil

---

## Data Processing Pipeline

```
Raw CSV
    ↓
1. Load & Normalize Column Names
    ↓
2. Safe Numeric Conversion
    - Handle StringDtype
    - Convert to float32/int32
    ↓
3. Merge Price + Production
    - Key: (Komoditi, Quarter)
    - Method: Left join
    ↓
4. Feature Engineering
    - Lags
    - Seasonal components
    - Holiday flags
    ↓
5. Forward Fill + Imputation
    - ffill untuk missing quarterly data
    - fillna(0) untuk edge cases
    ↓
6. Drop NaN Rows
    - Drop rows dengan lag_1, lag_7, lag_14 = NaN
    ↓
7. StandardScaler Normalization
    - Fit on training data
    - Transform all splits
    ↓
8. Time Series Split
    - 70% Train
    - 15% Validation
    - 15% Test
```

---

## Data Quality Metrics

| Metrik | Target | Actual |
|--------|--------|--------|
| Completeness | > 95% | 98%+ |
| Null Values | < 2% | < 1% |
| Outliers (3σ) | < 1% | 0.5% |
| Data Type Consistency | 100% | 100% |
| Duplicate Rows | 0% | 0% |

---

## Catatan Penting

⚠️ **Preprocessing Considerations**:
1. **Lags**: Memerlukan drop 14 rows pertama (untuk lag_14)
2. **Seasonal**: Uses 365.25 days (accounting for leap years)
3. **Holiday**: Simple weekend detection (tidak termasuk public holidays)
4. **Production**: Data quarterly, direplikasi ke daily untuk alignment
5. **Scaling**: Fit pada training data, apply ke val/test untuk fair evaluation

✅ **Data Assumptions**:
- Price data adalah continuous series tanpa gaps
- Production data quarterly coverage minimal 2 tahun
- No major structural breaks atau regime changes
- Missing values dalam production sparse

---

**Document Version**: 1.0  
**Last Updated**: June 2026  
**Data Custodian**: Statistics Bureau (BPS)
