from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CONFIG_PATH = REPO_ROOT / 'config.yaml'
PRICE_PATH = REPO_ROOT / 'data' / 'processed' / 'price_cleaned.csv'
PRODUCTION_PATH = REPO_ROOT / 'data' / 'processed' / 'production_transformed.csv'


def _read_csv_safe(filepath: Path, **kwargs) -> pd.DataFrame:
    """Read CSV and ensure no StringDtype columns remain."""
    # Use dtype_backend to avoid StringDtype from the start
    try:
        df = pd.read_csv(filepath, dtype_backend='numpy', **kwargs)
    except (TypeError, ValueError):
        # Fallback if dtype_backend not supported
        df = pd.read_csv(filepath, **kwargs)
    
    # Ensure all numeric columns are properly typed
    df = _safe_numeric_conversion(df)
    return df


def _safe_numeric_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame columns to numeric types, handling StringDtype."""
    df = df.copy()
    # Skip columns that shouldn't be converted
    skip_cols = {'Tanggal', 'Komoditi', 'quarter_start'}
    
    # Convert StringDtype to regular object, then to numeric
    for col in df.columns:
        if col in skip_cols:
            continue
        # Check if it's already a datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        # Try to convert StringDtype
        if hasattr(df[col].dtype, 'name') and 'string' in str(df[col].dtype).lower():
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Try to convert other non-numeric types
        elif not np.issubdtype(df[col].dtype, np.number):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except (TypeError, ValueError):
                pass
    return df


@st.cache_data
def load_config() -> Dict:
    import yaml

    with open(CONFIG_PATH, 'r', encoding='utf-8') as buffer:
        return yaml.safe_load(buffer)


@st.cache_data
def load_price_data() -> pd.DataFrame:
    df = _read_csv_safe(PRICE_PATH, parse_dates=['Tanggal'])
    df.columns = df.columns.str.strip()
    df = df.sort_values(['Komoditi', 'Tanggal']).reset_index(drop=True)
    # Ensure numeric conversion to avoid StringDtype issues
    df = _safe_numeric_conversion(df)
    return df


def add_time_series_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df['lag_1'] = df[target_col].shift(1)
    df['lag_7'] = df[target_col].shift(7)
    df['lag_14'] = df[target_col].shift(14)
    df['day_of_year'] = df['Tanggal'].dt.dayofyear
    df['season_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['season_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['holiday_flag'] = (df['Tanggal'].dt.weekday >= 5).astype(float)
    return df


def get_commodity_names() -> List[str]:
    config = load_config()
    return [commodity['name'] for commodity in config['commodities']]


def _get_commodity_info(config: Dict, commodity_name: str) -> Dict:
    for commodity in config['commodities']:
        if commodity['name'] == commodity_name:
            return commodity
    raise ValueError(f'Komoditas tidak ditemukan: {commodity_name}')


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def _guess_production_columns(prod: pd.DataFrame) -> Dict[str, str]:
    columns = [col.strip() for col in prod.columns]
    lower_cols = [col.lower() for col in columns]
    commodity_col = None
    quarter_col = None

    if 'komoditi_harga' in lower_cols:
        commodity_col = columns[lower_cols.index('komoditi_harga')]
    elif 'komoditi' in lower_cols:
        commodity_col = columns[lower_cols.index('komoditi')]
    elif 'commodity' in lower_cols:
        commodity_col = columns[lower_cols.index('commodity')]

    if 'quarter_start' in lower_cols:
        quarter_col = columns[lower_cols.index('quarter_start')]
    else:
        quarter_candidates = [col for col in columns if 'quarter' in col.lower() or 'triwulan' in col.lower()]
        quarter_col = quarter_candidates[0] if quarter_candidates else None

    return {'commodity_col': commodity_col, 'quarter_col': quarter_col}


def _merge_production_features(df: pd.DataFrame, commodity_info: Dict) -> pd.DataFrame:
    prod = _read_csv_safe(PRODUCTION_PATH)
    prod = _normalize_column_names(prod)
    # Apply safe numeric conversion to production data immediately after load
    prod = _safe_numeric_conversion(prod)
    prod_cols = _guess_production_columns(prod)
    commodity_key = prod_cols['commodity_col']
    quarter_col = prod_cols['quarter_col']

    if commodity_key is None or quarter_col is None:
        return df

    prod[commodity_key] = prod[commodity_key].astype(str).str.strip()
    prod[quarter_col] = pd.to_datetime(prod[quarter_col], errors='coerce')
    df = df.copy()
    df['quarter_start'] = df['Tanggal'].dt.to_period('Q').dt.to_timestamp()
    if commodity_key.lower() == 'komoditi_harga':
        merge_left = ['Komoditi', 'quarter_start']
        merge_right = [commodity_key, quarter_col]
    else:
        merge_left = ['Komoditi', 'quarter_start']
        merge_right = [commodity_key, quarter_col]

    merged = df.merge(
        prod,
        left_on=merge_left,
        right_on=merge_right,
        how='left',
        suffixes=('', '_prod'),
    )

    numeric_cols = [col for col in merged.columns if col not in ['Tanggal', 'Komoditi', commodity_info['target_column'], 'quarter_start'] and np.issubdtype(merged[col].dtype, np.number)]
    merged[numeric_cols] = merged[numeric_cols].ffill().fillna(0.0)
    # Ensure numeric conversion after merge to avoid StringDtype issues
    merged = _safe_numeric_conversion(merged)
    return merged


@st.cache_data
def prepare_commodity_dataset(commodity_name: str, mode: str) -> Dict:
    config = load_config()
    commodity_info = _get_commodity_info(config, commodity_name)
    price_df = load_price_data()
    df = price_df[price_df['Komoditi'] == commodity_info['series_id']].copy()
    if df.empty:
        raise ValueError(f'Tidak ada data untuk komoditas {commodity_name}')

    df = df.sort_values('Tanggal').reset_index(drop=True)
    # Ensure numeric conversion early in the pipeline
    df = _safe_numeric_conversion(df)
    df = add_time_series_features(df, commodity_info['target_column'])
    df = _safe_numeric_conversion(df)
    df = df.dropna(subset=['lag_1', 'lag_7', 'lag_14']).reset_index(drop=True)

    temporal_cols = [commodity_info['target_column'], 'lag_1', 'lag_7', 'lag_14', 'season_sin', 'season_cos', 'holiday_flag']
    feature_df = df.copy()
    production_df = None
    production_key = None
    quarter_col = None

    if mode.lower() == 'multivariate':
        try:
            feature_df = _merge_production_features(feature_df, commodity_info)
            # Ensure conversion after production merge
            feature_df = _safe_numeric_conversion(feature_df)
            prod = _read_csv_safe(PRODUCTION_PATH)
            prod = _normalize_column_names(prod)
            # Apply safe numeric conversion to production data immediately after load
            prod = _safe_numeric_conversion(prod)
            prod_cols = _guess_production_columns(prod)
            production_df = prod
            production_key = prod_cols['commodity_col']
            quarter_col = prod_cols['quarter_col']
        except FileNotFoundError:
            production_df = None

    feature_cols = temporal_cols[:]
    if mode.lower() == 'multivariate':
        extra_cols = [
            col for col in feature_df.columns
            if col not in ['Tanggal', 'Komoditi', commodity_info['target_column'], 'quarter_start']
            and col not in temporal_cols
            and np.issubdtype(feature_df[col].dtype, np.number)
        ]
        feature_cols.extend(sorted(extra_cols))

    feature_df = feature_df[['Tanggal', 'Komoditi'] + feature_cols].copy()
    feature_df = feature_df.dropna().reset_index(drop=True)
    
    # Safe conversion to numeric
    feature_df_numeric = _safe_numeric_conversion(feature_df)

    feature_matrix = feature_df_numeric[feature_cols].astype(float).to_numpy()
    target_array = feature_df_numeric[commodity_info['target_column']].astype(float).to_numpy()

    scaler_x = StandardScaler().fit(feature_matrix)
    scaler_y = StandardScaler().fit(target_array.reshape(-1, 1))

    model_cfg = config['model'][mode.lower()]
    seq_len = int(model_cfg.get('sequence_length', 28))
    hidden_size = int(model_cfg.get('hidden_size', 64))
    num_layers = int(model_cfg.get('num_layers', 2))
    dropout = float(model_cfg.get('dropout', 0.2))

    history = feature_df[['Tanggal', commodity_info['target_column']]].rename(columns={commodity_info['target_column']: 'target'})

    # Ensure production_df is safe before storing in payload
    if production_df is not None:
        production_df = _safe_numeric_conversion(production_df)

    return {
        'history': history,
        'feature_df': feature_df,
        'feature_cols': feature_cols,
        'seq_len': seq_len,
        'input_size': len(feature_cols),
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'target_col': commodity_info['target_column'],
        'mode': mode.lower(),
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'commodity_info': commodity_info,
        'production_df': production_df,
        'production_key': production_key,
        'quarter_col': quarter_col,
    }


def _forecast_exogenous_row(date: pd.Timestamp, payload: Dict) -> Dict[str, float]:
    if payload['production_df'] is None or payload['production_key'] is None or payload['quarter_col'] is None:
        return {}

    quarter_start = date.to_period('Q').to_timestamp()
    production = payload['production_df'].copy()
    # Ensure numeric safety after copying
    production = _safe_numeric_conversion(production)
    commodity_name = payload['commodity_info']['series_id']
    production[payload['production_key']] = production[payload['production_key']].astype(str).str.strip()

    if payload['production_key'].lower() == 'komoditi_harga':
        match_mask = production[payload['production_key']] == commodity_name
    else:
        match_mask = production[payload['production_key']].astype(str).str.lower() == commodity_name.lower()

    production = production[match_mask]
    if production.empty:
        return {}
    
    # Ensure numeric safety after filtering
    production = _safe_numeric_conversion(production)

    production[payload['quarter_col']] = pd.to_datetime(production[payload['quarter_col']], errors='coerce')
    exact_match = production[production[payload['quarter_col']] == quarter_start]
    if not exact_match.empty:
        row = exact_match.iloc[0]
    else:
        earlier = production[production[payload['quarter_col']] < quarter_start]
        if not earlier.empty:
            row = earlier.sort_values(payload['quarter_col']).iloc[-1]
        else:
            row = production.sort_values(payload['quarter_col']).iloc[-1]

    output = {
        col: float(row[col])
        for col in production.columns
        if col not in [payload['production_key'], payload['quarter_col'], 'Komoditi', 'Komoditi_Harga']
        and np.issubdtype(row[col].dtype, np.number)
    }
    return output


def create_forecast(model, payload: Dict, horizon: int, manual_feature: float = 0.0):
    data = payload['feature_df'].copy()
    # Ensure numeric conversion
    data = _safe_numeric_conversion(data)
    feature_cols = payload['feature_cols']
    seq_len = payload['seq_len']
    scaler_x = payload['scaler_x']
    scaler_y = payload['scaler_y']
    mode = payload['mode']

    if len(data) < seq_len:
        raise ValueError('Data tidak cukup untuk membuat forecast. Pastikan dataset komoditas sudah diproses.')

    predictions = []
    forecast_dates = []
    last_date = data['Tanggal'].iloc[-1]

    for step in range(horizon):
        sequence = data[feature_cols].iloc[-seq_len:].astype(float).to_numpy()
        scaled_seq = scaler_x.transform(sequence)
        input_tensor = np.expand_dims(scaled_seq, axis=0)

        with torch.no_grad():
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
            output = model(input_tensor).cpu().numpy().ravel()

        predicted_scaled = float(output[0])
        predicted = float(scaler_y.inverse_transform([[predicted_scaled]])[0, 0])
        next_date = last_date + pd.Timedelta(days=1)

        next_row = {
            payload['target_col']: predicted,
            'Tanggal': next_date,
            'Komoditi': payload['commodity_info']['series_id'],
            'lag_1': float(data[payload['target_col']].iloc[-1]),
            'lag_7': float(data[payload['target_col']].iloc[-7]) if len(data) >= 7 else float(data[payload['target_col']].iloc[-1]),
            'lag_14': float(data[payload['target_col']].iloc[-14]) if len(data) >= 14 else float(data[payload['target_col']].iloc[-1]),
            'day_of_year': float(next_date.dayofyear),
            'season_sin': float(np.sin(2 * np.pi * next_date.dayofyear / 365.25)),
            'season_cos': float(np.cos(2 * np.pi * next_date.dayofyear / 365.25)),
            'holiday_flag': float(next_date.weekday() >= 5),
        }

        if mode == 'multivariate':
            exog_values = _forecast_exogenous_row(next_date, payload)
            if exog_values:
                if manual_feature and 'Produksi' in exog_values:
                    exog_values['Produksi'] = float(exog_values['Produksi'] + manual_feature)
                next_row.update(exog_values)

        for feature_col in feature_cols:
            if feature_col not in next_row:
                next_row[feature_col] = 0.0

        data = pd.concat([data, pd.DataFrame([next_row])], ignore_index=True, axis=0)
        forecast_dates.append(next_date)
        predictions.append(predicted)
        last_date = next_date

    predictions = np.array(predictions, dtype=float)
    if manual_feature != 0 and mode != 'multivariate':
        adjustment = np.tanh(float(manual_feature) / 1000.0) * 0.03
        predictions = predictions * (1.0 + adjustment)

    width = np.maximum(np.abs(predictions) * 0.06, 50.0)
    lower = np.maximum(predictions - width, 0.0)
    upper = predictions + width

    forecast_frame = pd.DataFrame({'date': forecast_dates, 'mean': predictions})
    return forecast_frame, lower, upper


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-8
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_pred - y_true) / y_true_safe)) * 100.0)


def evaluate_model(model, payload: Dict, max_points: int = 200) -> Dict[str, float]:
    feature_df_numeric = _safe_numeric_conversion(payload['feature_df'])
    feature_matrix = feature_df_numeric[payload['feature_cols']].astype(float).to_numpy()
    target_values = feature_df_numeric[payload['target_col']].astype(float).to_numpy()
    seq_len = payload['seq_len']
    scaler_x = payload['scaler_x']
    scaler_y = payload['scaler_y']

    if len(feature_matrix) <= seq_len:
        return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'smape': 0.0}

    n_points = len(feature_matrix) - seq_len
    start_index = max(0, n_points - max_points)
    indices = range(start_index, n_points)
    sequences = np.stack([feature_matrix[i:i + seq_len] for i in indices], axis=0)
    scaled_sequences = scaler_x.transform(sequences.reshape(-1, payload['input_size'])).reshape(-1, seq_len, payload['input_size'])

    with torch.no_grad():
        input_tensor = torch.tensor(scaled_sequences, dtype=torch.float32)
        outputs = model(input_tensor).cpu().numpy().ravel()

    predicted = scaler_y.inverse_transform(outputs.reshape(-1, 1)).ravel()
    actual = target_values[seq_len + start_index: seq_len + start_index + len(predicted)]

    mae = float(mean_absolute_error(actual, predicted))
    mse = mean_squared_error(actual, predicted)
    rmse = float(np.sqrt(mse))
    mape = _mape(actual, predicted)
    smape = _smape(actual, predicted)

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
    }
