import argparse
import yaml
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import csv
from torch.utils.tensorboard import SummaryWriter

# Ensure repo root is available for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class PriceSequenceDataset(Dataset):
    """Sequence dataset that works for univariate (single feature) or multivariate inputs.

    features: 2D numpy array (n_samples, n_features)
    targets: 1D numpy array (n_samples,)
    scalers: scaler_x (for features), scaler_y (for target)
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int, scaler_x: StandardScaler, scaler_y: StandardScaler):
        self.seq_len = seq_len
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.features = self.scaler_x.transform(features)
        self.targets = self.scaler_y.transform(targets.reshape(-1, 1)).ravel()

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]  # (seq_len, n_features)
        y = self.targets[idx + self.seq_len]  # scalar (scaled)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def add_time_series_features(df, target_col):
    df = df.copy()
    df = df.sort_values(['Komoditi', 'Tanggal']).reset_index(drop=True)
    df['lag_1'] = df.groupby('Komoditi')[target_col].shift(1)
    df['lag_7'] = df.groupby('Komoditi')[target_col].shift(7)
    df['lag_14'] = df.groupby('Komoditi')[target_col].shift(14)
    df['day_of_year'] = df['Tanggal'].dt.dayofyear
    df['season_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['season_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['holiday_flag'] = (df['Tanggal'].dt.weekday >= 5).astype(float)
    return df


def split_time_series_dataset(dataset, val_ratio: float, forecast_horizon: int):
    n = len(dataset)
    if forecast_horizon >= n // 3:
        forecast_horizon = max(1, n // 5)
    test_n = min(forecast_horizon, max(1, n // 10))
    train_val_n = n - test_n
    val_n = max(1, int(train_val_n * val_ratio))
    train_n = train_val_n - val_n
    train_idx = list(range(train_n))
    val_idx = list(range(train_n, train_n + val_n))
    test_idx = list(range(train_n + val_n, n))
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom == 0
    denom[mask] = 1e-8
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def mape(y_true, y_pred):
    mask = y_true == 0
    y_true_safe = y_true.copy()
    y_true_safe[mask] = 1e-8
    return 100.0 * np.mean(np.abs((y_pred - y_true) / y_true_safe))


def directional_accuracy(y_true, y_pred, last_actual):
    dir_true = np.sign(y_true - last_actual)
    dir_pred = np.sign(y_pred - last_actual)
    return 100.0 * np.mean(dir_true == dir_pred)


def run_training(commodity_name: str, epochs: int, batch_size: int, device_mode: str, log_csv: str = None, mode: str = 'univariate', log_dir: str = None, production_file: str = None, hidden_size_override: int = None, num_layers_override: int = None, dropout_override: float = None, lr_override: float = None, early_stopping_patience: int = 10):
    root = Path('.').resolve()
    cfg = yaml.safe_load((root / 'config.yaml').read_text())
    if mode == 'multivariate' and 'multivariate' in cfg.get('model', {}):
        model_cfg = cfg['model']['multivariate']
    else:
        model_cfg = cfg['model']['univariate']
    seq_len = model_cfg.get('sequence_length', 28)
    hidden_size = hidden_size_override if hidden_size_override is not None else model_cfg.get('hidden_size', 64)
    num_layers = num_layers_override if num_layers_override is not None else model_cfg.get('num_layers', 2)
    dropout = dropout_override if dropout_override is not None else model_cfg.get('dropout', 0.2)
    lr = lr_override if lr_override is not None else model_cfg.get('learning_rate', 1e-3)

    processed_dir = root / 'data' / 'processed'
    price_df = pd.read_csv(processed_dir / 'price_cleaned.csv', parse_dates=['Tanggal'])
    price_df = price_df.sort_values(['Komoditi', 'Tanggal']).reset_index(drop=True)

    # prepare features / target depending on mode
    df = price_df.loc[price_df['Komoditi'] == commodity_name].copy()
    if df.empty:
        raise RuntimeError(f'No data found for commodity {commodity_name}')

    target_col = cfg['preprocessing']['target_col']
    df = add_time_series_features(df, target_col)
    df = df.dropna(subset=['lag_1', 'lag_7', 'lag_14']).reset_index(drop=True)
    engineered_cols = ['lag_1', 'lag_7', 'lag_14', 'season_sin', 'season_cos', 'holiday_flag']
    temporal_feature_cols = [target_col] + [c for c in engineered_cols if c in df.columns]

    if mode == 'univariate':
        features = df[temporal_feature_cols].values.astype(float)
        targets = df[target_col].values.astype(float)
        scaler_x = StandardScaler().fit(features)
        scaler_y = StandardScaler().fit(targets.reshape(-1, 1))
    else:
        prod_path = Path(production_file) if production_file else processed_dir / 'production_transformed.csv'
        if not prod_path.exists():
            print('Production file not found at', prod_path, '; continuing with temporal features only')
            features = df[temporal_feature_cols].values.astype(float)
            targets = df[target_col].values.astype(float)
            scaler_x = StandardScaler().fit(features)
            scaler_y = StandardScaler().fit(targets.reshape(-1, 1))
        else:
            prod = pd.read_csv(prod_path, parse_dates=True)
            commodity_col = None
            quarter_col = None
            for c in prod.columns:
                if 'komod' in c.lower() or 'komoditi' in c.lower() or 'commodity' in c.lower():
                    commodity_col = c
                if 'quarter' in c.lower() or 'quarter_start' in c.lower() or 'q_start' in c.lower():
                    quarter_col = c
            df['quarter_start'] = df['Tanggal'].dt.to_period('Q').dt.to_timestamp()
            if commodity_col and quarter_col:
                prod[quarter_col] = pd.to_datetime(prod[quarter_col])
                merged = df.merge(prod, left_on=['Komoditi', 'quarter_start'], right_on=[commodity_col, quarter_col], how='left')
            else:
                possible_q = [c for c in prod.columns if 'quarter' in c.lower()]
                possible_k = [c for c in prod.columns if 'komod' in c.lower() or 'commodity' in c.lower()]
                if possible_q and possible_k:
                    prod[possible_q[0]] = pd.to_datetime(prod[possible_q[0]])
                    merged = df.merge(prod, left_on=['Komoditi', 'quarter_start'], right_on=[possible_k[0], possible_q[0]], how='left')
                else:
                    print('Could not find commodity/quarter columns in production file; using temporal features only')
                    merged = df.copy()

            exog_cols = [c for c in merged.columns if c not in ['Tanggal', 'Komoditi', target_col, 'quarter_start'] and np.issubdtype(merged[c].dtype, np.number)]
            if len(exog_cols) == 0:
                print('No numeric exogenous features found after merge; using temporal features only')
                features = df[temporal_feature_cols].values.astype(float)
                targets = df[target_col].values.astype(float)
                scaler_x = StandardScaler().fit(features)
                scaler_y = StandardScaler().fit(targets.reshape(-1, 1))
            else:
                merged[exog_cols] = merged[exog_cols].ffill().fillna(0)
                features = merged[temporal_feature_cols + exog_cols].values.astype(float)
                targets = merged[target_col].values.astype(float)
                scaler_x = StandardScaler().fit(features)
                scaler_y = StandardScaler().fit(targets.reshape(-1, 1))

    full_dataset = PriceSequenceDataset(features, targets, seq_len, scaler_x, scaler_y)

    forecast_horizon = cfg.get('project', {}).get('forecast_horizon', 14)
    train_ds, val_ds, test_ds = split_time_series_dataset(full_dataset, cfg.get('training', {}).get('validation_ratio', 0.15), forecast_horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    use_cuda = (device_mode == 'cuda' or (device_mode == 'auto' and cfg.get('training', {}).get('use_cuda', True))) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device:', device)

    from src.models.base_model import BaseLSTMModel

    input_size = features.shape[1] if 'features' in locals() else 1
    model = BaseLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # TensorBoard writer
    tb_writer = None
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir)

    def train_one_epoch(loader):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(-1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        return total_loss / len(loader.dataset)

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        preds_all = []
        trues_all = []
        last_actuals = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy().ravel()
                last_scaled = xb[:, -1, 0].cpu().numpy().ravel()
                last_actual = scaler_y.inverse_transform(last_scaled.reshape(-1, 1)).ravel()
                trues = scaler_y.inverse_transform(yb.cpu().numpy().reshape(-1, 1)).ravel()
                preds_inv = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
                preds_all.extend(preds_inv.tolist())
                trues_all.extend(trues.tolist())
                last_actuals.extend(last_actual.tolist())
                loss = loss_fn(torch.tensor(preds).unsqueeze(-1), yb.cpu().unsqueeze(-1))
                total_loss += loss.item() * xb.size(0)
        return total_loss / len(loader.dataset), np.array(trues_all), np.array(preds_all), np.array(last_actuals)

    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    start = time.time()
    epochs_since_improve = 0
    checkpoint_path = None
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(train_loader)
        val_loss, _, _, _ = evaluate(val_loader)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch}/{epochs} - train_loss={tr_loss:.6f} val_loss={val_loss:.6f}")
        if tb_writer:
            tb_writer.add_scalar('loss/train', tr_loss, epoch)
            tb_writer.add_scalar('loss/val', val_loss, epoch)
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            epochs_since_improve = 0
            checkpoint_dir = Path(log_dir or 'models')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"lstm_{commodity_name.replace(' ', '_')}_{mode}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  Saved best model to {checkpoint_path}')
        else:
            epochs_since_improve += 1

        if early_stopping_patience is not None and epochs_since_improve >= early_stopping_patience:
            print(f'Early stopping after {epoch} epochs with no improvement for {early_stopping_patience} epochs.')
            break

    if checkpoint_path is not None:
        print(f'Loading best checkpoint from {checkpoint_path} for final evaluation')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    elapsed = time.time() - start
    print('Training finished in %.1f sec' % elapsed)

    # final evaluation on test
    test_loss, y_true, y_pred, last_actuals = evaluate(test_loader)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape_v = mape(y_true, y_pred)
    smape_v = smape(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred, last_actuals)

    print('\nFinal test metrics:')
    print(f'MAE: {mae:.3f} Rp/kg')
    print(f'RMSE: {rmse:.3f} Rp/kg')
    print(f'MAPE: {mape_v:.3f} %')
    print(f'sMAPE: {smape_v:.3f} %')
    print(f'Directional Accuracy: {da:.2f} %')

    if log_csv:
        Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
        write_header = not Path(log_csv).exists()
        with open(log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['commodity', 'mode', 'epochs', 'mae', 'rmse', 'mape', 'smape', 'directional_acc'])
            writer.writerow([commodity_name, mode, epochs, f'{mae:.3f}', f'{rmse:.3f}', f'{mape_v:.3f}', f'{smape_v:.3f}', f'{da:.3f}'])

    if tb_writer:
        tb_writer.add_scalar('metrics/mae', mae)
        tb_writer.add_scalar('metrics/rmse', rmse)
        tb_writer.add_scalar('metrics/mape', mape_v)
        tb_writer.add_scalar('metrics/smape', smape_v)
        tb_writer.close()

    return {'mae': mae, 'rmse': rmse, 'mape': mape_v, 'smape': smape_v, 'da': da, 'val_loss': best_val}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commodity', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--log-csv', type=str, default='runs/metrics.csv')
    parser.add_argument('--compare-csv', type=str, default=None, help='Optional comparison CSV file for both modes')
    parser.add_argument('--mode', choices=['univariate', 'multivariate', 'both'], default='univariate')
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--log-dir', type=str, default=None, help='TensorBoard log dir')
    parser.add_argument('--production-file', type=str, default=None, help='Path to production/exogenous file')
    parser.add_argument('--optuna', action='store_true', help='Run Optuna hyperparameter search')
    parser.add_argument('--optuna-trials', type=int, default=10)
    args = parser.parse_args()

    if args.optuna and args.mode == 'both':
        raise ValueError('Optuna mode does not support --mode both. Choose univariate or multivariate.')

    modes_to_run = ['univariate', 'multivariate'] if args.mode == 'both' else [args.mode]
    results = []

    if args.optuna:
        try:
            import optuna

            def objective(trial):
                hs = trial.suggest_int('hidden_size', 16, 128)
                nl = trial.suggest_int('num_layers', 1, 3)
                dr = trial.suggest_float('dropout', 0.0, 0.5)
                lr_s = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                trial_epochs = max(3, int(max(3, args.epochs // 6)))
                res = run_training(args.commodity, trial_epochs, args.batch_size, args.device, log_csv=None, mode=args.mode, log_dir=None, production_file=args.production_file, hidden_size_override=hs, num_layers_override=nl, dropout_override=dr, lr_override=lr_s, early_stopping_patience=args.early_stopping_patience)
                return res['val_loss']

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=args.optuna_trials)
            print('Best trial params:', study.best_trial.params)
            best = study.best_trial.params
            print('Retraining full model with best params...')
            run_training(
                args.commodity,
                args.epochs,
                args.batch_size,
                args.device,
                log_csv=args.log_csv,
                mode=args.mode,
                log_dir=(args.log_dir or f"runs/optuna/{args.commodity.replace(' ','_')}_best"),
                production_file=args.production_file,
                hidden_size_override=best.get('hidden_size'),
                num_layers_override=best.get('num_layers'),
                dropout_override=best.get('dropout'),
                lr_override=best.get('lr'),
                early_stopping_patience=args.early_stopping_patience,
            )
        except Exception as e:
            print('Optuna not available or failed:', e)
    else:
        for submode in modes_to_run:
            sub_log_dir = args.log_dir
            if args.mode == 'both' and args.log_dir:
                sub_log_dir = str(Path(args.log_dir) / submode)
            result = run_training(
                args.commodity,
                args.epochs,
                args.batch_size,
                args.device,
                args.log_csv,
                mode=submode,
                log_dir=sub_log_dir,
                production_file=args.production_file,
                early_stopping_patience=args.early_stopping_patience,
            )
            result['mode'] = submode
            results.append(result)

        if args.compare_csv and args.mode == 'both':
            Path(args.compare_csv).parent.mkdir(parents=True, exist_ok=True)
            write_header = not Path(args.compare_csv).exists()
            with open(args.compare_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['commodity', 'mode', 'epochs', 'mae', 'rmse', 'mape', 'smape', 'directional_acc'])
                for result in results:
                    writer.writerow([
                        args.commodity,
                        result['mode'],
                        args.epochs,
                        f"{result['mae']:.3f}",
                        f"{result['rmse']:.3f}",
                        f"{result['mape']:.3f}",
                        f"{result['smape']:.3f}",
                        f"{result['da']:.3f}",
                    ])
            print(f'Comparison output file: {args.compare_csv}')


if __name__ == '__main__':
    main()
