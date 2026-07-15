import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ev_lstm_shared import (
    EarlyStopping,
    MultivariateLSTM,
    TimeSeriesDataset,
    scale_data,
    create_sequences,
    unscale_and_clip,
    evaluate_predictions,
)

RESAMPLE_INTERVAL = '15min'
SEQ_LENGTH = 96
FORECAST_HORIZON = 48
HIDDEN_SIZE = 200
NUM_LAYERS = 3
EPOCHS = 30
BATCH_SIZE = 1
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
DROP_ALWAYS_ZERO = True
OUTPUT_DIR = "sequential_results"
DATA_FILE = "data/combined_port_summary.csv"


def load_and_prepare_data_resampled(file_path, resample_interval=RESAMPLE_INTERVAL,
                                    drop_always_zero=DROP_ALWAYS_ZERO):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').copy()
    df = df[df['Total'] > 0].copy()

    df_pivot = df.pivot_table(
        index='timestamp', columns='location_id', values='Available', aggfunc='mean'
    ).fillna(0)

    if drop_always_zero:
        always_zero = (df_pivot == 0).all()
        num_dropped = always_zero.sum()
        if num_dropped:
            print(f"  Dropping {num_dropped} always-zero stations "
                  f"(keeping {(~always_zero).sum()} / {len(always_zero)})")
            df_pivot = df_pivot.loc[:, ~always_zero]

    station_cols = df_pivot.columns.tolist()

    features = df.groupby('timestamp')[['Total', 'Coming Soon', 'Offline']].mean()

    df_pivot['hour'] = df_pivot.index.hour
    df_pivot['day_of_week'] = df_pivot.index.dayofweek
    df_pivot['month'] = df_pivot.index.month
    df_combined = pd.concat([df_pivot, features[['Total', 'Coming Soon', 'Offline']]], axis=1)

    n_stations = len(station_cols)
    station_data = df_combined.iloc[:, :n_stations].resample(resample_interval).mean().fillna(0)
    feature_data = df_combined.iloc[:, n_stations:].resample(resample_interval).first().ffill()
    df_resampled = pd.concat([station_data, feature_data], axis=1)

    total_connectors = df.groupby('location_id')['Total'].first().reindex(station_cols).values
    return df_resampled.values, station_cols, total_connectors


def train_and_evaluate(file_path):
    data, station_ids, total_connectors = load_and_prepare_data_resampled(file_path)
    num_stations = len(station_ids)
    input_size = data.shape[1]

    print(f"Stations: {num_stations}, Features: {input_size}")
    print(f"Resampled to: {RESAMPLE_INTERVAL}")
    print(f"Forecast horizon: {FORECAST_HORIZON} steps ({FORECAST_HORIZON * 15} min = {FORECAST_HORIZON * 15 // 60} hours)")

    scaled_data, scaler = scale_data(data)
    X, y = create_sequences(scaled_data, SEQ_LENGTH, FORECAST_HORIZON, num_stations)
    n_seq = len(X)

    train_end = int(TRAIN_SPLIT * n_seq)
    val_end = int((TRAIN_SPLIT + VAL_SPLIT) * n_seq)
    gap = SEQ_LENGTH + FORECAST_HORIZON - 1

    val_start = train_end + gap
    test_start = val_end + gap

    # refit scaler only on training data to avoid leakage
    last_train_target_row = (train_end - 1) + SEQ_LENGTH + FORECAST_HORIZON
    scaled_data, scaler = scale_data(data, fit_rows=last_train_target_row)
    X, y = create_sequences(scaled_data, SEQ_LENGTH, FORECAST_HORIZON, num_stations)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[val_start:val_end], y[val_start:val_end]
    X_test,  y_test  = X[test_start:], y[test_start:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} sequences")
    print(f"Gap between partitions: {gap} sequences")

    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    model = MultivariateLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, num_stations, FORECAST_HORIZON)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=5)
    stopped_epoch = EPOCHS

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        ep_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                ep_val_loss += criterion(model(inputs), targets).item()
        ep_val_loss /= len(val_loader)
        val_losses.append(ep_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss:.6f}, Val Loss: {ep_val_loss:.6f}")

        if early_stopping.step(ep_val_loss, epoch + 1):
            stopped_epoch = epoch + 1
            print(f"Early stopping at epoch {stopped_epoch} (best epoch: {early_stopping.best_epoch})")
            break

    y_true, y_pred = _collect_predictions(model, test_loader, scaler, input_size,
                                          num_stations, total_connectors)

    overall_metrics = evaluate_predictions(y_true, y_pred)
    overall_metrics['stopped_epoch'] = stopped_epoch

    print("\n=== Overall Regression Metrics ===")
    for k in ('MAE', 'RMSE', 'MAPE (%)', 'R2'):
        print(f"  {k}: {overall_metrics[k]:.4f}")

    per_station_metrics = {}
    for i, sid in enumerate(station_ids):
        st_true = y_true[:, :, i].reshape(-1)
        st_pred = y_pred[:, :, i].reshape(-1)

        mae = np.mean(np.abs(st_true - st_pred))
        rmse = np.sqrt(np.mean((st_true - st_pred) ** 2))
        non_zero = st_true != 0
        mape = np.mean(np.abs((st_true[non_zero] - st_pred[non_zero]) / st_true[non_zero])) * 100 if non_zero.any() else 0.0
        ss_res = np.sum((st_true - st_pred) ** 2)
        ss_tot = np.sum((st_true - np.mean(st_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

        per_station_metrics[str(sid)] = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'MAPE (%)': round(mape, 2),
            'R2': round(r2, 4),
            'total_connectors': int(total_connectors[i]),
        }

    print(f"\n=== Per-Station Results (showing top 10 by RMSE) ===")
    sorted_stations = sorted(per_station_metrics.items(), key=lambda x: x[1]['RMSE'], reverse=True)
    for sid, m in sorted_stations[:10]:
        print(f"  Station {sid} (max {m['total_connectors']}): MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R2={m['R2']:.4f}")

    predictions_df = _predict_future(model, scaled_data, scaler, input_size,
                                     num_stations, total_connectors, station_ids)

    return {
        'overall': overall_metrics,
        'per_station': per_station_metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'station_ids': station_ids,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'stopped_epoch': stopped_epoch,
        'predictions_df': predictions_df,
    }


def _collect_predictions(model, loader, scaler, input_size, num_stations, total_connectors):
    true_list = []
    pred_list = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs).numpy()
            pred_list.append(
                unscale_and_clip(outputs, scaler, input_size, num_stations,
                                 model.forecast_horizon, total_connectors)
                .reshape(-1, model.forecast_horizon, num_stations)
            )
            true_list.append(
                unscale_and_clip(targets.numpy(), scaler, input_size, num_stations,
                                 model.forecast_horizon, total_connectors)
                .reshape(-1, model.forecast_horizon, num_stations)
            )
    return np.concatenate(true_list), np.concatenate(pred_list)


def _predict_future(model, scaled_data, scaler, input_size, num_stations,
                    total_connectors, station_ids):
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(
            scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, input_size),
            dtype=torch.float32
        )
        raw = model(last_seq).numpy()

    prediction = unscale_and_clip(raw, scaler, input_size, num_stations,
                                  FORECAST_HORIZON, total_connectors)

    timestamps = pd.date_range(
        start=pd.Timestamp.now().ceil(RESAMPLE_INTERVAL),
        periods=FORECAST_HORIZON,
        freq=RESAMPLE_INTERVAL,
    )
    return pd.DataFrame(
        prediction.reshape(FORECAST_HORIZON, num_stations),
        columns=station_ids,
        index=timestamps
    )


def _plot_pred_vs_actual(y_true, y_pred, station_ids, total_connectors, output_file):
    # pick top 6 stations by variance for the plot
    variances = np.var(y_true.reshape(-1, y_true.shape[2]), axis=0)
    top_indices = np.argsort(variances)[-6:][::-1]

    n_plots = len(top_indices)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, station_idx in enumerate(top_indices):
        ax = axes[idx // cols][idx % cols]
        sid = station_ids[station_idx]
        max_conn = total_connectors[station_idx]
        horizon = y_true.shape[1]

        ax.plot(range(horizon), y_true[0, :, station_idx],
                label='Actual', marker='o', markersize=3, color='blue')
        ax.plot(range(horizon), y_pred[0, :, station_idx],
                label='Predicted', marker='x', markersize=3, linestyle='--', color='orange')
        ax.set_title(f'Station {sid} (max {max_conn})')
        ax.set_xlabel('Time Step (15-min)')
        ax.set_ylabel('Available Connectors')
        ax.set_ylim(-0.5, max_conn + 0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_plots, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle('Predicted vs Actual Connector Counts (Sequential LSTM)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved {output_file}")


def _plot_per_station_error(per_station_metrics, output_file):
    stations = sorted(per_station_metrics.items(), key=lambda x: x[1]['MAE'], reverse=True)
    stations = [(sid, m) for sid, m in stations if m['MAE'] > 0]
    if not stations:
        print("  All stations have MAE=0, skipping per-station error plot.")
        return

    if len(stations) > 30:
        stations = stations[:30]

    sids = [s[0] for s in stations]
    maes = [s[1]['MAE'] for s in stations]

    fig, ax = plt.subplots(figsize=(12, max(4, len(sids) * 0.3)))
    bars = ax.barh(range(len(sids)), maes, color='steelblue')
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels(sids, fontsize=7)
    ax.set_xlabel('MAE (connectors)')
    ax.set_title('Per-Station Mean Absolute Error (Sequential LSTM)')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved {output_file}")


def _plot_loss_curve(train_losses, val_losses, stopped_epoch, output_file):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.axvline(x=stopped_epoch, color='red', linestyle='--', label=f'Early Stop (epoch {stopped_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Sequential LSTM Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved {output_file}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = train_and_evaluate(DATA_FILE)

    metrics_out = {
        'overall': results['overall'],
        'per_station': results['per_station'],
    }
    metrics_file = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_out, f, indent=4)
    print(f"\nMetrics saved to {metrics_file}")

    pred_file = os.path.join(OUTPUT_DIR, 'predictions.csv')
    results['predictions_df'].to_csv(pred_file)
    print(f"Future predictions saved to {pred_file}")

    print("\nGenerating plots...")
    _plot_pred_vs_actual(
        results['y_true'], results['y_pred'],
        results['station_ids'],
        [results['per_station'][str(s)]['total_connectors'] for s in results['station_ids']],
        os.path.join(OUTPUT_DIR, 'pred_vs_actual.png')
    )
    _plot_per_station_error(
        results['per_station'],
        os.path.join(OUTPUT_DIR, 'per_station_error.png')
    )
    _plot_loss_curve(
        results['train_losses'], results['val_losses'],
        results['stopped_epoch'],
        os.path.join(OUTPUT_DIR, 'loss_curve.png')
    )

    print("\nDone.")
