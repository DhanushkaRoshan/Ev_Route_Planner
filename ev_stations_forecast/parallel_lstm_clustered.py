import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ev_lstm_shared import (
    EarlyStopping,
    TimeSeriesDataset,
    scale_data,
    create_sequences,
    unscale_and_clip,
    evaluate_predictions,
)

torch.set_num_threads(1)

RESAMPLE_INTERVAL = '15min'
SEQ_LENGTH = 96
FORECAST_HORIZON = 48
HIDDEN_SIZE = 200
NUM_LAYERS = 3
DROPOUT = 0.0
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
DROP_ALWAYS_ZERO = True
N_CLUSTERS = 5
DATA_FILE = "data/combined_port_summary.csv"
OUTPUT_DIR = "parallel_clustered_results"


class ClusterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 forecast_horizon, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * forecast_horizon)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.forecast_horizon, self.output_size)


def load_full_dataset(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').copy()
    df = df[df['Total'] > 0].copy()

    df_pivot = df.pivot_table(
        index='timestamp', columns='location_id', values='Available', aggfunc='mean'
    ).fillna(0)

    if DROP_ALWAYS_ZERO:
        always_zero = (df_pivot == 0).all()
        num_dropped = always_zero.sum()
        if num_dropped:
            print(f"Dropping {num_dropped} always-zero stations "
                  f"(keeping {(~always_zero).sum()} / {len(always_zero)})")
            df_pivot = df_pivot.loc[:, ~always_zero]

    station_cols = df_pivot.columns.tolist()
    features = df.groupby('timestamp')[['Total', 'Coming Soon', 'Offline']].mean()

    total_connectors = df.groupby('location_id')['Total'].first().reindex(station_cols).values

    return df, df_pivot, features, station_cols, total_connectors


def cluster_stations(df_pivot, n_clusters=N_CLUSTERS):
    station_ids = df_pivot.columns.tolist()
    station_feats = []

    for sid in station_ids:
        series = df_pivot[sid]
        mean_val = series.mean()
        std_val = series.std()
        zero_frac = (series == 0).mean()
        max_val = series.max()

        # get hourly and daily patterns for clustering
        hourly = series.groupby(series.index.hour).mean().values
        daily = series.groupby(series.index.dayofweek).mean().values

        feat = np.concatenate([[mean_val, std_val, zero_frac, max_val], hourly, daily])
        station_feats.append(feat)

    X = np.array(station_feats)
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    clusters = {}
    for i in range(n_clusters):
        mask = labels == i
        cluster_sids = [station_ids[j] for j in range(len(station_ids)) if mask[j]]
        clusters[f"cluster_{i}"] = cluster_sids

    print(f"\nStation clustering (K={n_clusters}):")
    for name, sids in sorted(clusters.items(), key=lambda x: -len(x[1])):
        print(f"  {name}: {len(sids)} stations")

    return clusters, labels, X_scaled


def prepare_full_input_data(df_pivot, features, resample_interval=RESAMPLE_INTERVAL):
    all_station_cols = df_pivot.columns.tolist()
    n_all = len(all_station_cols)

    df_full = df_pivot.copy()
    df_full['hour'] = df_full.index.hour
    df_full['day_of_week'] = df_full.index.dayofweek
    df_full['month'] = df_full.index.month
    df_combined = pd.concat([df_full, features[['Total', 'Coming Soon', 'Offline']]], axis=1)

    station_data = df_combined.iloc[:, :n_all].resample(resample_interval).mean().fillna(0)
    feature_data = df_combined.iloc[:, n_all:].resample(resample_interval).first().ffill()
    df_resampled = pd.concat([station_data, feature_data], axis=1)

    return df_resampled, all_station_cols


def get_cluster_indices(all_station_cols, cluster_stations, total_connectors_all):
    indices = [all_station_cols.index(s) for s in cluster_stations]
    total_connectors = total_connectors_all[indices]
    return indices, total_connectors


def create_sequences_full_input(data, seq_length, forecast_horizon, target_col_indices):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon, target_col_indices])
    return np.array(X), np.array(y)


def train_and_evaluate_cluster(full_data, cluster_station_ids, target_col_indices,
                               total_connectors, cluster_name):
    num_targets = len(cluster_station_ids)
    input_size = full_data.shape[1]

    scaled_data, scaler = scale_data(full_data)
    X, y = create_sequences_full_input(scaled_data, SEQ_LENGTH, FORECAST_HORIZON, target_col_indices)
    n_seq = len(X)

    train_end = int(TRAIN_SPLIT * n_seq)
    val_end = int((TRAIN_SPLIT + VAL_SPLIT) * n_seq)
    gap = SEQ_LENGTH + FORECAST_HORIZON - 1

    val_start = train_end + gap
    test_start = val_end + gap

    # refit the scaler to avoid data leakage
    last_train_row = (train_end - 1) + SEQ_LENGTH + FORECAST_HORIZON
    scaled_data, scaler = scale_data(full_data, fit_rows=last_train_row)
    X, y = create_sequences_full_input(scaled_data, SEQ_LENGTH, FORECAST_HORIZON, target_col_indices)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[val_start:val_end], y[val_start:val_end]
    X_test, y_test = X[test_start:], y[test_start:]

    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"  Target stations: {num_targets}, Input features: {input_size}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, Gap: {gap}")

    model = ClusterLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, num_targets,
                        FORECAST_HORIZON, dropout=DROPOUT)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping = EarlyStopping(patience=5)
    stopped_epoch = EPOCHS

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                val_loss += criterion(model(inputs), targets).item()
        val_loss /= len(val_loader)

        print(f"  Epoch {epoch+1}/{EPOCHS}, Train: {train_loss:.4f}, "
              f"Val: {val_loss:.4f}", flush=True)

        if early_stopping.step(val_loss, epoch + 1):
            stopped_epoch = epoch + 1
            print(f"  Early stopping at epoch {stopped_epoch}", flush=True)
            break

    y_true, y_pred = _collect_predictions(model, test_loader, scaler, input_size,
                                          target_col_indices, total_connectors)

    overall_metrics = evaluate_predictions(y_true, y_pred)
    overall_metrics['stopped_epoch'] = stopped_epoch

    per_station_metrics = {}
    for i, sid in enumerate(cluster_station_ids):
        st_true = y_true[:, :, i].reshape(-1)
        st_pred = y_pred[:, :, i].reshape(-1)

        mae = np.mean(np.abs(st_true - st_pred))
        rmse = np.sqrt(np.mean((st_true - st_pred) ** 2))
        non_zero = st_true != 0
        mape = (np.mean(np.abs((st_true[non_zero] - st_pred[non_zero]) / st_true[non_zero])) * 100
                if non_zero.any() else 0.0)
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

    predictions_df = _predict_future(model, scaled_data, scaler, input_size,
                                     num_targets, target_col_indices,
                                     total_connectors, cluster_station_ids)

    return {
        'overall': overall_metrics,
        'per_station': per_station_metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'station_ids': cluster_station_ids,
        'predictions_df': predictions_df,
    }


def _collect_predictions(model, loader, scaler, input_size, target_col_indices, total_connectors):
    num_target = len(target_col_indices)
    true_list = []
    pred_list = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            raw_pred = model(inputs).numpy()
            raw_true = targets.numpy()

            for raw, out_list in [(raw_pred, pred_list), (raw_true, true_list)]:
                batch_sz = raw.shape[0]
                full = np.zeros((batch_sz * FORECAST_HORIZON, input_size))
                flat = raw.reshape(batch_sz * FORECAST_HORIZON, num_target)
                for j, col_idx in enumerate(target_col_indices):
                    full[:, col_idx] = flat[:, j]
                unscaled = scaler.inverse_transform(full)
                vals = unscaled[:, target_col_indices]
                vals = np.round(np.clip(vals, 0, total_connectors)).astype(int)
                out_list.append(vals.reshape(batch_sz, FORECAST_HORIZON, num_target))

    return np.concatenate(true_list), np.concatenate(pred_list)


def _predict_future(model, scaled_data, scaler, input_size, num_target,
                    target_col_indices, total_connectors, station_ids):
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(
            scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, input_size),
            dtype=torch.float32
        )
        raw = model(last_seq).numpy()

    full = np.zeros((FORECAST_HORIZON, input_size))
    flat = raw.reshape(FORECAST_HORIZON, num_target)
    for j, col_idx in enumerate(target_col_indices):
        full[:, col_idx] = flat[:, j]
    unscaled = scaler.inverse_transform(full)
    prediction = unscaled[:, target_col_indices]
    prediction = np.round(np.clip(prediction, 0, total_connectors)).astype(int)

    timestamps = pd.date_range(
        start=pd.Timestamp.now().ceil(RESAMPLE_INTERVAL),
        periods=FORECAST_HORIZON,
        freq=RESAMPLE_INTERVAL,
    )
    return pd.DataFrame(prediction, columns=station_ids, index=timestamps)


def plot_cluster_profiles(df_pivot, clusters, labels, output_file):
    n_clusters = len(clusters)
    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 4), sharey=True)
    if n_clusters == 1:
        axes = [axes]

    for i, (name, sids) in enumerate(sorted(clusters.items())):
        ax = axes[i]
        for sid in sids:
            hourly = df_pivot[sid].groupby(df_pivot.index.hour).mean()
            ax.plot(hourly.index, hourly.values, alpha=0.3, linewidth=0.8)
        # plot the cluster average on top
        cluster_avg = df_pivot[sids].mean(axis=1).groupby(df_pivot.index.hour).mean()
        ax.plot(cluster_avg.index, cluster_avg.values, color='red', linewidth=2, label='Cluster avg')
        ax.set_title(f'{name}\n({len(sids)} stations)', fontsize=10)
        ax.set_xlabel('Hour of day')
        if i == 0:
            ax.set_ylabel('Avg available connectors')
        ax.set_xlim(0, 23)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle('Daily Availability Profiles by Cluster', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved {output_file}")


def plot_cluster_comparison(cluster_results, output_file):
    names = [r['cluster'] for r in cluster_results]
    maes = [r['overall']['MAE'] for r in cluster_results]
    rmses = [r['overall']['RMSE'] for r in cluster_results]
    n_stations = [len(r['station_ids']) for r in cluster_results]

    order = np.argsort(rmses)[::-1]
    names = [f"{names[i]} ({n_stations[i]} st.)" for i in order]
    maes = [maes[i] for i in order]
    rmses = [rmses[i] for i in order]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(x - width/2, rmses, width, label='RMSE', color='steelblue')
    ax.barh(x + width/2, maes, width, label='MAE', color='coral')
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Connectors')
    ax.set_title('Per-Cluster Prediction Error (12h horizon)')
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved {output_file}")


def plot_all_station_errors(cluster_results, output_file):
    all_stations = []
    for r in cluster_results:
        for sid, m in r['per_station'].items():
            if m['MAE'] > 0:
                all_stations.append((str(sid), m['MAE'], m['total_connectors']))

    if not all_stations:
        print("  All stations have MAE=0, skipping.")
        return

    all_stations.sort(key=lambda x: x[1], reverse=True)
    if len(all_stations) > 40:
        all_stations = all_stations[:40]

    station_labels = [s[0] for s in all_stations]
    maes = [s[1] for s in all_stations]

    fig, ax = plt.subplots(figsize=(12, max(4, len(station_labels) * 0.3)))
    ax.barh(range(len(station_labels)), maes, color='steelblue')
    ax.set_yticks(range(len(station_labels)))
    ax.set_yticklabels(station_labels, fontsize=7)
    ax.set_xlabel('MAE (connectors)')
    ax.set_title('Per-Station Mean Absolute Error (All Clusters)')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved {output_file}")


def plot_sample_predictions(cluster_results, output_file):
    plot_data = []
    for r in cluster_results:
        y_true = r['y_true']
        station_ids = r['station_ids']
        variances = np.var(y_true.reshape(-1, y_true.shape[2]), axis=0)
        best_idx = np.argmax(variances)
        if variances[best_idx] > 0:
            plot_data.append((r['cluster'], r['y_true'], r['y_pred'], station_ids, best_idx,
                              r['per_station'][str(station_ids[best_idx])]['total_connectors']))

    if not plot_data:
        print("  No stations with variance > 0, skipping.")
        return

    plot_data = plot_data[:6]
    cols = min(3, len(plot_data))
    rows = (len(plot_data) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, (cluster, y_true, y_pred, sids, si, max_c) in enumerate(plot_data):
        ax = axes[idx // cols][idx % cols]
        plot_len = min(48, y_true.shape[1])
        ax.plot(range(plot_len), y_true[0, :plot_len, si],
                label='Actual', marker='o', markersize=2, color='blue')
        ax.plot(range(plot_len), y_pred[0, :plot_len, si],
                label='Predicted', marker='x', markersize=2, linestyle='--', color='orange')
        ax.set_title(f'{cluster}\nStation {sids[si]} (max {max_c})', fontsize=9)
        ax.set_xlabel('Time Step (15-min)')
        ax.set_ylabel('Available Connectors')
        ax.set_ylim(-0.5, max_c + 0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(len(plot_data), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle('Predicted vs Actual Connector Counts (Clustered Parallel LSTM, 12h)', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved {output_file}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pred_dir = os.path.join(OUTPUT_DIR, "predictions")
    metrics_dir = os.path.join(OUTPUT_DIR, "metrics")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    print("Loading dataset...")
    df, df_pivot, features, all_station_cols, total_connectors_all = load_full_dataset(DATA_FILE)

    clusters, labels, X_scaled = cluster_stations(df_pivot, N_CLUSTERS)

    assignments = []
    for cluster_name, sids in clusters.items():
        for sid in sids:
            assignments.append({'station_id': sid, 'cluster': cluster_name})
    assignments_df = pd.DataFrame(assignments)
    assignments_df.to_csv(os.path.join(OUTPUT_DIR, 'cluster_assignments.csv'), index=False)
    print(f"Cluster assignments saved ({len(assignments_df)} stations)")

    print("\nGenerating cluster profile plot...")
    plot_cluster_profiles(df_pivot, clusters, labels,
                          os.path.join(OUTPUT_DIR, 'cluster_profiles.png'))

    print("\nResampling full dataset...")
    full_resampled, resampled_station_cols = prepare_full_input_data(df_pivot, features)
    full_data = full_resampled.values
    print(f"  Full data shape: {full_data.shape} ({len(resampled_station_cols)} stations + 6 features)")

    cluster_results = []
    longest_time = 0.0

    for cluster_name, cluster_sids in sorted(clusters.items()):
        print(f"\n{'='*60}")
        print(f"  {cluster_name} ({len(cluster_sids)} stations)")
        print(f"{'='*60}")

        t0 = time.time()
        target_col_indices, total_connectors = get_cluster_indices(
            resampled_station_cols, cluster_sids, total_connectors_all
        )
        results = train_and_evaluate_cluster(
            full_data, cluster_sids, target_col_indices, total_connectors, cluster_name
        )
        elapsed = time.time() - t0
        longest_time = max(longest_time, elapsed)

        results['predictions_df'].to_csv(
            os.path.join(pred_dir, f"{cluster_name}_predictions.csv")
        )
        cluster_metrics = {
            'overall': results['overall'],
            'per_station': results['per_station'],
            'time_s': round(elapsed, 1),
        }
        with open(os.path.join(metrics_dir, f"{cluster_name}_metrics.json"), 'w') as f:
            json.dump(cluster_metrics, f, indent=4)

        cluster_results.append({
            'cluster': cluster_name,
            'time_s': elapsed,
            **results,
        })

        print(f"  Time: {elapsed:.1f}s | MAE: {results['overall']['MAE']:.4f} | "
              f"RMSE: {results['overall']['RMSE']:.4f} | R2: {results['overall']['R2']:.4f}")

    n_clusters = len(cluster_results)
    total_time = sum(r['time_s'] for r in cluster_results)
    avg_mae = np.mean([r['overall']['MAE'] for r in cluster_results])
    avg_rmse = np.mean([r['overall']['RMSE'] for r in cluster_results])
    avg_r2 = np.mean([r['overall']['R2'] for r in cluster_results])
    avg_mape = np.mean([r['overall']['MAPE (%)'] for r in cluster_results])
    avg_epoch = np.mean([r['overall']['stopped_epoch'] for r in cluster_results])

    all_true = np.concatenate([r['y_true'].reshape(-1) for r in cluster_results])
    all_pred = np.concatenate([r['y_pred'].reshape(-1) for r in cluster_results])
    combined_mae = np.mean(np.abs(all_true - all_pred))
    combined_rmse = np.sqrt(np.mean((all_true - all_pred) ** 2))
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    combined_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    non_zero = all_true != 0
    combined_mape = (np.mean(np.abs((all_true[non_zero] - all_pred[non_zero]) / all_true[non_zero])) * 100
                     if non_zero.any() else 0.0)

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({n_clusters} clusters, {sum(len(r['station_ids']) for r in cluster_results)} stations)")
    print(f"{'='*60}")
    print(f"  Combined MAE:  {combined_mae:.4f} connectors")
    print(f"  Combined RMSE: {combined_rmse:.4f} connectors")
    print(f"  Combined MAPE: {combined_mape:.2f}%")
    print(f"  Combined R2:   {combined_r2:.4f}")
    print(f"  Avg per-cluster R2: {avg_r2:.4f}")
    print(f"  Avg stop epoch: {avg_epoch:.1f}")
    print(f"  Wall-clock time (longest cluster): {longest_time:.0f}s")
    print(f"  Total sequential time: {total_time:.0f}s")

    summary = {
        'n_clusters': n_clusters,
        'n_stations': sum(len(r['station_ids']) for r in cluster_results),
        'wall_clock_time_s': round(longest_time, 1),
        'total_sequential_time_s': round(total_time, 1),
        'combined_MAE': round(combined_mae, 4),
        'combined_RMSE': round(combined_rmse, 4),
        'combined_MAPE': round(combined_mape, 2),
        'combined_R2': round(combined_r2, 4),
        'avg_MAE': round(avg_mae, 4),
        'avg_RMSE': round(avg_rmse, 4),
        'avg_MAPE': round(avg_mape, 2),
        'avg_R2': round(avg_r2, 4),
        'avg_stopped_epoch': round(avg_epoch, 1),
        'clusters': [
            {
                'cluster': r['cluster'],
                'n_stations': len(r['station_ids']),
                'station_ids': [str(s) for s in r['station_ids']],
                'time_s': round(r['time_s'], 1),
                'stopped_epoch': r['overall']['stopped_epoch'],
                **{k: r['overall'][k] for k in ('MAE', 'RMSE', 'MAPE (%)', 'R2')},
                'per_station': r['per_station'],
            }
            for r in cluster_results
        ],
    }
    summary_file = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary saved to {summary_file}")

    print("\nGenerating plots...")
    plot_cluster_comparison(cluster_results,
                            os.path.join(OUTPUT_DIR, 'cluster_comparison.png'))
    plot_all_station_errors(cluster_results,
                            os.path.join(OUTPUT_DIR, 'per_station_error.png'))
    plot_sample_predictions(cluster_results,
                            os.path.join(OUTPUT_DIR, 'pred_vs_actual.png'))

    print("\nDone.")
