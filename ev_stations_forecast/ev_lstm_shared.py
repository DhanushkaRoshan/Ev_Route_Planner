import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

TARGET_STATIONS = [121805, 125065]
SEQ_LENGTH = 288


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, forecast_horizon):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * forecast_horizon)

    def forward(self, x):
        # init hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.forecast_horizon, self.output_size)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').copy()
    # filter out rows where Total is 0
    df = df[df['Total'] > 0].copy()

    df_pivot = df.pivot_table(index='timestamp', columns='location_id', values='Available', aggfunc='mean').fillna(0)
    station_cols = df_pivot.columns.tolist()

    features = df.groupby('timestamp')[['Total', 'Coming Soon', 'Offline']].mean()
    df_pivot['hour'] = df_pivot.index.hour
    df_pivot['day_of_week'] = df_pivot.index.dayofweek
    df_pivot['month'] = df_pivot.index.month
    df_combined = pd.concat([df_pivot, features[['Total', 'Coming Soon', 'Offline']]], axis=1)

    total_connectors = df.groupby('location_id')['Total'].first().reindex(station_cols).values
    return df_combined.values, df_pivot.columns[:-3].tolist(), total_connectors


def scale_data(data, fit_rows=None):
    scaler = MinMaxScaler(feature_range=(0, 1))
    if fit_rows is not None:
        scaler.fit(data[:fit_rows])
    else:
        scaler.fit(data)
    scaled = scaler.transform(data)
    return scaled, scaler


def create_sequences(data, seq_length, forecast_horizon, num_stations):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon, :num_stations])
    return np.array(X), np.array(y)


def unscale_and_clip(raw_output, scaler, input_size, num_stations, forecast_horizon, total_connectors):
    batch_sz = raw_output.shape[0]
    padding = np.zeros((batch_sz, forecast_horizon, input_size - num_stations))
    reshaped = np.concatenate((raw_output, padding), axis=2)
    unscaled = scaler.inverse_transform(reshaped.reshape(-1, input_size))[:, :num_stations]
    return np.round(np.clip(unscaled, 0, total_connectors)).astype(int)


# compute regression metrics for predictions
def evaluate_predictions(y_true, y_pred):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))

    non_zero = y_true_flat != 0
    mape = np.mean(
        np.abs((y_true_flat[non_zero] - y_pred_flat[non_zero]) / y_true_flat[non_zero])
    ) * 100

    r2 = r2_score(y_true_flat, y_pred_flat)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'R2': r2}


def evaluate_classification(y_true, y_pred):
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # binarize - available or not
    true_bin = (y_true_flat > 0).astype(int)
    pred_bin = (y_pred_flat > 0).astype(int)

    acc = accuracy_score(true_bin, pred_bin)
    prec = precision_score(true_bin, pred_bin, zero_division=0)
    rec = recall_score(true_bin, pred_bin, zero_division=0)
    f1 = f1_score(true_bin, pred_bin, zero_division=0)

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
    }


def plot_predictions(y_true, y_pred, station_ids, output_file='accuracy_plot.png'):
    valid_stations = [s for s in station_ids if s in TARGET_STATIONS]
    if not valid_stations:
        print(f"No target stations {TARGET_STATIONS} found in station_ids. Skipping plot.")
        return

    forecast_horizon = y_true.shape[1]
    plt.figure(figsize=(12, 6))
    for i, station in enumerate(station_ids):
        if station not in TARGET_STATIONS:
            continue
        subplot_idx = valid_stations.index(station) + 1
        plt.subplot(1, len(valid_stations), subplot_idx)
        plt.plot(range(forecast_horizon), y_true[0, :, i], label='Actual', marker='o', color='blue')
        plt.plot(range(forecast_horizon), y_pred[0, :, i], label='Predicted', marker='x', linestyle='--', color='orange')
        plt.title(f'Station {station}')
        plt.xlabel('Time Step (5-min intervals)')
        plt.ylabel('Available Connectors')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
