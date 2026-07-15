import json
import os
import signal
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

STATION_TIMEOUT = 180

RESAMPLE_INTERVAL = '1h'
FORECAST_HORIZON = 12
SEASONAL_PERIOD = 24
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
DROP_ALWAYS_ZERO = True
DATA_FILE = "data/combined_port_summary.csv"
OUTPUT_DIR = "sarima_results"
DYNAMIC_THRESHOLD = 0.5

MAX_P = 3
MAX_Q = 3
MAX_D = 2
MAX_SEASONAL_P = 1
MAX_SEASONAL_Q = 1
MAX_SEASONAL_D = 1


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').copy()
    df = df[df['Total'] > 0].copy()

    df_pivot = df.pivot_table(
        index='timestamp', columns='location_id',
        values='Available', aggfunc='mean'
    ).fillna(0)

    if DROP_ALWAYS_ZERO:
        always_zero = (df_pivot == 0).all()
        num_dropped = always_zero.sum()
        if num_dropped:
            print(f"  Dropping {num_dropped} always-zero stations "
                  f"(keeping {(~always_zero).sum()} / {len(always_zero)})")
            df_pivot = df_pivot.loc[:, ~always_zero]

    station_cols = df_pivot.columns.tolist()

    # resample to hourly
    station_data = df_pivot.resample(RESAMPLE_INTERVAL).mean().fillna(0)

    total_connectors = (
        df.groupby('location_id')['Total'].first()
        .reindex(station_cols).values
    )
    return station_data, station_cols, total_connectors


def evaluate_per_station(y_true, y_pred, station_ids, total_connectors,
                         train_data, n_stations):
    train_std = np.std(train_data, axis=0)

    per_station = {}
    for i, sid in enumerate(station_ids):
        st_true = y_true[:, i]
        st_pred = y_pred[:, i]

        mae = np.mean(np.abs(st_true - st_pred))
        rmse = np.sqrt(np.mean((st_true - st_pred) ** 2))
        non_zero = st_true != 0
        mape = (np.mean(np.abs((st_true[non_zero] - st_pred[non_zero]) /
                                st_true[non_zero])) * 100
                if non_zero.any() else 0.0)

        ss_res = np.sum((st_true - st_pred) ** 2)
        ss_tot = np.sum((st_true - np.mean(st_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

        exact = np.mean(st_true == st_pred)
        test_std = np.std(st_true)
        is_dynamic = train_std[i] > DYNAMIC_THRESHOLD

        per_station[str(sid)] = {
            'MAE': round(float(mae), 4),
            'RMSE': round(float(rmse), 4),
            'MAPE (%)': round(float(mape), 2),
            'R2': round(float(r2), 4),
            'exact_match_rate': round(float(exact), 4),
            'total_connectors': int(total_connectors[i]),
            'train_std': round(float(train_std[i]), 4),
            'test_std': round(float(test_std), 4),
            'category': 'dynamic' if is_dynamic else 'static',
        }
    return per_station


def compute_category_summary(per_station_metrics):
    categories = {}
    for sid, m in per_station_metrics.items():
        cat = m['category']
        categories.setdefault(cat, []).append(m)

    summary = {}
    for cat, stations in categories.items():
        r2_vals = [s['R2'] for s in stations]
        mae_vals = [s['MAE'] for s in stations]
        emr_vals = [s['exact_match_rate'] for s in stations]
        summary[cat] = {
            'n_stations': len(stations),
            'avg_MAE': round(np.mean(mae_vals), 4),
            'avg_RMSE': round(np.mean([s['RMSE'] for s in stations]), 4),
            'avg_R2': round(np.mean(r2_vals), 4),
            'median_R2': round(np.median(r2_vals), 4),
            'avg_exact_match_rate': round(np.mean(emr_vals), 4),
            'n_negative_r2': sum(1 for r in r2_vals if r < 0),
        }
    return summary


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


def fit_and_forecast_station(train_series, test_len, total_conn, station_id):
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(STATION_TIMEOUT)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = auto_arima(
                train_series,
                seasonal=True, m=SEASONAL_PERIOD,
                max_p=MAX_P, max_q=MAX_Q, max_d=MAX_D,
                max_P=MAX_SEASONAL_P, max_Q=MAX_SEASONAL_Q,
                max_D=MAX_SEASONAL_D,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                n_jobs=1,
                maxiter=50,
            )
            preds = model.predict(n_periods=test_len)
        signal.alarm(0)
        preds = np.round(np.clip(preds, 0, total_conn)).astype(int)
        return preds
    except _Timeout:
        print(f"TIMEOUT ({STATION_TIMEOUT}s), using naive fallback", end=' ')
        return _naive_seasonal(train_series, test_len, total_conn)
    except Exception as e:
        signal.alarm(0)
        print(f"failed ({e}), using naive fallback", end=' ')
        return _naive_seasonal(train_series, test_len, total_conn)
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def _naive_seasonal(train_series, test_len, total_conn):
    # just repeat the last seasonal cycle
    cycle = train_series[-SEASONAL_PERIOD:]
    reps = int(np.ceil(test_len / SEASONAL_PERIOD))
    preds = np.tile(cycle, reps)[:test_len]
    return np.round(np.clip(preds, 0, total_conn)).astype(int)


def plot_pred_vs_actual_by_category(y_true, y_pred, station_ids, per_station,
                                    total_connectors, output_dir):
    for category in ['dynamic', 'static']:
        cat_indices = [i for i, sid in enumerate(station_ids)
                       if per_station[str(sid)]['category'] == category]
        if not cat_indices:
            continue

        variances = [np.var(y_true[:, i]) for i in cat_indices]
        order = np.argsort(variances)[::-1]
        show_indices = [cat_indices[j] for j in order[:6]]

        cols = min(3, len(show_indices))
        rows = (len(show_indices) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows),
                                 squeeze=False)

        plot_len = min(FORECAST_HORIZON, y_true.shape[0])
        for idx, si in enumerate(show_indices):
            ax = axes[idx // cols][idx % cols]
            sid = str(station_ids[si])
            m = per_station[sid]
            tc = m['total_connectors']

            ax.plot(range(plot_len), y_true[:plot_len, si],
                    label='Actual', marker='o', markersize=3, color='blue')
            ax.plot(range(plot_len), y_pred[:plot_len, si],
                    label='Predicted', marker='x', markersize=3,
                    linestyle='--', color='orange')
            ax.set_title(f'Station {sid}\nR2={m["R2"]:.3f}, '
                         f'Match={m["exact_match_rate"]*100:.0f}%', fontsize=9)
            ax.set_xlabel('Time Step (15-min)')
            ax.set_ylabel('Connectors')
            ax.set_ylim(-0.5, tc + 0.5)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # hide unused subplots
        for idx in range(len(show_indices), rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        plt.suptitle(f'SARIMA - Predicted vs Actual - {category.upper()} Stations',
                     fontsize=13)
        plt.tight_layout()
        out = os.path.join(output_dir, f'pred_vs_actual_{category}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved {out}")


def plot_per_station_r2(per_station, output_file):
    items = sorted(per_station.items(), key=lambda x: x[1]['R2'], reverse=True)
    sids = [s[0] for s in items]
    r2s = [s[1]['R2'] for s in items]
    colors = ['steelblue' if s[1]['category'] == 'dynamic' else 'lightcoral'
              for s in items]

    fig, ax = plt.subplots(figsize=(14, max(5, len(sids) * 0.25)))
    ax.barh(range(len(sids)), r2s, color=colors)
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels(sids, fontsize=5)
    ax.set_xlabel('R2')
    ax.set_title('SARIMA - Per-Station R2 (blue = dynamic, red = static)')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='steelblue', label='Dynamic'),
                       Patch(facecolor='lightcoral', label='Static')],
              loc='lower right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved {output_file}")


def plot_exact_match_rate(per_station, output_file):
    items = sorted(per_station.items(), key=lambda x: x[1]['exact_match_rate'])
    if len(items) > 40:
        items = items[:40]

    sids = [s[0] for s in items]
    emr = [s[1]['exact_match_rate'] * 100 for s in items]
    colors = ['steelblue' if s[1]['category'] == 'dynamic' else 'lightcoral'
              for s in items]

    fig, ax = plt.subplots(figsize=(12, max(5, len(sids) * 0.3)))
    ax.barh(range(len(sids)), emr, color=colors)
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels(sids, fontsize=7)
    ax.set_xlabel('Exact Match Rate (%)')
    ax.set_title('SARIMA - Per-Station Exact Match Rate')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, 105)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='steelblue', label='Dynamic'),
                       Patch(facecolor='lightcoral', label='Static')],
              loc='lower right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved {output_file}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    print("Loading data...")
    station_data, station_ids, total_connectors = load_and_prepare_data(DATA_FILE)
    n_stations = len(station_ids)
    n_rows = len(station_data)
    data = station_data.values
    print(f"  {n_stations} stations, {n_rows} timesteps")

    train_end = int(TRAIN_SPLIT * n_rows)
    val_end = int((TRAIN_SPLIT + VAL_SPLIT) * n_rows)

    train_data = data[:train_end]
    test_data = data[val_end:]
    test_len = len(test_data)
    print(f"  Train rows: {train_end}, Test rows: {test_len}")

    all_preds = np.zeros((test_len, n_stations), dtype=int)

    for i, sid in enumerate(station_ids):
        tc = int(total_connectors[i])
        train_series = train_data[:, i]

        print(f"  [{i+1}/{n_stations}] Station {sid} "
              f"(total={tc}, train_std={np.std(train_series):.3f})...",
              end=' ', flush=True)

        t0 = time.time()
        preds = fit_and_forecast_station(train_series, test_len, tc, sid)
        elapsed = time.time() - t0
        all_preds[:, i] = preds
        print(f"done ({elapsed:.1f}s)")

    y_true = np.round(test_data).astype(int)
    y_pred = all_preds

    # clip to valid connector range
    for i in range(n_stations):
        y_true[:, i] = np.clip(y_true[:, i], 0, int(total_connectors[i]))

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
    non_zero = y_true_flat != 0
    mape = (np.mean(np.abs((y_true_flat[non_zero] - y_pred_flat[non_zero]) /
                            y_true_flat[non_zero])) * 100
            if non_zero.any() else 0.0)
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    exact_match = np.mean(y_true_flat == y_pred_flat)

    overall = {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'MAPE (%)': round(float(mape), 2),
        'R2': round(float(r2), 4),
        'exact_match_rate': round(float(exact_match), 4),
        'n_stations': n_stations,
        'test_timesteps': test_len,
    }

    total_time = time.time() - start_time
    overall['total_time_seconds'] = round(total_time, 1)

    per_station = evaluate_per_station(y_true, y_pred, station_ids,
                                       total_connectors, train_data, n_stations)
    cat_summary = compute_category_summary(per_station)

    print(f"\n{'='*60}")
    print(f"  SARIMA - OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"  MAE:  {overall['MAE']:.4f} connectors")
    print(f"  RMSE: {overall['RMSE']:.4f} connectors")
    print(f"  R2:   {overall['R2']:.4f}")
    print(f"  Exact Match Rate: {overall['exact_match_rate']*100:.1f}%")
    print(f"  Total time: {total_time:.0f}s")

    print(f"\n{'='*60}")
    print(f"  RESULTS BY STATION CATEGORY (threshold: std > {DYNAMIC_THRESHOLD})")
    print(f"{'='*60}")
    for cat in ['dynamic', 'static']:
        if cat not in cat_summary:
            continue
        s = cat_summary[cat]
        print(f"\n  {cat.upper()} stations ({s['n_stations']} stations):")
        print(f"    Avg MAE:  {s['avg_MAE']:.4f}")
        print(f"    Avg R2:   {s['avg_R2']:.4f} (median: {s['median_R2']:.4f})")
        print(f"    Avg Exact Match Rate: {s['avg_exact_match_rate']*100:.1f}%")
        print(f"    Negative R2: {s['n_negative_r2']} / {s['n_stations']}")

    metrics_out = {
        'overall': overall,
        'category_summary': cat_summary,
        'per_station': per_station,
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=4)
    print(f"\nMetrics saved to {OUTPUT_DIR}/metrics.json")

    print("\nGenerating future predictions...")
    future_preds = np.zeros((FORECAST_HORIZON, n_stations), dtype=int)
    for i, sid in enumerate(station_ids):
        tc = int(total_connectors[i])
        full_series = data[:, i]
        fp = fit_and_forecast_station(full_series, FORECAST_HORIZON, tc, sid)
        future_preds[:, i] = fp

    timestamps = pd.date_range(
        start=station_data.index[-1] + pd.Timedelta(RESAMPLE_INTERVAL),
        periods=FORECAST_HORIZON, freq=RESAMPLE_INTERVAL,
    )
    pred_df = pd.DataFrame(future_preds, columns=station_ids, index=timestamps)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'))
    print("Predictions saved.")

    print("\nGenerating plots...")
    plot_pred_vs_actual_by_category(y_true, y_pred, station_ids, per_station,
                                    total_connectors, OUTPUT_DIR)
    plot_per_station_r2(per_station, os.path.join(OUTPUT_DIR, 'per_station_r2.png'))
    plot_exact_match_rate(per_station,
                          os.path.join(OUTPUT_DIR, 'exact_match_rate.png'))

    print(f"\nDone. Total time: {time.time() - start_time:.0f}s")


if __name__ == '__main__':
    main()
