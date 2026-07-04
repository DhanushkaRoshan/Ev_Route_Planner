import numpy as np


def fold_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def summarize_cv(fold_pairs, sigma_y, model_name):
    # pool all folds together
    pooled_true = np.concatenate([yt for yt, _ in fold_pairs])
    pooled_pred = np.concatenate([yp for _, yp in fold_pairs])

    mae_pool = float(np.mean(np.abs(pooled_true - pooled_pred)))
    rmse_pool = float(np.sqrt(np.mean((pooled_true - pooled_pred) ** 2)))

    ss_res = float(np.sum((pooled_true - pooled_pred) ** 2))
    ss_tot = float(np.sum((pooled_true - pooled_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    per_fold_mae = np.array([np.mean(np.abs(yt - yp)) for yt, yp in fold_pairs])
    per_fold_rmse = np.array([np.sqrt(np.mean((yt - yp) ** 2)) for yt, yp in fold_pairs])

    summary = {
        'mae': mae_pool,
        'rmse': rmse_pool,
        'r2': r2,
        'mae_std': float(per_fold_mae.std()),
        'rmse_std': float(per_fold_rmse.std()),
        'std_mae': mae_pool / sigma_y if sigma_y > 0 else float('nan'),
        'std_rmse': rmse_pool / sigma_y if sigma_y > 0 else float('nan'),
    }

    bar = '-' * 64
    print(f"\n{bar}\n {model_name} - pooled 5-fold CV  (sigma_y = {sigma_y:.2f})\n{bar}")
    print(f"  MAE        : {summary['mae']:8.2f}   (fold std +/- {summary['mae_std']:.2f})")
    print(f"  RMSE       : {summary['rmse']:8.2f}   (fold std +/- {summary['rmse_std']:.2f})")
    print(f"  R2         : {summary['r2']:8.4f}")
    print(f"  Std. MAE   : {summary['std_mae']:8.4f}")
    print(f"  Std. RMSE  : {summary['std_rmse']:8.4f}")
    print(bar)

    return summary
