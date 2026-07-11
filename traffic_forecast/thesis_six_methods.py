import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
from datetime import time
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pykrige.ok import OrdinaryKriging
import xgboost as xgb

from utility import load_filtered_data, road_type_weights, graph_from_place

warnings.filterwarnings('ignore')
os.makedirs('thesis_outputs', exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans',
})

TIME_START = time(8, 0, 0)
TIME_END = time(8, 30, 0)
RANDOM_STATE = 42
N_SPLITS = 5
EPSG_PROJ = 32610
WINNIPEG_CBD = (49.8951, -97.1384)


def parse_speed(v, default=50):
    if isinstance(v, list):
        v = v[0]
    if pd.isna(v):
        return default
    m = re.search(r'(\d+)', str(v))
    if m:
        return int(m.group(1))
    return default


def load_or_compute(path, fn):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    result = fn()
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    return result


def prepare_dataset():
    print('Loading 6-year weekday 8:00-8:30 AM observations ...')
    df = load_filtered_data(TIME_START, TIME_END)
    df = df[df.DateTime.dt.dayofweek < 5]
    df = df.groupby(['Latitude', 'Longitude'], as_index=False)['Count'].mean()
    df['sid'] = range(len(df))
    print(f'  {len(df)} unique sensor locations')

    geom = [Point(lon, lat) for lat, lon in zip(df.Latitude, df.Longitude)]
    pg = gpd.GeoDataFrame(df, geometry=geom, crs='EPSG:4326').to_crs(EPSG_PROJ)

    # get CBD coordinates in projected CRS
    cbd_gdf = gpd.GeoDataFrame(
        geometry=[Point(WINNIPEG_CBD[1], WINNIPEG_CBD[0])],
        crs='EPSG:4326').to_crs(EPSG_PROJ)
    cbd_x = cbd_gdf.geometry.x.iloc[0]
    cbd_y = cbd_gdf.geometry.y.iloc[0]

    print('Loading road network ...')
    G = graph_from_place()
    roads = ox.graph_to_gdfs(G, nodes=False, edges=True).to_crs(EPSG_PROJ)

    # extract road features
    roads['road_type'] = roads['highway'].apply(
        lambda v: v[0] if isinstance(v, list) else v)
    roads['length'] = roads.length
    roads['log_len'] = np.log1p(roads['length'])
    roads['lanes'] = roads['lanes'].apply(
        lambda v: int(v[0]) if isinstance(v, list)
        else (int(v) if pd.notnull(v) else 2))
    roads['maxspeed'] = roads['maxspeed'].apply(parse_speed)
    roads['oneway'] = roads['oneway'].astype(int)
    roads['has_name'] = roads['name'].notna().astype(int)

    # centrality measures (cached)
    bc = load_or_compute('output/betweenness_cache.pkl',
                          lambda: nx.edge_betweenness_centrality(G, k=500, seed=RANDOM_STATE))
    nd = load_or_compute('output/degree_centrality_cache.pkl',
                          lambda: nx.degree_centrality(G))
    nc = load_or_compute('output/closeness_centrality_cache.pkl',
                          lambda: nx.closeness_centrality(G))

    bc_d = {(u, v, k): val for (u, v, k), val in bc.items()}
    roads['betweenness'] = roads.apply(lambda r: bc_d.get(r.name, 0), axis=1)
    roads['degree'] = roads.apply(
        lambda r: (nd.get(r.name[0], 0) + nd.get(r.name[1], 0)) / 2, axis=1)
    roads['closeness'] = roads.apply(
        lambda r: (nc.get(r.name[0], 0) + nc.get(r.name[1], 0)) / 2, axis=1)

    # spatial join sensors to roads
    NET = ['geometry', 'road_type', 'length', 'log_len', 'lanes', 'maxspeed',
           'oneway', 'has_name', 'betweenness', 'degree', 'closeness']
    joined = (gpd.sjoin_nearest(pg, roads[NET], how='left')
                .drop_duplicates(subset=['sid'])
                .reset_index(drop=True))
    joined['x'] = joined.geometry.x
    joined['y'] = joined.geometry.y
    joined['dist_cbd'] = np.sqrt(
        (joined.x - cbd_x) ** 2 + (joined.y - cbd_y) ** 2)

    return joined, roads


def idw_predict(x_tr, y_tr, z_tr, x_q, y_q, p=2.0):
    train_coords = np.column_stack([x_tr, y_tr])
    result = np.empty(len(x_q))
    for i, q in enumerate(np.column_stack([x_q, y_q])):
        d = np.linalg.norm(train_coords - q, axis=1)
        if np.any(d == 0):
            result[i] = z_tr[np.argmin(d)]
        else:
            w = 1 / (d ** p)
            result[i] = (w * z_tr).sum() / w.sum()
    return result


def cv_idw(j):
    x = j.x.values
    y = j.y.values
    z = j.Count.values.astype(float)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pairs = []
    for tr, te in kf.split(x):
        zhat = idw_predict(x[tr], y[tr], z[tr], x[te], y[te], p=2.0)
        pairs.append((z[te], np.clip(zhat, 0, None)))
    return pairs


def cv_kriging(j):
    x = j.x.values
    y = j.y.values
    z = j.Count.values.astype(float)
    rt = j.road_type.fillna('residential').values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pairs = []

    for tr, te in kf.split(x):
        # road-type trend
        type_means = pd.Series(z[tr]).groupby(rt[tr]).mean().to_dict()
        gmean = float(z[tr].mean())
        trend_tr = np.array([type_means.get(r, gmean) for r in rt[tr]])
        trend_te = np.array([type_means.get(r, gmean) for r in rt[te]])
        resid = z[tr] - trend_tr

        try:
            ok = OrdinaryKriging(x[tr], y[tr], resid,
                                  variogram_model='exponential', nlags=6,
                                  weight=True, exact_values=True,
                                  verbose=False, enable_plotting=False)
            rh, _ = ok.execute('points', x[te], y[te])
            zhat = np.clip(trend_te + rh.data.flatten(), 0, None)
        except Exception:
            # fallback to just the trend if kriging fails
            zhat = np.clip(trend_te, 0, None)
        pairs.append((z[te], zhat))
    return pairs


def cv_tabular(j, all_road_types, features, model_factory):
    # create dummy columns
    dums = pd.get_dummies(all_road_types, prefix='rt').columns
    d = pd.get_dummies(j['road_type'], prefix='rt')
    for c in dums:
        j[c] = d.get(c, 0)
    FEATS = features + list(dums)

    X = j[FEATS].values.astype(float)
    y = j['Count'].astype(float).values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pairs = []
    for tr, te in kf.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds = np.clip(m.predict(X[te]), 0, None)
        pairs.append((y[te], preds))
    return pairs


def gb_factory():
    return xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=50, learning_rate=0.1,
        max_depth=2, min_child_weight=3, reg_alpha=1.0, reg_lambda=2.0,
        random_state=RANDOM_STATE)


def knn_factory():
    return Pipeline([('sc', StandardScaler()),
                     ('m',  KNeighborsRegressor(n_neighbors=7, weights='distance'))])


def pooled_summary(pairs, label):
    yt = np.concatenate([t for t, _ in pairs])
    yp = np.concatenate([p for _, p in pairs])
    sigma_y = float(yt.std(ddof=1))
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    tss  = float(np.sum((yt - yt.mean()) ** 2))

    if tss > 0:
        r2 = float(1 - np.sum((yt - yp) ** 2) / tss)
    else:
        r2 = 0.0

    return {
        'method': label,
        'R2': round(r2, 4),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'std_RMSE': round(rmse / sigma_y, 4),
        '_yt': yt, '_yp': yp,
    }


def main():
    j_master, roads = prepare_dataset()
    all_rt = roads.road_type.unique()

    BASE = ['x', 'y', 'length', 'log_len', 'lanes', 'maxspeed',
            'oneway', 'has_name', 'dist_cbd']
    NETF = ['betweenness', 'degree', 'closeness']

    print('\nRunning 5-fold pooled CV for the 6 methods ...\n')

    results = []

    print('  1/6  IDW (p=2)')
    results.append(pooled_summary(cv_idw(j_master.copy()), 'IDW'))

    print('  2/6  Regression Kriging')
    results.append(pooled_summary(cv_kriging(j_master.copy()),
                                    'Regression Kriging'))

    print('  3/6  Gradient Boosting (base features)')
    results.append(pooled_summary(
        cv_tabular(j_master.copy(), all_rt, BASE, gb_factory),
        'Gradient Boosting'))

    print('  4/6  Gradient Boosting + network')
    results.append(pooled_summary(
        cv_tabular(j_master.copy(), all_rt, BASE + NETF, gb_factory),
        'Gradient Boosting + Network'))

    print('  5/6  KNN (base features)')
    results.append(pooled_summary(
        cv_tabular(j_master.copy(), all_rt, BASE, knn_factory),
        'KNN'))

    print('  6/6  KNN + network')
    results.append(pooled_summary(
        cv_tabular(j_master.copy(), all_rt, BASE + NETF, knn_factory),
        'KNN + Network'))

    # build comparison table
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                        for r in results])
    df = df.sort_values('R2', ascending=False).reset_index(drop=True)
    df.insert(0, 'rank', np.arange(1, len(df) + 1))
    df.to_csv('thesis_outputs/table01_six_methods.csv', index=False)

    print('\nMethod comparison (sorted):')
    print(df.to_string(index=False))

    # bar chart of R2 scores
    plot_df = df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(plot_df)))
    bars = ax.barh(plot_df['method'], plot_df['R2'],
                    color=bar_colors, edgecolor='black', linewidth=0.5)
    for b, val in zip(bars, plot_df['R2']):
        ax.text(val + 0.005, b.get_y() + b.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10)
    ax.axvline(0.0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Pooled 5-fold CV $R^2$')
    ax.set_xlim(0, max(plot_df.R2) * 1.18)
    ax.set_title('Volume-prediction methods - weekday 08:00-08:30 AM, '
                  'six-year data ($n = 367$ sensors)')
    plt.savefig('thesis_outputs/fig03_six_methods.png')
    plt.close()
    print('\n  Saved fig03_six_methods.png')

    # scatter plots for top 2 methods
    rs = sorted(results, key=lambda r: -r['R2'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, r in zip(axes, rs[:2]):
        yt = r['_yt']
        yp = r['_yp']
        ax.scatter(yt, yp, s=22, alpha=0.6,
                    edgecolor='black', linewidth=0.3,
                    c=plt.cm.viridis(0.4))
        upper_lim = max(yt.max(), yp.max()) * 1.05
        ax.plot([0, upper_lim], [0, upper_lim], 'k--', linewidth=1, label='1:1')
        ax.set_xlim(0, upper_lim)
        ax.set_ylim(0, upper_lim)
        ax.set_xlabel('Observed mean 15-min count')
        ax.set_ylabel('Predicted (out-of-fold)')
        ax.set_title(f"{r['method']}  ($R^2 = {r['R2']:.3f}$, "
                      f"MAE = {r['MAE']:.1f})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
    fig.suptitle('Predicted vs observed - top-2 methods', y=1.02)
    plt.savefig('thesis_outputs/fig10_six_methods_pred_obs.png')
    plt.close()
    print('  Saved fig10_six_methods_pred_obs.png')


if __name__ == '__main__':
    main()
