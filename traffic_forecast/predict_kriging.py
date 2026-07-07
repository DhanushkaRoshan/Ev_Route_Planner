import os
import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from datetime import time
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import KFold
from concurrent.futures import ProcessPoolExecutor

from utility import load_filtered_data, road_type_weights, to_map, graph_from_place
from evaluation import summarize_cv

os.makedirs('output', exist_ok=True)

TIME_START = time(8, 0, 0)
TIME_END = time(8, 30, 0)
N_SPLITS = 5
RANDOM_STATE = 42
N_CORES = min(os.cpu_count() or 4, N_SPLITS)
EPSG_PROJ = 32610
VARIO_MODEL = 'exponential'
NLAGS = 6


def clean_highway(val):
    if isinstance(val, list):
        return val[0]
    return val if pd.notna(val) else 'residential'


def krige_residuals(x_tr, y_tr, resid_tr, x_pred, y_pred):
    ok = OrdinaryKriging(
        x_tr, y_tr, resid_tr,
        variogram_model=VARIO_MODEL,
        nlags=NLAGS,
        weight=True,
        exact_values=True,
        verbose=False,
        enable_plotting=False,
    )
    z_hat, _ = ok.execute('points', x_pred, y_pred)
    return z_hat.data.flatten()


def regression_kriging(x_tr, y_tr, z_tr, rt_tr, x_pred, y_pred, rt_pred):
    # trend component: mean count per road type
    type_mean = pd.Series(z_tr).groupby(rt_tr).mean().to_dict()
    global_mean = float(z_tr.mean())
    trend_tr = np.array([type_mean.get(r, global_mean) for r in rt_tr])
    trend_pred = np.array([type_mean.get(r, global_mean) for r in rt_pred])

    # krige the residuals
    resid_tr = z_tr - trend_tr
    resid_hat = krige_residuals(x_tr, y_tr, resid_tr, x_pred, y_pred)
    return np.clip(trend_pred + resid_hat, 0, None)


def run_cv_fold(args):
    tr_idx, te_idx, x, y, z, rt = args
    z_hat = regression_kriging(
        x[tr_idx], y[tr_idx], z[tr_idx], rt[tr_idx],
        x[te_idx], y[te_idx], rt[te_idx],
    )
    return z[te_idx], z_hat


def predict_chunk(args):
    x_tr, y_tr, z_tr, rt_tr, x_chunk, y_chunk, rt_chunk = args
    return regression_kriging(x_tr, y_tr, z_tr, rt_tr,
                               x_chunk, y_chunk, rt_chunk)


def chunkify(arr, n):
    k, m = divmod(len(arr), n)
    return [arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main():
    print("Loading 8:00-8:30 AM observations ...")
    df = load_filtered_data(TIME_START, TIME_END)
    df = df.groupby(['Latitude', 'Longitude'], as_index=False)['Count'].mean()
    df['sid'] = np.arange(len(df))
    print(f"  {len(df)} unique sensor locations")

    geom = [Point(lon, lat) for lat, lon in zip(df.Latitude, df.Longitude)]
    pts = gpd.GeoDataFrame(df, geometry=geom, crs='EPSG:4326').to_crs(epsg=EPSG_PROJ)

    graph = graph_from_place()
    roads = ox.graph_to_gdfs(graph, nodes=False, edges=True).to_crs(epsg=EPSG_PROJ)
    roads['road_type'] = roads['highway'].apply(clean_highway)

    joined = gpd.sjoin_nearest(pts, roads[['geometry', 'road_type']], how='left')
    joined = joined.drop_duplicates(subset=['sid']).reset_index(drop=True)
    print(f"  matched {len(joined)} sensors to road types")

    x = joined.geometry.x.values
    y = joined.geometry.y.values
    z = joined.Count.values.astype(float)
    rt = joined.road_type.fillna('residential').values
    sigma_y = float(z.std(ddof=1))

    # cross validation with parallel folds
    print(f"\nRunning {N_SPLITS}-fold CV ({N_CORES} cores) ...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_args = [(tr, te, x, y, z, rt) for tr, te in kf.split(x)]

    with ProcessPoolExecutor(max_workers=N_CORES) as pool:
        pairs = list(pool.map(run_cv_fold, cv_args))

    summarize_cv(pairs, sigma_y, 'Regression Kriging (road-type trend + OK residuals)')

    # print variogram parameters from full model
    type_mean = pd.Series(z).groupby(rt).mean().to_dict()
    global_mean = float(z.mean())
    trend_full = np.array([type_mean.get(r, global_mean) for r in rt])
    resid_full = z - trend_full
    ok_full = OrdinaryKriging(
        x, y, resid_full, variogram_model=VARIO_MODEL, nlags=NLAGS,
        weight=True, exact_values=True, verbose=False, enable_plotting=False,
    )
    sill, rng, nugget = ok_full.variogram_model_parameters
    print(f"\nVariogram ({VARIO_MODEL}): sill={sill:.1f}, range={rng:.1f}m, nugget={nugget:.1f}")

    # predict all road segments
    print("\nPredicting on road segments ...")
    midpts = roads.geometry.centroid
    xp = midpts.x.values
    yp = midpts.y.values
    rt_pred = roads['road_type'].fillna('residential').values

    x_chunks = chunkify(xp, N_CORES)
    y_chunks = chunkify(yp, N_CORES)
    rt_chunks = chunkify(rt_pred, N_CORES)
    args = [(x, y, z, rt, x_chunks[i], y_chunks[i], rt_chunks[i])
            for i in range(len(x_chunks))]

    with ProcessPoolExecutor(max_workers=N_CORES) as pool:
        chunks = list(pool.map(predict_chunk, args))

    z_pred = np.concatenate(chunks)

    wt_map = road_type_weights()
    weights = np.array([wt_map.get(r, 1.0) for r in rt_pred])
    z_pred = np.clip(z_pred * weights, 0, None)

    roads['pred_count'] = np.rint(z_pred).astype(int)
    to_map(roads, 'kriging_6yr')
    print("Done -> output/kriging_6yr.html")


if __name__ == '__main__':
    main()
