import os
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from datetime import time
from sklearn.model_selection import KFold

from utility    import load_filtered_data, road_type_weights, to_map, graph_from_place
from evaluation import summarize_cv

os.makedirs('output', exist_ok=True)

TIME_START = time(8, 0, 0)
TIME_END = time(8, 30, 0)
N_SPLITS = 5
RANDOM_STATE = 42
EPSG_PROJ = 32610
POWER_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]


def idw_predict(x_tr, y_tr, z_tr, x_query, y_query, p):
    train_pts = np.column_stack([x_tr, y_tr])
    query_pts = np.column_stack([x_query, y_query])
    z_hat = np.empty(len(query_pts))

    for i, q in enumerate(query_pts):
        d = np.linalg.norm(train_pts - q, axis=1)
        # if query point is exactly on a training point, just use that value
        if np.any(d == 0):
            z_hat[i] = z_tr[np.argmin(d)]
        else:
            w = 1.0 / (d ** p)
            z_hat[i] = (w * z_tr).sum() / w.sum()

    return z_hat


def cv_predictions(x, y, z, p, kf):
    pairs = []
    for tr, te in kf.split(x):
        z_hat = idw_predict(x[tr], y[tr], z[tr], x[te], y[te], p)
        pairs.append((z[te], z_hat))
    return pairs


def main():
    print("Loading 8:00-8:30 AM observations from 4yr dataset ...")
    df = load_filtered_data(TIME_START, TIME_END)
    df = df.groupby(['Latitude', 'Longitude'], as_index=False)['Count'].mean()
    print(f"  {len(df)} unique sensor locations after averaging")

    geom = [Point(lon, lat) for lat, lon in zip(df.Latitude, df.Longitude)]
    points_gdf = gpd.GeoDataFrame(df, geometry=geom, crs='EPSG:4326').to_crs(epsg=EPSG_PROJ)
    x = points_gdf.geometry.x.values
    y = points_gdf.geometry.y.values
    z = points_gdf['Count'].values.astype(float)
    sigma_y = float(z.std(ddof=1))

    # try different power values
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print(f"\nTuning IDW power p over {POWER_GRID} via pooled {N_SPLITS}-fold CV ...")
    grid = {}
    for p in POWER_GRID:
        pairs = cv_predictions(x, y, z, p, kf)
        yt = np.concatenate([t for t, _ in pairs])
        yp = np.concatenate([h for _, h in pairs])
        rmse_val = float(np.sqrt(np.mean((yt - yp) ** 2)))
        grid[p] = (rmse_val, pairs)
        print(f"  p = {p:>3}  ->  pooled RMSE = {rmse_val:7.2f}")

    best_p = min(grid, key=lambda p: grid[p][0])
    print(f"\nBest p = {best_p}")

    summarize_cv(grid[best_p][1], sigma_y, f"IDW (p = {best_p})")

    # predict on all road segments
    print("\nPredicting at all road segment midpoints ...")
    graph = graph_from_place()
    roads = ox.graph_to_gdfs(graph, nodes=False, edges=True).to_crs(epsg=EPSG_PROJ)
    midpts = roads.geometry.centroid
    z_pred = idw_predict(x, y, z, midpts.x.values, midpts.y.values, best_p)

    # apply road type weighting
    rt_map = road_type_weights()
    road_types = roads['highway'].apply(lambda v: v[0] if isinstance(v, list) else v)
    weights = road_types.fillna('residential').map(lambda rt: rt_map.get(rt, 1.0)).values
    z_pred = np.clip(z_pred * weights, 0, None)

    roads['pred_count'] = np.rint(z_pred).astype(int)
    to_map(roads, 'idw_6yr')
    print("Done -> output/idw_6yr.html")


if __name__ == '__main__':
    main()
