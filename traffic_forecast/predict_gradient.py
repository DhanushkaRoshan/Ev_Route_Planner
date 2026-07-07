import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import time
from sklearn.model_selection import KFold
import xgboost as xgb
import osmnx as ox
import networkx as nx

from utility import load_filtered_data, to_map, graph_from_place
from evaluation import summarize_cv

os.makedirs('output', exist_ok=True)

TIME_START = time(8, 0, 0)
TIME_END = time(8, 30, 0)
N_SPLITS = 5
RANDOM_STATE = 42
EPSG_PROJ = 32610

XGB_PARAMS = dict(
    objective='reg:squarederror',
    n_estimators=50,
    learning_rate=0.1,
    max_depth=2,
    min_child_weight=3,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
)


def load_or_compute(path, compute_fn):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    result = compute_fn()
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    return result


def main():
    print("Loading 8:00-8:30 AM observations ...")
    df = load_filtered_data(TIME_START, TIME_END)
    df = df.groupby(['Latitude', 'Longitude'], as_index=False)['Count'].mean()
    print(f"  {len(df)} unique sensor locations")

    print("Loading road network ...")
    G = graph_from_place()
    roads = ox.graph_to_gdfs(G, nodes=False, edges=True).to_crs(epsg=EPSG_PROJ)

    roads['road_type'] = roads['highway'].apply(lambda v: v[0] if isinstance(v, list) else v)
    roads['length'] = roads.length
    roads['lanes'] = roads['lanes'].apply(
        lambda v: int(v[0]) if isinstance(v, list)
        else (int(v) if pd.notnull(v) else 1)
    )

    # network centrality features (cached to pkl)
    print("Computing network centrality features ...")
    betweenness = load_or_compute(
        'output/betweenness_cache.pkl',
        lambda: nx.edge_betweenness_centrality(G, k=500, seed=RANDOM_STATE)
    )
    node_deg = load_or_compute(
        'output/degree_centrality_cache.pkl',
        lambda: nx.degree_centrality(G)
    )
    node_cls = load_or_compute(
        'output/closeness_centrality_cache.pkl',
        lambda: nx.closeness_centrality(G)
    )

    bc_dict = {(u, v, k): val for (u, v, k), val in betweenness.items()}
    roads['betweenness'] = roads.apply(lambda r: bc_dict.get(r.name, 0), axis=1)
    roads['degree'] = roads.apply(
        lambda r: (node_deg.get(r.name[0], 0) + node_deg.get(r.name[1], 0)) / 2, axis=1)
    roads['closeness'] = roads.apply(
        lambda r: (node_cls.get(r.name[0], 0) + node_cls.get(r.name[1], 0)) / 2, axis=1)

    # spatial join sensors to nearest road
    df['sid'] = np.arange(len(df))
    geom = [Point(lon, lat) for lat, lon in zip(df.Latitude, df.Longitude)]
    pts = gpd.GeoDataFrame(df, geometry=geom, crs='EPSG:4326').to_crs(epsg=EPSG_PROJ)

    NET_COLS = ['geometry', 'road_type', 'length', 'lanes',
                'betweenness', 'degree', 'closeness']
    joined = gpd.sjoin_nearest(pts, roads[NET_COLS], how='left')
    joined = joined.drop_duplicates(subset=['sid']).reset_index(drop=True)

    # one-hot encode road types
    all_road_types = roads['road_type'].unique()
    dummy_cols = pd.get_dummies(all_road_types, prefix='road').columns

    joined['x'] = joined.geometry.x
    joined['y'] = joined.geometry.y
    roads['x'] = roads.geometry.centroid.x
    roads['y'] = roads.geometry.centroid.y

    for part in (joined, roads):
        d = pd.get_dummies(part['road_type'], prefix='road')
        for col in dummy_cols:
            part[col] = d.get(col, 0)

    FEAT_COLS = ['x', 'y', 'length', 'lanes',
                 'betweenness', 'degree', 'closeness'] + list(dummy_cols)

    X_train = joined[FEAT_COLS]
    y_train = joined['Count'].astype(float).values
    X_pred = roads[FEAT_COLS]
    sigma_y = float(y_train.std(ddof=1))

    # cross validation
    print(f"\n{N_SPLITS}-fold CV ...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pairs = []
    for tr_idx, te_idx in kf.split(X_train):
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train.iloc[tr_idx], y_train[tr_idx], verbose=False)
        y_hat = model.predict(X_train.iloc[te_idx])
        pairs.append((y_train[te_idx], y_hat))

    summarize_cv(pairs, sigma_y, 'XGBoost Gradient Boosting')

    # train final model on all data
    print("\nTraining final model ...")
    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_train, y_train, verbose=False)

    z_pred = np.clip(final_model.predict(X_pred), 0, None)
    roads['pred_count'] = np.rint(z_pred).astype(int)

    to_map(roads, 'gradient_6yr')
    print("Done -> output/gradient_6yr.html")


if __name__ == '__main__':
    main()
