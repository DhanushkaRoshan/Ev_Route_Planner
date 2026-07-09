import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from branca.element import Template, MacroElement
from shapely.geometry import Point
from datetime import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import osmnx as ox
import networkx as nx

from utility import load_filtered_data, graph_from_place

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)
os.makedirs('data', exist_ok=True)

TIME_START = time(8, 0, 0)
TIME_END = time(8, 30, 0)
RANDOM_STATE = 42
EPSG_PROJ = 32610
WINNIPEG_CBD = (49.8951, -97.1384)

KNN_K = 7
KNN_WEIGHTS = 'distance'
RIDGE_ALPHA = 10.0
KNN_WEIGHT = 0.75
RIDGE_WEIGHT = 0.25

# BPR function parameters
BPR_ALPHA = 0.15
BPR_BETA = 4.0

VOLUME_PEAK_HOUR_FACTOR = 4

LOS_THRESHOLDS = [
    (0.60, 'A'),
    (0.70, 'B'),
    (0.80, 'C'),
    (0.90, 'D'),
    (1.00, 'E'),
    (np.inf, 'F'),
]
LOS_COLORS = {
    'A': '#1a9850', 'B': '#66bd63', 'C': '#a6d96a',
    'D': '#fee08b', 'E': '#f46d43', 'F': '#a50026',
}


def parse_speed(v, default=50):
    if isinstance(v, list):
        v = v[0]
    if pd.isna(v):
        return default
    m = re.search(r'(\d+)', str(v))
    return int(m.group(1)) if m else default


def load_or_compute(path, fn):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    out = fn()
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    return out


def los_for_vc(vc):
    for thresh, letter in LOS_THRESHOLDS:
        if vc <= thresh:
            return letter
    return 'F'


def train_and_predict_volume(roads, cbd_x, cbd_y):
    print("Training KNN+Ridge volume model on weekday 8:00-8:30 AM ...")
    df = load_filtered_data(TIME_START, TIME_END)
    df = df[df['DateTime'].dt.dayofweek < 5]
    df = df.groupby(['Latitude', 'Longitude'], as_index=False)['Count'].mean()
    df['sid'] = np.arange(len(df))

    geom = [Point(lon, lat) for lat, lon in zip(df.Latitude, df.Longitude)]
    pts_gdf = gpd.GeoDataFrame(df, geometry=geom, crs='EPSG:4326').to_crs(epsg=EPSG_PROJ)

    NET_COLS = ['geometry', 'road_type', 'length', 'log_len', 'lanes',
                'maxspeed', 'oneway', 'has_name',
                'betweenness', 'degree', 'closeness']
    joined = gpd.sjoin_nearest(pts_gdf, roads[NET_COLS], how='left')
    joined = joined.drop_duplicates(subset=['sid']).reset_index(drop=True)
    joined['x'] = joined.geometry.x
    joined['y'] = joined.geometry.y
    joined['dist_cbd'] = np.sqrt((joined.x - cbd_x)**2 + (joined.y - cbd_y)**2)

    all_road_types = roads['road_type'].unique()
    dummy_cols = pd.get_dummies(all_road_types, prefix='road').columns
    for part in (joined, roads):
        d = pd.get_dummies(part['road_type'], prefix='road')
        for col in dummy_cols:
            part[col] = d.get(col, 0)

    FEAT_COLS = ['x', 'y', 'length', 'log_len', 'lanes', 'maxspeed',
                 'oneway', 'has_name', 'dist_cbd',
                 'betweenness', 'degree', 'closeness'] + list(dummy_cols)

    X_train = joined[FEAT_COLS].values.astype(float)
    y_train = joined['Count'].astype(float).values
    X_pred = roads[FEAT_COLS].values.astype(float)

    # ensemble of KNN and Ridge
    knn = Pipeline([('sc', StandardScaler()),
                    ('m', KNeighborsRegressor(n_neighbors=KNN_K, weights=KNN_WEIGHTS))])
    ridge = Pipeline([('sc', StandardScaler()),
                      ('m', Ridge(alpha=RIDGE_ALPHA))])
    knn.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    pred = KNN_WEIGHT * knn.predict(X_pred) + RIDGE_WEIGHT * ridge.predict(X_pred)
    return np.clip(pred, 0, None)


def prepare_roads():
    print("Loading Winnipeg road network ...")
    G = graph_from_place()
    roads = ox.graph_to_gdfs(G, nodes=False, edges=True).to_crs(epsg=EPSG_PROJ)

    roads['road_type'] = roads['highway'].apply(
        lambda v: v[0] if isinstance(v, list) else v)
    roads['length'] = roads.length
    roads['log_len'] = np.log1p(roads['length'])
    roads['lanes'] = roads['lanes'].apply(
        lambda v: int(v[0]) if isinstance(v, list)
        else (int(v) if pd.notnull(v) else 2))
    roads['maxspeed_kmh'] = roads['maxspeed'].apply(parse_speed)
    roads['oneway'] = roads['oneway'].astype(int)
    roads['has_name'] = roads['name'].notna().astype(int)
    roads['x'] = roads.geometry.centroid.x
    roads['y'] = roads.geometry.centroid.y

    # distance from downtown
    cbd_gdf = gpd.GeoDataFrame(
        geometry=[Point(WINNIPEG_CBD[1], WINNIPEG_CBD[0])],
        crs='EPSG:4326').to_crs(epsg=EPSG_PROJ)
    cbd_x, cbd_y = cbd_gdf.geometry.x.iloc[0], cbd_gdf.geometry.y.iloc[0]
    roads['dist_cbd'] = np.sqrt((roads.x - cbd_x)**2 + (roads.y - cbd_y)**2)

    # network centrality
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

    roads['maxspeed'] = roads['maxspeed_kmh']
    return roads, cbd_x, cbd_y


def attach_capacity(roads, cap_csv='data/road_capacity_canada.csv'):
    cap = pd.read_csv(cap_csv).set_index('highway')
    default_cap = 400
    default_speed = 30

    rt = roads['road_type']
    roads['cap_vph_per_lane'] = rt.map(cap['capacity_vph_per_lane']).fillna(default_cap)
    roads['cap_ff_speed_kmh'] = rt.map(cap['free_flow_speed_kmh']).fillna(default_speed)

    osm_speed = roads['maxspeed_kmh']
    roads['ff_speed_kmh'] = np.where(osm_speed > 0, osm_speed, roads['cap_ff_speed_kmh'])
    roads['capacity_vph'] = roads['cap_vph_per_lane'] * roads['lanes']
    return roads


def apply_bpr(roads):
    V = roads['pred_count_15min'].values * VOLUME_PEAK_HOUR_FACTOR
    C = roads['capacity_vph'].values
    L_m = roads['length'].values
    v_ff_kmh = roads['ff_speed_kmh'].values
    v_ff_ms = v_ff_kmh / 3.6

    t0_s = L_m / np.where(v_ff_ms > 0, v_ff_ms, 1e-9)
    vc = V / np.where(C > 0, C, 1e-9)
    t_s = t0_s * (1.0 + BPR_ALPHA * np.power(vc, BPR_BETA))
    delay_s = t_s - t0_s
    v_cong_kmh = (L_m / np.where(t_s > 0, t_s, 1e-9)) * 3.6

    roads['volume_vph'] = V
    roads['vc_ratio'] = vc
    roads['t0_seconds'] = t0_s
    roads['t_congested_s'] = t_s
    roads['delay_seconds'] = delay_s
    roads['v_congested_kmh'] = v_cong_kmh
    roads['LOS'] = pd.Series(vc).apply(los_for_vc).values
    return roads


def make_legend(rows):
    html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index:9999;
        background: white; padding: 10px; border: 2px solid grey;
        font-family: Arial; font-size: 13px; line-height: 18px;">
        <b>""" + rows['title'] + """</b><br>
    """ + ''.join(
        f'<div style="display:flex;align-items:center;margin-top:4px;">'
        f'<div style="width:20px;height:12px;background:{color};margin-right:8px;"></div>'
        f'{label}</div>' for color, label in rows['items']
    ) + "</div>{% endmacro %}"
    el = MacroElement()
    el._template = Template(html)
    return el


def map_los(roads_gdf, out_path):
    g = roads_gdf.to_crs(epsg=4326).copy()
    g['__color__'] = g['LOS'].map(LOS_COLORS)
    minx, miny, maxx, maxy = g.total_bounds

    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2],
                   zoom_start=11, tiles='cartodbpositron')
    folium.GeoJson(
        g,
        style_function=lambda feat: {
            'color': feat['properties']['__color__'],
            'weight': 3, 'opacity': 0.85,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['road_type', 'LOS', 'vc_ratio', 'volume_vph', 'capacity_vph',
                    'ff_speed_kmh', 'v_congested_kmh', 'delay_seconds'],
            aliases=['Road type:', 'LOS:', 'V/C:', 'Volume vph:', 'Capacity vph:',
                     'Free-flow km/h:', 'Congested km/h:', 'Delay (s):'],
            localize=True, sticky=False, max_width=300,
        ),
    ).add_to(m)

    items = [(LOS_COLORS[k], f'{k} - {d}') for k, d in [
        ('A', 'Free flow (V/C <= 0.60)'),
        ('B', 'Reasonably free (<= 0.70)'),
        ('C', 'Stable (<= 0.80)'),
        ('D', 'Approaching unstable (<= 0.90)'),
        ('E', 'At capacity (<= 1.00)'),
        ('F', 'Breakdown (> 1.00)')]]
    rows = {'title': 'Level of Service (HCM)', 'items': items}
    m.get_root().add_child(make_legend(rows))
    m.save(out_path)
    print(f"  Map -> {out_path}")


def map_travel_time(roads_gdf, out_path):
    g = roads_gdf.to_crs(epsg=4326).copy()
    vmax = float(np.nanpercentile(g['delay_seconds'], 95))
    cmap = plt.get_cmap('RdYlGn_r')

    def color_fn(d):
        x = min(max(d / (vmax + 1e-9), 0), 1)
        return mcolors.to_hex(cmap(x))

    g['__color__'] = g['delay_seconds'].apply(color_fn)
    minx, miny, maxx, maxy = g.total_bounds

    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2],
                   zoom_start=11, tiles='cartodbpositron')
    folium.GeoJson(
        g,
        style_function=lambda feat: {'color': feat['properties']['__color__'],
                                      'weight': 3, 'opacity': 0.85},
        tooltip=folium.GeoJsonTooltip(
            fields=['road_type', 'LOS', 'vc_ratio', 'delay_seconds',
                    't0_seconds', 't_congested_s', 'v_congested_kmh'],
            aliases=['Road type:', 'LOS:', 'V/C:', 'Delay (s):',
                     'Free-flow time (s):', 'Congested time (s):', 'Congested speed (km/h):'],
            localize=True, sticky=False, max_width=300,
        ),
    ).add_to(m)

    n_bins = 6
    bins = np.linspace(0, vmax, n_bins + 1)
    items = [(mcolors.to_hex(cmap(i / n_bins)),
              f'{bins[i]:.1f}s - {bins[i + 1]:.1f}s') for i in range(n_bins)]
    rows = {'title': f'Peak-hour delay (95th pct = {vmax:.1f}s)', 'items': items}
    m.get_root().add_child(make_legend(rows))
    m.save(out_path)
    print(f"  Map -> {out_path}")


def main():
    roads, cbd_x, cbd_y = prepare_roads()
    roads['pred_count_15min'] = train_and_predict_volume(roads, cbd_x, cbd_y)
    roads = attach_capacity(roads)
    roads = apply_bpr(roads)

    print(f"\nBPR results ({len(roads):,} segments)")
    print(f"  Mean V/C     : {roads.vc_ratio.mean():.3f}")
    print(f"  Median V/C   : {roads.vc_ratio.median():.3f}")
    print(f"  95th pct V/C : {roads.vc_ratio.quantile(0.95):.3f}")
    print(f"  Max V/C      : {roads.vc_ratio.max():.3f}")
    print(f"  Mean delay   : {roads.delay_seconds.mean():.2f}s")
    print(f"  Total delay  : {roads.delay_seconds.sum() / 3600:.0f}h")
    print()
    print('  LOS distribution:')
    los_dist = roads.LOS.value_counts().reindex(list('ABCDEF'), fill_value=0)
    los_df = los_dist.to_frame(name='segments')
    los_df['pct'] = (los_df.segments / los_df.segments.sum() * 100).round(1)
    print(los_df.to_string())

    # save per-segment results
    out_csv = 'data/bpr_segment_results_6yr.csv'
    keep = ['road_type', 'lanes', 'length', 'ff_speed_kmh',
            'pred_count_15min', 'volume_vph', 'capacity_vph', 'vc_ratio',
            't0_seconds', 't_congested_s', 'delay_seconds',
            'v_congested_kmh', 'LOS']
    roads[keep].reset_index().to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv}")

    print('  Rendering maps ...')
    map_los(roads, 'output/bpr_vc_ratio_6yr.html')
    map_travel_time(roads, 'output/bpr_travel_time_6yr.html')
    print("Done.")


if __name__ == '__main__':
    main()
