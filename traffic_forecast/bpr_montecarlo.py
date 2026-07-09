import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from branca.element import Template, MacroElement
import osmnx as ox

from utility import graph_from_place

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

EPSG_PROJ = 32610
BPR_ALPHA = 0.15
BPR_BETA = 4.0
VOLUME_PEAK_HOUR_FACTOR = 4
N_SIMULATIONS = 1000
RANDOM_STATE = 42

# growth scenarios: (name, annual_rate, years)
SCENARIOS = [
    ('baseline_0y', 0.000, 0),
    ('low_5y', 0.010, 5),
    ('mid_5y', 0.015, 5),
    ('high_5y', 0.030, 5),
]
DEFAULT_LABEL = 'mid_5y'

LOS_THRESHOLDS = [
    (0.60, 'A'), (0.70, 'B'), (0.80, 'C'),
    (0.90, 'D'), (1.00, 'E'), (np.inf, 'F'),
]
LOS_COLORS = {
    'A': '#1a9850', 'B': '#66bd63', 'C': '#a6d96a',
    'D': '#fee08b', 'E': '#f46d43', 'F': '#a50026',
}


def los_for_vc(vc):
    for thresh, letter in LOS_THRESHOLDS:
        if vc <= thresh:
            return letter
    return 'F'


def make_legend(rows):
    html = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
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


def load_deterministic():
    print('Loading deterministic predictions ...')
    preds = pd.read_csv('data/bpr_stratified_v3_results_6yr.csv')
    if not {'u', 'v', 'key'} <= set(preds.columns):
        raise RuntimeError("CSV missing (u,v,key) index columns")
    preds = preds.set_index(['u', 'v', 'key'])

    print('Loading road network ...')
    G = graph_from_place()
    roads = ox.graph_to_gdfs(G, nodes=False, edges=True).to_crs(epsg=EPSG_PROJ)

    keep = ['pred_count_15min', 'capacity_vph', 'capacity_vph_lanes_only',
            'length', 'ff_speed_kmh', 't0_seconds', 'tier', 'lanes']
    roads = roads.join(preds[keep], how='left', rsuffix='_pred')

    n_missing = roads['pred_count_15min'].isna().sum()
    if n_missing:
        print(f'  {n_missing} edges have no prediction, dropping them')
        roads = roads.dropna(subset=['pred_count_15min']).copy()
    return roads


def simulate_scenario(roads, growth, horizon, n_sim, seed):
    rng = np.random.default_rng(seed)

    growth_factor = (1.0 + growth) ** horizon
    mu_15min = roads['pred_count_15min'].values * growth_factor

    n_seg = len(mu_15min)
    capacity = roads['capacity_vph'].values
    L = roads['length'].values
    v_ff_ms = roads['ff_speed_kmh'].values / 3.6
    t0 = L / np.where(v_ff_ms > 0, v_ff_ms, 1e-9)

    # output arrays
    mean_vc = np.empty(n_seg)
    p50_vc = np.empty(n_seg)
    p95_vc = np.empty(n_seg)
    p99_vc = np.empty(n_seg)
    mean_dly = np.empty(n_seg)
    p95_dly = np.empty(n_seg)
    mean_t = np.empty(n_seg)
    p95_t = np.empty(n_seg)
    prob_F = np.empty(n_seg)
    prob_E = np.empty(n_seg)

    # process in chunks to manage memory
    CHUNK = 4096
    for start in range(0, n_seg, CHUNK):
        end = min(start + CHUNK, n_seg)
        mu = mu_15min[start:end][:, None]
        cap = capacity[start:end][:, None]
        t0c = t0[start:end][:, None]

        draws = rng.poisson(lam=np.maximum(mu, 0), size=(end - start, n_sim))
        V = draws.astype(float) * VOLUME_PEAK_HOUR_FACTOR
        vc = V / np.where(cap > 0, cap, 1e-9)
        t = t0c * (1.0 + BPR_ALPHA * np.power(vc, BPR_BETA))
        delay = t - t0c

        mean_vc[start:end] = vc.mean(axis=1)
        p50_vc[start:end] = np.median(vc, axis=1)
        p95_vc[start:end] = np.percentile(vc, 95, axis=1)
        p99_vc[start:end] = np.percentile(vc, 99, axis=1)
        mean_dly[start:end] = delay.mean(axis=1)
        p95_dly[start:end] = np.percentile(delay, 95, axis=1)
        mean_t[start:end] = t.mean(axis=1)
        p95_t[start:end] = np.percentile(t, 95, axis=1)
        prob_F[start:end] = (vc > 1.00).mean(axis=1)
        prob_E[start:end] = (vc > 0.90).mean(axis=1)

    result = pd.DataFrame({
        'mu_future_15min': mu_15min,
        'mean_vc': mean_vc, 'p50_vc': p50_vc,
        'p95_vc': p95_vc, 'p99_vc': p99_vc,
        'mean_delay_s': mean_dly, 'p95_delay_s': p95_dly,
        'mean_t_s': mean_t, 'p95_t_s': p95_t,
        'prob_LOS_F': prob_F, 'prob_LOS_E_or_worse': prob_E,
    }, index=roads.index)
    return result


def map_vc(roads_gdf, vc_col, title, out_path, los_legend=True):
    g = roads_gdf.to_crs(epsg=4326).copy()

    if los_legend:
        g['__color__'] = g[vc_col].apply(los_for_vc).map(LOS_COLORS)
        legend_rows = {
            'title': title,
            'items': [(LOS_COLORS[k], f'{k} - {d}') for k, d in [
                ('A', 'Free flow'), ('B', 'Reasonably free'),
                ('C', 'Stable'), ('D', 'Approaching unstable'),
                ('E', 'At capacity'), ('F', 'Breakdown')]]}
    else:
        vmax = float(np.nanpercentile(g[vc_col], 95))
        cmap = plt.get_cmap('RdYlGn_r')
        g['__color__'] = g[vc_col].apply(
            lambda d: mcolors.to_hex(cmap(min(max(d / (vmax + 1e-9), 0), 1))))
        n_bins = 6
        bins = np.linspace(0, vmax, n_bins + 1)
        items = [(mcolors.to_hex(cmap(i / n_bins)),
                  f'{bins[i]:.1f} - {bins[i + 1]:.1f}') for i in range(n_bins)]
        legend_rows = {'title': title, 'items': items}

    minx, miny, maxx, maxy = g.total_bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2],
                   zoom_start=11, tiles='cartodbpositron')
    folium.GeoJson(
        g,
        style_function=lambda feat: {'color': feat['properties']['__color__'],
                                      'weight': 3, 'opacity': 0.85},
        tooltip=folium.GeoJsonTooltip(
            fields=['road_type', 'tier', 'lanes', 'mu_future_15min',
                    'mean_vc', 'p95_vc', 'p99_vc', 'prob_LOS_F',
                    'mean_delay_s', 'p95_delay_s'],
            aliases=['Road type:', 'Tier:', 'Lanes:', 'Future 15-min:',
                     'Mean V/C:', 'P95 V/C:', 'P99 V/C:', 'P(LOS F):',
                     'Mean delay (s):', 'P95 delay (s):'],
            localize=True, sticky=False, max_width=320),
    ).add_to(m)
    m.get_root().add_child(make_legend(legend_rows))
    m.save(out_path)
    print(f"  Map -> {out_path}")


def main():
    roads = load_deterministic()

    summaries = {}
    print(f'\nMonte Carlo simulation (N={N_SIMULATIONS} draws per scenario)')
    for label, growth, horizon in SCENARIOS:
        print(f'  Scenario "{label}": growth={growth * 100:+.1f}%/yr, horizon={horizon}yr')
        df = simulate_scenario(roads, growth, horizon, N_SIMULATIONS, RANDOM_STATE)

        df['road_type'] = roads['highway'].apply(
            lambda v: v[0] if isinstance(v, list) else v).values
        df['tier'] = roads['tier'].values
        df['lanes'] = roads['lanes'].values
        df['capacity_vph'] = roads['capacity_vph'].values
        df['ff_speed_kmh'] = roads['ff_speed_kmh'].values

        out_csv = f'data/bpr_mc_{label}_results.csv'
        df.reset_index().to_csv(out_csv, index=False)
        print(f'    -> {out_csv}')
        summaries[label] = df

    # print summary comparison
    print('\nCongestion shift across scenarios:')
    rows = []
    for label, growth, horizon in SCENARIOS:
        d = summaries[label]
        rows.append({
            'scenario': label,
            'growth': f'+{growth * 100:.1f}%',
            'horizon': horizon,
            'mean_VC': round(d.mean_vc.mean(), 3),
            'median_VC': round(d.mean_vc.median(), 3),
            'p95_VC_at_p95': round(d.p95_vc.quantile(0.95), 3),
            'pct_meanVC_over_1': round(100 * (d.mean_vc > 1.0).mean(), 2),
            'pct_p95VC_over_1': round(100 * (d.p95_vc > 1.0).mean(), 2),
            'total_delay_h': round(d.mean_delay_s.sum() / 3600, 2),
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv('data/bpr_mc_summary.csv', index=False)

    # render maps for the default scenario
    print(f'\nRendering maps for "{DEFAULT_LABEL}" ...')
    d = summaries[DEFAULT_LABEL]
    map_roads = roads.copy()
    for col in d.columns:
        if col not in map_roads.columns:
            map_roads[col] = d[col]

    map_vc(map_roads, 'mean_vc',
           f'Mean V/C - {DEFAULT_LABEL} (MC, N={N_SIMULATIONS})',
           f'output/bpr_mc_{DEFAULT_LABEL}_vc_mean.html', los_legend=True)
    map_vc(map_roads, 'p95_vc',
           f'P95 V/C - {DEFAULT_LABEL} (worst case)',
           f'output/bpr_mc_{DEFAULT_LABEL}_vc_p95.html', los_legend=True)
    map_vc(map_roads, 'mean_delay_s',
           f'Mean delay (s) - {DEFAULT_LABEL}',
           f'output/bpr_mc_{DEFAULT_LABEL}_delay_mean.html', los_legend=False)

    print('Done.')


if __name__ == '__main__':
    main()
