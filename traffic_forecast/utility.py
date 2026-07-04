import pandas as pd
import folium
import pickle
import os
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from branca.element import Template, MacroElement


def load_fix_traffic_data(csv_file):
    df = pd.read_csv(csv_file, parse_dates=['Timestamp'])
    df['Count'] = df['Total']
    df['Time'] = df['Timestamp'].dt.time
    df['DateTime'] = df['Timestamp'].dt.strftime('%m/%d/%Y %I:%M:%S %p')
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%Y %I:%M:%S %p')
    return df[['Time', 'DateTime', 'Count', 'Latitude', 'Longitude']]


def load_midpoint_traffic_data(csv_file):
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['Count Date'], format='%m/%d/%Y %I:%M:%S %p')
    df['Time'] = df['timestamp'].dt.time
    df = df.rename(columns={'Count 15 Minutes': 'Count'})
    df['DateTime'] = pd.to_datetime(df['Count Date'], format='%m/%d/%Y %I:%M:%S %p')
    return df[['Time', 'DateTime', 'Count', 'Latitude', 'Longitude']]


def load_data_with_datetime(start_date, end_date):
    fixed = load_fix_traffic_data('data/filtered_fix_data.csv')
    combined_df = pd.concat([fixed], ignore_index=True)
    start = pd.to_datetime(start_date, format='%Y-%m-%d %H:%M:%S')
    end = pd.to_datetime(end_date, format='%Y-%m-%d %H:%M:%S')
    return combined_df[(combined_df['DateTime'] >= start) & (combined_df['DateTime'] <= end)]


def load_data_with_time(start_time, end_time):
    fixed = load_fix_traffic_data('data/filtered_fix_data.csv')
    combined_df = pd.concat([fixed], ignore_index=True)
    combined_df['Time'] = pd.to_datetime(combined_df['Time'], format='%H:%M:%S').dt.time
    return combined_df[(combined_df['Time'] >= start_time) & (combined_df['Time'] <= end_time)]


def style_function_factory(max_count):
    def style_fn(feature):
        count = feature['properties']['pred_count']
        norm = max(min(count / max_count, 1.0), 0.0)
        cmap = plt.get_cmap('RdYlGn_r')
        color = mcolors.to_hex(cmap(norm))
        return {'color': color, 'weight': 3, 'fillOpacity': 0.7}
    return style_fn


def road_type_weights():
    # volume ranges based on Winnipeg road classification
    ranges = {
        'Major Arterials': 40000 / 40000,
        'Minor Arterials': (20000 + 40000) / (2 * 40000),
        'Major Collectors': (10000 + 20000) / (2 * 40000),
        'Minor Collectors': (5000 + 10000) / (2 * 40000),
        'Local/Residential': (0 + 5000) / (2 * 40000)
    }
    return {
        'motorway': ranges['Major Arterials'],
        'motorway_link': ranges['Major Arterials'],
        'trunk': ranges['Major Arterials'],
        'trunk_link': ranges['Major Arterials'],
        'primary': ranges['Major Arterials'],
        'primary_link': ranges['Major Arterials'],
        'secondary': ranges['Minor Arterials'],
        'secondary_link': ranges['Minor Arterials'],
        'tertiary': ranges['Major Collectors'],
        'tertiary_link': ranges['Major Collectors'],
        'unclassified': ranges['Minor Collectors'],
        'residential': ranges['Local/Residential'],
        'busway': ranges['Minor Arterials'],
    }


def to_map(roads, map_name):
    roads_gdf = roads.to_crs(epsg=4326)
    minx, miny, maxx, maxy = roads_gdf.total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2

    max_count = roads_gdf['pred_count'].max()
    min_count = roads_gdf['pred_count'].min()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    folium.GeoJson(
        roads_gdf,
        style_function=style_function_factory(max_count),
        tooltip=folium.GeoJsonTooltip(fields=['pred_count'])
    ).add_to(m)

    # build legend
    n_bins = 6
    colormap = plt.get_cmap('RdYlGn_r')
    bin_edges = np.linspace(min_count, max_count, n_bins + 1)
    ranges = [(int(bin_edges[i]), int(bin_edges[i + 1])) for i in range(n_bins)]
    step_colors = [mcolors.to_hex(colormap(i / n_bins)) for i in range(n_bins)]

    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 220px;
        z-index:9999;
        background-color: white;
        padding: 10px;
        border: 2px solid grey;
        font-size: 14px;
        line-height: 18px;">
        <b>Predicted Vehicle Count</b><br>
        {''.join([
        f'<div style="display: flex; align-items: center; margin-top: 4px;">'
        f'<div style="width: 20px; height: 12px; background:{color}; margin-right: 8px;"></div>'
        f'{r[0]} - {r[1]}'
        f'</div>' for color, r in zip(step_colors, ranges)
    ])}
    </div>
    {{% endmacro %}}
    """

    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    m.save("output/" + map_name + ".html")
    print("Map saved as " + map_name + ".html")


def load_filtered_data(start_time=None, end_time=None, suffix='6yr'):
    perm = pd.read_csv(f'data/permanent_{suffix}.csv', parse_dates=['DateTime'])
    perm['Time'] = perm['DateTime'].dt.time

    mid = pd.read_csv(f'data/midblock_{suffix}.csv', parse_dates=['DateTime'])
    mid['Time'] = mid['DateTime'].dt.time

    combined = pd.concat(
        [perm[['DateTime', 'Time', 'Count', 'Latitude', 'Longitude']],
         mid[['DateTime', 'Time', 'Count', 'Latitude', 'Longitude']]],
        ignore_index=True
    )
    combined = combined.dropna(subset=['Count', 'Latitude', 'Longitude'])

    if start_time is not None and end_time is not None:
        combined = combined[(combined['Time'] >= start_time) & (combined['Time'] <= end_time)]

    return combined.reset_index(drop=True)


def graph_from_place():
    # load from cache if available
    if os.path.exists('data/winnipeg.pkl'):
        with open('data/winnipeg.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        G = ox.graph_from_place("Winnipeg, Canada", network_type='drive')
        with open('data/winnipeg.pkl', 'wb') as f:
            pickle.dump(G, f)
        return G
