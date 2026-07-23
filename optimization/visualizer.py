import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from config import PathScores


class WinnipegVisualizer:

    @staticmethod
    def _node_coords(graph, node):
        return graph.nodes[node]['y'], graph.nodes[node]['x']

    @staticmethod
    def _collect_used_stations(path_details, charging_manager):
        used = {}
        for ps in path_details:
            for sid in ps.used_station_ids:
                if sid not in used:
                    station = charging_manager.get_station(sid)
                    if station:
                        used[sid] = station
        return used

    @staticmethod
    def plot_pareto_front(scores, title="Pareto Front", start_node=None, end_node=None):
        fig = plt.figure(figsize=(15, 10))

        if start_node and end_node:
            header = f"Route Optimization: Time vs Fee\nWinnipeg: Node {start_node} -> Node {end_node}\n{title}"
        else:
            header = f"Route Optimization: Time vs Fee\n{title}"
        fig.suptitle(header, fontsize=13)

        ax = fig.add_subplot(111)
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=scores[:, 1],
                        cmap='viridis', s=100, alpha=0.8, edgecolors='black')
        plt.colorbar(sc, ax=ax, label='Charging Fee ($)')
        ax.set_xlabel('Total Time (minutes)', fontsize=12)
        ax.set_ylabel('Total Charging Fee ($)', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_3d_pareto_front(path_details, title="NSGA-II Optimisation",
                             start_node=None, end_node=None):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if start_node and end_node:
            header = f"Multi-Objective EV Routing: Node {start_node} -> Node {end_node}\n{title}"
        else:
            header = f"Multi-Objective EV Routing\n{title}"
        fig.suptitle(header, fontsize=13)

        x = np.array([ps.distance_km for ps in path_details])
        y = np.array([ps.total_time_minutes for ps in path_details])
        z = np.array([ps.total_fee for ps in path_details])
        c = np.array([ps.num_charging_stops for ps in path_details], dtype=float)

        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Total Time (min)')
        ax.set_zlabel('Charging Fee ($)')

        scatter = ax.scatter(x, y, z, c=c, cmap='hot', s=60, alpha=0.85)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Number of Charging Stops')
        import matplotlib.ticker as mticker
        cbar.locator = mticker.MaxNLocator(integer=True)
        cbar.update_ticks()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_parallel_coordinates(path_details,
                                  title="EV Route Objectives - Parallel Coordinates"):
        # estimate final SOC based on distance and charging stops
        def _approx_final_soc(ps):
            consumed = (ps.distance_km * 0.18) / 10.0
            charged = ps.num_charging_stops * (0.55 - 0.25)
            return round(max(0.0, min(1.0, 0.80 - consumed + charged)) * 100, 1)

        df = pd.DataFrame({
            'Total Time (min)': [ps.total_time_minutes for ps in path_details],
            'Charging Fee ($)': [ps.total_fee for ps in path_details],
            'Distance (km)': [ps.distance_km for ps in path_details],
            'Charging Stops': [float(ps.num_charging_stops) for ps in path_details],
            'Final SOC (%)': [_approx_final_soc(ps) for ps in path_details],
        })

        fig = px.parallel_coordinates(
            df,
            color='Charging Fee ($)',
            color_continuous_scale=px.colors.sequential.Plasma,
            labels={col: col for col in df.columns},
            title=title
        )
        return fig

    @staticmethod
    def plot_route_comparison(graph, solutions, path_details, charging_stations,
                              charging_manager=None, labels=None):
        fig = go.Figure()
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        used_station_ids = set()

        for idx, (path, ps) in enumerate(zip(solutions[:5], path_details[:5])):
            if not path:
                continue

            lats = []
            lons = []
            for node in path:
                if graph.nodes[node].get('is_virtual', False):
                    continue
                lat, lon = WinnipegVisualizer._node_coords(graph, node)
                lats.append(lat)
                lons.append(lon)

            if ps.used_station_ids:
                used_station_ids.update(ps.used_station_ids)

            # draw detour lines to charging stations if we have that info
            if charging_manager and getattr(ps, 'charging_trigger_nodes', []):
                for trigger_node, sid in ps.charging_trigger_nodes:
                    st = charging_manager.get_station(sid)
                    if not st:
                        continue
                    try:
                        t_lat = graph.nodes[trigger_node]['y']
                        t_lon = graph.nodes[trigger_node]['x']
                        fig.add_trace(go.Scattermapbox(
                            mode='lines+markers',
                            lat=[t_lat, st.latitude],
                            lon=[t_lon, st.longitude],
                            line={'width': 2, 'color': colors[idx % len(colors)]},
                            marker={'size': [6, 14], 'color': ['white', '#FFD700']},
                            opacity=0.85,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    except KeyError:
                        pass

            label = labels[idx] if labels else f"Route {idx + 1}"
            fig.add_trace(go.Scattermapbox(
                mode='lines',
                lon=lons,
                lat=lats,
                name=f"{label}<br>Dist: {ps.distance_km:.1f}km<br>"
                     f"Time: {ps.total_time_minutes:.1f}min<br>"
                     f"Fee: ${ps.total_fee:.2f}",
                line={'width': 3, 'color': colors[idx % len(colors)]}
            ))

            # add start and end markers
            if lats:
                fig.add_trace(go.Scattermapbox(
                    mode='markers',
                    lon=[lons[0]], lat=[lats[0]],
                    marker={'size': 15, 'color': 'green', 'symbol': 'circle'},
                    text=[f"{label} - START"],
                    name=f"{label} Start",
                    showlegend=False
                ))
                fig.add_trace(go.Scattermapbox(
                    mode='markers',
                    lon=[lons[-1]], lat=[lats[-1]],
                    marker={'size': 22, 'color': '#FF2020',
                            'symbol': 'circle', 'opacity': 1.0},
                    text=[f"<b>{label} - END</b>"],
                    name=f"{label} End",
                    showlegend=True
                ))

        # used stations
        used_lats, used_lons, used_texts = [], [], []
        for sid in used_station_ids:
            station = charging_manager.get_station(sid) if charging_manager else None
            if station:
                used_lats.append(station.latitude)
                used_lons.append(station.longitude)
                used_texts.append(
                    f"USED CHARGING STATION<br>{station.title}<br>"
                    f"ID: {station.station_id}<br>"
                    f"Power: {station.charging_power} kW<br>"
                    f"Price: ${station.electricity_price}/kWh<br>"
                    f"Ports: {station.num_ports}"
                )

        if used_lats:
            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lon=used_lons,
                lat=used_lats,
                marker={'size': 20, 'color': '#FFD700', 'symbol': 'circle'},
                text=used_texts,
                name=f'Used Charging Stations ({len(used_lats)})',
                showlegend=True
            ))

        # other available stations (not used)
        other_lats, other_lons, other_texts = [], [], []
        for station in list(charging_stations.values())[:120]:
            if station.station_id in used_station_ids:
                continue
            other_lats.append(station.latitude)
            other_lons.append(station.longitude)
            other_texts.append(
                f"Available Station<br>{station.title}<br>"
                f"Power: {station.charging_power} kW<br>"
                f"Price: ${station.electricity_price}/kWh"
            )

        if other_lats:
            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lon=other_lons,
                lat=other_lats,
                marker={'size': 8, 'color': 'lightblue', 'symbol': 'circle', 'opacity': 0.6},
                text=other_texts,
                name=f'Available Stations ({len(other_lats)})',
                showlegend=True
            ))

        # figure out center of the map
        all_lats, all_lons = [], []
        for path in solutions[:5]:
            for node in path:
                if not graph.nodes[node].get('is_virtual', False):
                    lat, lon = WinnipegVisualizer._node_coords(graph, node)
                    all_lats.append(lat)
                    all_lons.append(lon)

        center_lat = sum(all_lats) / len(all_lats) if all_lats else 49.895
        center_lon = sum(all_lons) / len(all_lons) if all_lons else -97.138

        fig.update_layout(
            mapbox=dict(style='open-street-map',
                        center=dict(lat=center_lat, lon=center_lon), zoom=11),
            height=800,
            title={'text': "EV Route Comparison with Charging Stations", 'x': 0.5,
                   'xanchor': 'center', 'font': {'size': 18}},
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                        bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="black", borderwidth=1)
        )
        return fig

    @staticmethod
    def create_route_summary(graph, path, path_scores, charging_manager,
                             route_name="Route"):
        summary = []
        summary.append("=" * 80)
        summary.append(f"{route_name.upper()} SUMMARY")
        summary.append("=" * 80)
        summary.append(f"\nOVERALL:")
        summary.append(f"  Total Distance: {path_scores.distance_km:.2f} km")
        summary.append(f"  Total Time: {path_scores.total_time_minutes:.1f} minutes ({path_scores.total_time_minutes / 60:.2f} hours)")
        summary.append(f"  Charging Fee: ${path_scores.total_fee:.2f}")
        summary.append(f"  Charging Stops: {path_scores.num_charging_stops}")
        summary.append(f"  Battery Feasible: {'Yes' if path_scores.battery_feasible else 'No'}")

        if path_scores.used_station_ids:
            summary.append(f"\nCHARGING STOPS ({len(path_scores.used_station_ids)}):")
            for i, sid in enumerate(path_scores.used_station_ids, 1):
                station = charging_manager.get_station(sid)
                if station:
                    summary.append(f"\n  Stop {i}:")
                    summary.append(f"    Station: {station.title}")
                    summary.append(f"    Station ID: {station.station_id}")
                    summary.append(f"    Charging Power: {station.charging_power} kW")
                    summary.append(f"    Price: ${station.electricity_price}/kWh")
                    summary.append(f"    Ports: {station.num_ports}")
        else:
            summary.append(f"\nCHARGING STOPS: None (sufficient battery)")

        summary.append(f"\nPATH:")
        summary.append(f"  Number of nodes: {len(path)}")
        summary.append(f"  Start: {path[0]}")
        summary.append(f"  End: {path[-1]}")

        return "\n".join(summary)

    @staticmethod
    def plot_network(graph, charging_stations=None,
                     title="Winnipeg Road Network & EV Stations"):
        fig = go.Figure()
        edge_lats = []
        edge_lons = []

        for u, v in graph.edges():
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            if not u_data.get('is_virtual', False) and not v_data.get('is_virtual', False):
                if u_data.get('y') and v_data.get('y'):
                    edge_lats.extend([u_data['y'], v_data['y'], None])
                    edge_lons.extend([u_data['x'], v_data['x'], None])

        fig.add_trace(go.Scattermapbox(
            mode='lines', lon=edge_lons, lat=edge_lats,
            line={'width': 1, 'color': 'gray'}, name='Roads', hoverinfo='skip'
        ))

        node_lats = []
        node_lons = []
        for node in graph.nodes():
            data = graph.nodes[node]
            if not data.get('is_virtual', False) and data.get('y'):
                node_lats.append(data['y'])
                node_lons.append(data['x'])

        if charging_stations:
            station_lats = [s.latitude for s in charging_stations.values()]
            station_lons = [s.longitude for s in charging_stations.values()]
            station_texts = [f"{s.title}<br>ID: {s.station_id}<br>Power: {s.charging_power} kW"
                             for s in charging_stations.values()]

            if station_lats:
                fig.add_trace(go.Scattermapbox(
                    mode='markers', lon=station_lons, lat=station_lats,
                    marker={'size': 8, 'color': 'green', 'symbol': 'circle'},
                    text=station_texts, name=f'Stations ({len(station_lats)})'
                ))

        center_lat = np.mean(node_lats) if node_lats else 49.895
        center_lon = np.mean(node_lons) if node_lons else -97.138

        fig.update_layout(
            mapbox=dict(style='open-street-map',
                        center=dict(lat=center_lat, lon=center_lon), zoom=10),
            height=800,
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            showlegend=True
        )
        return fig

    @staticmethod
    def plot_route_networkx(graph, routes, start_node, end_node,
                            labels=None, path_details=None,
                            charging_manager=None, output_path=None):
        route_colours = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

        G2 = nx.DiGraph()
        route_node_sets = []
        route_edge_sets = []
        for route in routes:
            if not route:
                route_node_sets.append([])
                route_edge_sets.append([])
                continue
            real_nodes = [n for n in route if not graph.nodes[n].get('is_virtual', False)]
            edges = [(real_nodes[i], real_nodes[i + 1])
                     for i in range(len(real_nodes) - 1)
                     if graph.has_edge(real_nodes[i], real_nodes[i + 1])]
            G2.add_nodes_from(real_nodes)
            G2.add_edges_from(edges)
            route_node_sets.append(real_nodes)
            route_edge_sets.append(edges)

        pos = {node: (graph.nodes[node]['x'], graph.nodes[node]['y'])
               for node in G2.nodes
               if graph.nodes[node].get('x') is not None}

        fig, ax = plt.subplots(figsize=(20, 15), facecolor='white')
        ax.set_facecolor('white')

        legend_items = []
        for i, (node_list, edge_list) in enumerate(zip(route_node_sets, route_edge_sets)):
            if not edge_list:
                continue
            colour = route_colours[i % len(route_colours)]
            for u, v in edge_list:
                if u in pos and v in pos:
                    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                            color=colour, linewidth=2.5, alpha=0.85,
                            solid_capstyle='round', zorder=2)
            mid_nodes = [n for n in node_list if n != start_node and n != end_node]
            if mid_nodes:
                xs = [pos[n][0] for n in mid_nodes if n in pos]
                ys = [pos[n][1] for n in mid_nodes if n in pos]
                ax.scatter(xs, ys, s=8, color=colour, alpha=0.5, zorder=3)

            label = labels[i] if labels else f"Route {i + 1}"
            ps = path_details[i] if path_details else None
            detail = f"  {ps.total_time_minutes:.1f} min  ${ps.total_fee:.2f}" if ps else ""
            legend_items.append(plt.Line2D([0], [0], color=colour, linewidth=3,
                                           label=f"{label}{detail}"))

            # draw charging station markers for this route
            if path_details and charging_manager:
                ps = path_details[i] if path_details else None
                if ps:
                    for sid in ps.used_station_ids:
                        st = charging_manager.get_station(sid)
                        if st:
                            ax.scatter([st.longitude], [st.latitude],
                                       marker='*', s=500, c=colour,
                                       edgecolors='black', linewidths=1,
                                       zorder=6)
                            ax.annotate(
                                st.title[:22],
                                (st.longitude, st.latitude),
                                fontsize=7, color='black', ha='center', va='bottom',
                                xytext=(0, 8), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                          alpha=0.7, ec='gray')
                            )

        # start node marker
        if start_node in pos:
            ax.scatter([pos[start_node][0]], [pos[start_node][1]],
                       s=350, color='#00C853', edgecolors='black',
                       linewidths=1.5, zorder=7, marker='o')
            ax.annotate('START', pos[start_node], fontsize=9, fontweight='bold',
                        color='#00C853', ha='center', va='bottom',
                        xytext=(0, 10), textcoords='offset points')

        # end node marker
        if end_node in pos:
            ax.scatter([pos[end_node][0]], [pos[end_node][1]],
                       s=350, color='#FF2020', edgecolors='black',
                       linewidths=1.5, zorder=7, marker='o')
            ax.annotate('END', pos[end_node], fontsize=9, fontweight='bold',
                        color='#FF2020', ha='center', va='bottom',
                        xytext=(0, 10), textcoords='offset points')

        legend_items += [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00C853',
                       markeredgecolor='black', markersize=12, label='Start'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF2020',
                       markeredgecolor='black', markersize=12, label='End'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                       markeredgecolor='black', markersize=14,
                       label='Charging Stop (colour = route)'),
        ]

        ax.legend(handles=legend_items, loc='upper left', fontsize=10,
                  facecolor='white', edgecolor='gray', framealpha=0.9)

        ax.set_title(f'Route Comparison  -  {start_node} -> {end_node}',
                     fontsize=15, color='black', pad=14)
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.tick_params(colors='black')
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight',
                        facecolor='white')

        return fig

    @staticmethod
    def _short_station_name(title, max_chars=28):
        words = str(title).split()
        if not words:
            return ''
        out = []
        for w in words:
            if out and len(' '.join(out + [w])) > max_chars:
                break
            out.append(w)
        trailing = {'of', 'on', 'at', 'the', 'and', 'de', '-', '-', 'in', 'for'}
        while len(out) > 1 and out[-1].lower().strip('--,') in trailing:
            out.pop()
        label = ' '.join(out).rstrip(' ,--')
        return label + '...' if len(out) < len(words) else label

    @staticmethod
    def plot_route_map_thesis(graph, routes, start_node, end_node,
                              labels=None, station_ids=None,
                              charging_manager=None, title=None,
                              output_path=None, figsize=(7.2, 4.4),
                              base_fontsize=10.0, show_station_names=True,
                              road_context=True, exclude_station_ids=None):
        route_colours = ['#D32F2F', '#1E88E5', '#2E7D32', '#EF6C00', '#6A1B9A']
        route_widths = [4.2, 3.0, 1.9, 1.2, 0.9]
        route_styles = ['-', '-', '-', (0, (4, 1.6)), (0, (1.5, 1.5))]
        route_alphas = [0.75, 0.9, 0.95, 1.0, 1.0]

        fs = base_fontsize
        excluded = {str(s) for s in (exclude_station_ids or [])}

        node_xy = {}
        route_edge_sets = []
        for route in routes:
            if not route:
                route_edge_sets.append([])
                continue
            real_nodes = [n for n in route
                          if not graph.nodes[n].get('is_virtual', False)]
            edges = [(real_nodes[i], real_nodes[i + 1])
                     for i in range(len(real_nodes) - 1)
                     if graph.has_edge(real_nodes[i], real_nodes[i + 1])]
            for n in real_nodes:
                if n not in node_xy and graph.nodes[n].get('x') is not None:
                    node_xy[n] = (graph.nodes[n]['x'], graph.nodes[n]['y'])
            route_edge_sets.append(edges)

        if not node_xy:
            raise ValueError('no positioned nodes in the supplied routes')

        # compute bounding box
        xs_all = [p[0] for p in node_xy.values()]
        ys_all = [p[1] for p in node_xy.values()]
        if charging_manager and station_ids:
            for sids in station_ids:
                for sid in sids or []:
                    if str(sid) in excluded:
                        continue
                    st = charging_manager.get_station(sid)
                    if st:
                        xs_all.append(st.longitude)
                        ys_all.append(st.latitude)
        x0, x1 = min(xs_all), max(xs_all)
        y0, y1 = min(ys_all), max(ys_all)
        padx = max((x1 - x0) * 0.06, 1e-3)
        pady = max((y1 - y0) * 0.10, 1e-3)
        xlim = (x0 - padx, x1 + padx)
        ylim = (y0 - pady, y1 + pady)

        # aspect ratio correction for lat/lon
        _KM_PER_DEG_LAT = 110.57
        _mid = np.radians(0.5 * (ylim[0] + ylim[1]))
        _km_lon = 111.32 * max(np.cos(_mid), 1e-6)
        w_km = (xlim[1] - xlim[0]) * _km_lon
        h_km = (ylim[1] - ylim[0]) * _KM_PER_DEG_LAT
        min_ratio, max_ratio = 0.45, 1.25
        if h_km < min_ratio * w_km:
            want_deg = (min_ratio * w_km) / _KM_PER_DEG_LAT
            cy = 0.5 * (ylim[0] + ylim[1])
            ylim = (cy - want_deg / 2, cy + want_deg / 2)
        elif h_km > max_ratio * w_km:
            want_deg = (h_km / max_ratio) / _km_lon
            cx = 0.5 * (xlim[0] + xlim[1])
            xlim = (cx - want_deg / 2, cx + want_deg / 2)

        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('#FBFBFB')

        # draw background road network for context
        if road_context:
            segs = []
            for u, v in graph.edges():
                du, dv = graph.nodes[u], graph.nodes[v]
                if du.get('is_virtual') or dv.get('is_virtual'):
                    continue
                ux, uy = du.get('x'), du.get('y')
                vx, vy = dv.get('x'), dv.get('y')
                if ux is None or vx is None:
                    continue
                if (max(ux, vx) < xlim[0] or min(ux, vx) > xlim[1] or
                        max(uy, vy) < ylim[0] or min(uy, vy) > ylim[1]):
                    continue
                segs.append([(ux, uy), (vx, vy)])
            if segs:
                from matplotlib.collections import LineCollection
                ax.add_collection(LineCollection(
                    segs, colors='#C9CDD2', linewidths=0.25, alpha=0.55,
                    zorder=1))

        legend_items = []
        for i, edge_list in enumerate(route_edge_sets):
            if not edge_list:
                continue
            colour = route_colours[i % len(route_colours)]
            width = route_widths[i % len(route_widths)]
            style = route_styles[i % len(route_styles)]
            segs = [[node_xy[u], node_xy[v]] for u, v in edge_list
                    if u in node_xy and v in node_xy]
            if segs:
                from matplotlib.collections import LineCollection
                ax.add_collection(LineCollection(
                    segs, colors=colour, linewidths=width,
                    alpha=route_alphas[i % len(route_alphas)],
                    linestyles=style, capstyle='round', zorder=3 + i))
            legend_items.append(plt.Line2D(
                [0], [0], color=colour, linewidth=max(width, 1.8),
                linestyle=style,
                label=labels[i] if labels else f'Route {i + 1}'))

        # plot charging stations used by each route
        if charging_manager and station_ids:
            drawn = {}
            for i, sids in enumerate(station_ids):
                colour = route_colours[i % len(route_colours)]
                for sid in sids or []:
                    if str(sid) in excluded:
                        continue
                    st = charging_manager.get_station(sid)
                    if not st:
                        continue
                    ax.scatter([st.longitude], [st.latitude], marker='*',
                               s=260, c=colour, edgecolors='black',
                               linewidths=0.7, zorder=9)
                    if not show_station_names or sid in drawn:
                        continue
                    drawn[sid] = True
                    name = WinnipegVisualizer._short_station_name(st.title)
                    ax.annotate(
                        name, (st.longitude, st.latitude),
                        fontsize=fs - 2.5, color='#212121',
                        ha='center', va='bottom',
                        xytext=(0, 9), textcoords='offset points', zorder=10,
                        bbox=dict(boxstyle='round,pad=0.22', fc='white',
                                  alpha=0.85, ec='#B0B0B0', linewidth=0.5))

        for node, colour, text, dy in ((start_node, '#00A63E', 'START', 11),
                                       (end_node, '#D50000', 'END', 11)):
            if node not in node_xy:
                continue
            ax.scatter([node_xy[node][0]], [node_xy[node][1]], s=150,
                       color=colour, edgecolors='black', linewidths=1.0,
                       zorder=11, marker='o')
            ax.annotate(text, node_xy[node], fontsize=fs - 1.5,
                        fontweight='bold', color=colour, ha='center',
                        va='bottom', xytext=(0, dy), textcoords='offset points',
                        zorder=11)

        legend_items += [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00A63E',
                       markeredgecolor='black', markersize=8, label='Start'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D50000',
                       markeredgecolor='black', markersize=8, label='End'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#757575',
                       markeredgecolor='black', markersize=13,
                       label='Charging stop (colour = plan)'),
        ]

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        mid_lat = np.radians(0.5 * (ylim[0] + ylim[1]))
        ax.set_aspect(1.0 / max(np.cos(mid_lat), 1e-6), adjustable='box')

        ax.grid(True, color='#E4E7EA', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel('Longitude (deg)', fontsize=fs)
        ax.set_ylabel('Latitude (deg)', fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs - 1.5, colors='#333333',
                       length=3, width=0.6)
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=4)
        for spine in ax.spines.values():
            spine.set_edgecolor('#9E9E9E')
            spine.set_linewidth(0.7)

        # draw a scale bar in the bottom right
        km_per_deg_lon = 111.32 * np.cos(mid_lat)
        span_km = (xlim[1] - xlim[0]) * km_per_deg_lon
        nice = [1, 2, 5, 10, 20, 50]
        bar_km = min(nice, key=lambda k: abs(k - span_km * 0.22))
        bar_deg = bar_km / km_per_deg_lon
        bx1 = xlim[1] - 0.04 * (xlim[1] - xlim[0])
        bx0 = bx1 - bar_deg
        by = ylim[0] + 0.07 * (ylim[1] - ylim[0])
        ax.plot([bx0, bx1], [by, by], color='#212121', linewidth=1.6,
                solid_capstyle='butt', zorder=12)
        for bx in (bx0, bx1):
            ax.plot([bx, bx], [by - 0.012 * (ylim[1] - ylim[0]),
                               by + 0.012 * (ylim[1] - ylim[0])],
                    color='#212121', linewidth=1.2, zorder=12)
        ax.annotate(f'{bar_km} km', (0.5 * (bx0 + bx1), by), fontsize=fs - 2.5,
                    color='#212121', ha='center', va='bottom',
                    xytext=(0, 3), textcoords='offset points', zorder=12)

        if title:
            ax.set_title(title, fontsize=fs + 1, color='black', pad=8)

        ncol = 4 if len(legend_items) > 5 else 3
        ax.legend(handles=legend_items, loc='upper center',
                  bbox_to_anchor=(0.5, -0.20), ncol=ncol, fontsize=fs - 1.5,
                  frameon=False, handlelength=2.4, columnspacing=1.4,
                  handletextpad=0.6)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                        facecolor='white')

        return fig

    @staticmethod
    def plot_soc_profile(graph, paths, path_details, data_loader, config,
                         charging_manager, labels=None, start_time_slot=0):
        colours = ['red', 'blue', 'green', 'orange', 'purple']
        fig, ax = plt.subplots(figsize=(14, 7))

        for idx, (path, ps) in enumerate(zip(paths, path_details)):
            if not path:
                continue
            soc = config.initial_soc
            cumulative_km = 0.0
            dist_points = [0.0]
            soc_points = [soc]
            charge_km = []
            charge_soc_before = []
            charge_soc_after = []
            visited = set()
            cur_time_slot = start_time_slot

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if not graph.has_edge(u, v):
                    continue
                length = graph.edges[u, v].get('length', 100)
                km = length / 1000.0
                consumed = km * config.battery_consumption_per_km / config.battery_capacity
                soc = max(0.0, soc - consumed)
                cumulative_km += km

                travel_time = data_loader.get_travel_time(u, v, cur_time_slot)
                cur_time_slot = (cur_time_slot + max(1, int(travel_time / 300))) % 288

                # check if we need to charge
                if soc < config.charge_trigger_soc:
                    station = None
                    for sid in ps.used_station_ids:
                        if sid not in visited:
                            station = charging_manager.get_station(sid)
                            break
                    if station:
                        charge_km.append(cumulative_km)
                        charge_soc_before.append(soc)
                        soc = config.charge_target_soc
                        charge_soc_after.append(soc)
                        visited.add(station.station_id)

                dist_points.append(cumulative_km)
                soc_points.append(soc)

            label = labels[idx] if labels else f"Route {idx + 1}"
            colour = colours[idx % len(colours)]
            ax.plot(dist_points, [s * 100 for s in soc_points],
                    color=colour, linewidth=2, label=label)

            # draw charging events as vertical lines
            for ckm, cb, ca in zip(charge_km, charge_soc_before, charge_soc_after):
                ax.vlines(ckm, cb * 100, ca * 100, color=colour, linestyle='--',
                          linewidth=1.5, alpha=0.7)
                ax.scatter([ckm], [cb * 100], marker='v', s=100, color=colour, zorder=5)
                ax.scatter([ckm], [ca * 100], marker='^', s=100, color=colour, zorder=5)

        ax.axhline(config.charge_trigger_soc * 100, color='orange', linestyle=':',
                   linewidth=1.5, label=f'Charge Trigger ({config.charge_trigger_soc * 100:.0f}%)')
        ax.axhline(config.min_soc * 100, color='red', linestyle=':',
                   linewidth=1.5, label=f'Min SoC ({config.min_soc * 100:.0f}%)')
        ax.axhline(config.initial_soc * 100, color='green', linestyle=':',
                   linewidth=1.0, alpha=0.5, label=f'Initial SoC ({config.initial_soc * 100:.0f}%)')

        ax.set_xlabel('Cumulative Distance (km)', fontsize=12)
        ax.set_ylabel('State of Charge (%)', fontsize=12)
        ax.set_title('Battery State of Charge Profile Along Route', fontsize=14)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_convergence(history):
        if not history:
            return None

        generations = [h['generation'] for h in history]
        pareto_sizes = [h['pareto_size'] for h in history]
        best_times = [h['best_time'] for h in history]
        best_fees = [h['best_fee'] for h in history]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('NSGA-II Convergence Analysis', fontsize=14, fontweight='bold')

        axes[0].plot(generations, pareto_sizes, 'b-o', markersize=4, linewidth=2)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Number of Pareto Solutions')
        axes[0].set_title('Pareto Front Size')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(generations, best_times, 'r-o', markersize=4, linewidth=2)
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Best Time (min)')
        axes[1].set_title('Best Travel Time')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(generations, best_fees, 'g-o', markersize=4, linewidth=2)
        axes[2].set_xlabel('Generation')
        axes[2].set_ylabel('Best Fee ($)')
        axes[2].set_title('Best Charging Fee')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_station_usage_analysis(path_details, charging_manager, labels=None):
        all_station_ids = set()
        for ps in path_details:
            all_station_ids.update(ps.used_station_ids)

        if not all_station_ids:
            return None

        stations = {sid: charging_manager.get_station(sid)
                    for sid in all_station_ids
                    if charging_manager.get_station(sid)}

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Charging Station Usage Analysis', fontsize=14, fontweight='bold')
        gs = gridspec.GridSpec(2, 2, figure=fig)

        # top left - station power
        ax1 = fig.add_subplot(gs[0, 0])
        names = [s.title[:25] for s in stations.values()]
        powers = [s.charging_power for s in stations.values()]
        bars = ax1.barh(names, powers, color='steelblue', edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Charging Power (kW)')
        ax1.set_title('Station Charging Power')
        ax1.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, powers):
            ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                     f'{val:.1f}', va='center', fontsize=9)

        # top right - electricity price
        ax2 = fig.add_subplot(gs[0, 1])
        prices = [s.electricity_price for s in stations.values()]
        bars2 = ax2.barh(names, prices, color='coral', edgecolor='black', alpha=0.8)
        ax2.set_xlabel('Price ($/kWh)')
        ax2.set_title('Electricity Price per Station')
        ax2.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars2, prices):
            ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'${val:.3f}', va='center', fontsize=9)

        # bottom left - fee by route
        ax3 = fig.add_subplot(gs[1, 0])
        route_labels = labels if labels else [f"Route {i+1}" for i in range(len(path_details))]
        x = np.arange(len(path_details))
        width = 0.6
        fees = [ps.total_fee for ps in path_details]
        bars3 = ax3.bar(x, fees, width, color=['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0'][:len(fees)],
                        edgecolor='black', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(route_labels, rotation=15, ha='right', fontsize=9)
        ax3.set_ylabel('Total Charging Fee ($)')
        ax3.set_title('Charging Fee by Route')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars3, fees):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'${val:.2f}', ha='center', va='bottom', fontsize=9)

        # bottom right - time vs fee scatter
        ax4 = fig.add_subplot(gs[1, 1])
        stops = [ps.num_charging_stops for ps in path_details]
        times = [ps.total_time_minutes for ps in path_details]
        scatter_colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
        for i, (t, f, s, rl) in enumerate(zip(times, fees, stops, route_labels)):
            ax4.scatter(t, f, s=200 + s * 100,
                        color=scatter_colors[i % len(scatter_colors)],
                        edgecolors='black', zorder=5, label=rl, alpha=0.85)
        ax4.set_xlabel('Total Time (min)')
        ax4.set_ylabel('Total Fee ($)')
        ax4.set_title('Time vs Fee Trade-off')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


    @staticmethod
    def plot_dynamic_journey(graph, journey_result, charging_stations,
                             charging_manager=None):
        from dynamic_router import ReplanEvent

        trigger_colors = {
            'intersection': '#FF6B35',
            'scheduled': '#4ECDC4',
            'soc_threshold': '#E74C3C',
            'post_charging': '#9B59B6',
        }

        fig = go.Figure()

        def path_coords(path):
            lats, lons = [], []
            for node in path:
                if graph.nodes[node].get('is_virtual', False):
                    continue
                try:
                    lats.append(graph.nodes[node]['y'])
                    lons.append(graph.nodes[node]['x'])
                except KeyError:
                    pass
            return lats, lons

        # initial plan (shown faded)
        init_lats, init_lons = path_coords(journey_result.initial_plan)
        fig.add_trace(go.Scattermapbox(
            mode='lines',
            lat=init_lats, lon=init_lons,
            line={'width': 2, 'color': 'gray'},
            name='Initial Plan',
            opacity=0.5
        ))

        # actual path
        act_lats, act_lons = path_coords(journey_result.actual_path)
        fig.add_trace(go.Scattermapbox(
            mode='lines',
            lat=act_lats, lon=act_lons,
            line={'width': 4, 'color': '#2196F3'},
            name='Actual Path'
        ))

        if act_lats:
            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lat=[act_lats[0]], lon=[act_lons[0]],
                marker={'size': 16, 'color': 'green', 'symbol': 'circle'},
                name='Start', showlegend=True
            ))
            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lat=[act_lats[-1]], lon=[act_lons[-1]],
                marker={'size': 22, 'color': '#FF2020',
                        'symbol': 'circle', 'opacity': 1.0},
                text=['<b>END</b>'],
                hoverinfo='text',
                name='End', showlegend=True
            ))

        # group replan events by trigger type
        by_trigger = {}
        for ev in journey_result.replan_events:
            by_trigger.setdefault(ev.trigger, []).append(ev)

        for trigger, events in by_trigger.items():
            ev_lats, ev_lons, ev_texts = [], [], []
            for ev in events:
                node = ev.node_id
                if graph.nodes[node].get('is_virtual', False):
                    continue
                try:
                    ev_lats.append(graph.nodes[node]['y'])
                    ev_lons.append(graph.nodes[node]['x'])
                    ev_texts.append(
                        f"Replan: {ev.trigger}<br>"
                        f"SOC: {ev.soc * 100:.1f}%<br>"
                        f"Slot: {ev.time_slot}"
                    )
                except KeyError:
                    pass
            if ev_lats:
                fig.add_trace(go.Scattermapbox(
                    mode='markers',
                    lat=ev_lats, lon=ev_lons,
                    marker={'size': 10, 'color': trigger_colors.get(trigger, 'yellow')},
                    text=ev_texts,
                    name=f'Replan: {trigger} ({len(events)})'
                ))

        # charging stations that were actually used
        if journey_result.used_station_ids and charging_manager:
            s_lats, s_lons, s_texts = [], [], []
            for sid in journey_result.used_station_ids:
                st = charging_manager.get_station(sid)
                if st:
                    s_lats.append(st.latitude)
                    s_lons.append(st.longitude)
                    s_texts.append(
                        f"Charged here<br>{st.title}<br>"
                        f"{st.charging_power} kW  ${st.electricity_price}/kWh"
                    )
            if s_lats:
                fig.add_trace(go.Scattermapbox(
                    mode='markers',
                    lat=s_lats, lon=s_lons,
                    marker={'size': 18, 'color': '#FFD700', 'symbol': 'circle'},
                    text=s_texts,
                    name=f'Charging stops ({len(s_lats)})'
                ))

            # draw lines from trigger node to station
            for trigger_node, sid in getattr(journey_result, 'charging_events', []):
                st = charging_manager.get_station(sid)
                if not st:
                    continue
                try:
                    t_lat = graph.nodes[trigger_node]['y']
                    t_lon = graph.nodes[trigger_node]['x']
                except KeyError:
                    continue
                fig.add_trace(go.Scattermapbox(
                    mode='lines',
                    lat=[t_lat, st.latitude],
                    lon=[t_lon, st.longitude],
                    line={'width': 2, 'color': '#FFD700'},
                    opacity=0.7,
                    showlegend=False,
                    hoverinfo='skip'
                ))

        # incidents on the map
        incident_colors = {
            'road_closure': '#FF0000',
            'congestion': '#FF8C00',
            'station_failure': '#C800FF',
        }
        incident_symbols = {
            'road_closure': 'circle',
            'congestion': 'circle',
            'station_failure': 'circle',
        }
        by_incident = {}
        for ie in getattr(journey_result, 'incident_events', []):
            by_incident.setdefault(ie.incident_type, []).append(ie)

        for itype, ievents in by_incident.items():
            i_lats, i_lons, i_texts = [], [], []
            for ie in ievents:
                node = ie.node_id
                if isinstance(node, int) and not graph.nodes[node].get('is_virtual', False):
                    try:
                        i_lats.append(graph.nodes[node]['y'])
                        i_lons.append(graph.nodes[node]['x'])
                        i_texts.append(
                            f"Incident: {ie.incident_type}<br>"
                            f"{ie.description}<br>"
                            f"Slot: {ie.time_slot}"
                        )
                    except KeyError:
                        pass
            if i_lats:
                label = itype.replace('_', ' ').title()
                fig.add_trace(go.Scattermapbox(
                    mode='markers',
                    lat=i_lats, lon=i_lons,
                    marker={'size': 14,
                            'color': incident_colors.get(itype, 'red'),
                            'symbol': incident_symbols.get(itype, 'circle')},
                    text=i_texts,
                    name=f'Incident: {label} ({len(ievents)})',
                    opacity=0.9
                ))

        center_lat = np.mean(act_lats) if act_lats else 49.895
        center_lon = np.mean(act_lons) if act_lons else -97.138
        fig.update_layout(
            mapbox=dict(style='open-street-map',
                        center=dict(lat=center_lat, lon=center_lon), zoom=11),
            height=800,
            title={'text': 'Dynamic Journey Map', 'x': 0.5, 'xanchor': 'center'},
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)')
        )
        return fig

    @staticmethod
    def plot_replan_timeline(journey_result):
        events = journey_result.replan_events
        if not events:
            return None

        trigger_colors = {
            'intersection': '#FF6B35',
            'scheduled': '#4ECDC4',
            'soc_threshold': '#E74C3C',
            'post_charging': '#9B59B6',
        }

        indices = [ev.path_index for ev in events]
        socs = [ev.soc * 100 for ev in events]
        triggers = [ev.trigger for ev in events]
        slots = [ev.time_slot for ev in events]
        times_min = [s * 5 for s in slots]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        colors = [trigger_colors.get(t, 'gray') for t in triggers]
        axes[0].scatter(indices, socs, c=colors, s=80, zorder=3)
        axes[0].axhline(journey_result.__class__.__mro__[0]
                        and 25, color='orange', linestyle='--',
                        linewidth=1, label='Trigger SOC (25%)')
        axes[0].set_ylabel('SOC at Replan (%)')
        axes[0].set_title('Battery SOC at Each Replan Event')
        axes[0].set_ylim(0, 100)
        axes[0].grid(True, alpha=0.3)

        trigger_order = ['intersection', 'scheduled', 'soc_threshold', 'post_charging']
        counts = {t: sum(1 for ev in events if ev.trigger == t) for t in trigger_order}
        bar_colors = [trigger_colors[t] for t in trigger_order]
        axes[1].bar(trigger_order, [counts[t] for t in trigger_order],
                    color=bar_colors, edgecolor='black', linewidth=0.5)
        axes[1].set_ylabel('Number of Replan Events')
        axes[1].set_title('Replan Events by Trigger Type')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_static_vs_dynamic(static_scores, journey_result):
        max_t = static_scores[:, 0].max() + 1e-9
        max_f = static_scores[:, 1].max() + 1e-9
        combined = static_scores[:, 0] / max_t + static_scores[:, 1] / max_f
        best_idx = int(np.argmin(combined))
        static_time = static_scores[best_idx, 0]
        static_fee = static_scores[best_idx, 1]

        dyn_time = journey_result.total_time_minutes
        dyn_fee = journey_result.total_fee

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        labels = ['Static', 'Dynamic']
        times = [static_time, dyn_time]
        fees = [static_fee, dyn_fee]
        bar_colors = ['#2196F3', '#FF6B35']

        axes[0].bar(labels, times, color=bar_colors, edgecolor='black', linewidth=0.6)
        axes[0].set_ylabel('Total Time (minutes)')
        axes[0].set_title('Journey Time Comparison')
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(times):
            axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10)

        axes[1].bar(labels, fees, color=bar_colors, edgecolor='black', linewidth=0.6)
        axes[1].set_ylabel('Charging Fee ($)')
        axes[1].set_title('Charging Fee Comparison')
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(fees):
            axes[1].text(i, v + 0.01, f'${v:.2f}', ha='center', fontsize=10)

        plt.suptitle('Static NSGA-II vs Dynamic Rolling-Horizon', fontsize=13,
                     fontweight='bold')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_journey_animation(graph, journey_result, config, data_loader,
                               charging_manager=None, start_time_slot=0,
                               frame_step=1):
        trigger_colors = {
            'intersection': '#FF6B35',
            'scheduled': '#4ECDC4',
            'soc_threshold': '#E74C3C',
            'post_charging': '#9B59B6',
        }

        # filter out virtual nodes
        real_path = []
        real_indices = []
        for idx, node in enumerate(journey_result.actual_path):
            if not graph.nodes[node].get('is_virtual', False):
                try:
                    _ = graph.nodes[node]['y']
                    real_path.append(node)
                    real_indices.append(idx)
                except KeyError:
                    pass

        if len(real_path) < 2:
            return None

        lats = [graph.nodes[n]['y'] for n in real_path]
        lons = [graph.nodes[n]['x'] for n in real_path]

        charge_at_node = {
            trigger_node: config.charge_target_soc
            for trigger_node, _ in getattr(journey_result, 'charging_events', [])
        }

        # simulate SOC along the path
        socs = [config.initial_soc]
        cum_dist_km = [0.0]
        cum_time_min = [0.0]
        current_soc = config.initial_soc
        current_time_slot = start_time_slot
        total_dist = 0.0
        total_time = 0.0

        for i in range(len(real_path) - 1):
            u, v = real_path[i], real_path[i + 1]
            if graph.has_edge(u, v):
                edge = graph.edges[u, v]
                length = edge.get('length', 100)
                travel_time = data_loader.get_travel_time(u, v, current_time_slot)
                distance_km = length / 1000.0
                consumed = distance_km * config.battery_consumption_per_km
                current_soc = max(0.0, current_soc - consumed / config.battery_capacity)
                total_dist += distance_km
                total_time += travel_time / 60.0
                slots = max(1, int(travel_time / 300))
                current_time_slot = (current_time_slot + slots) % 288

            if v in charge_at_node:
                current_soc = charge_at_node[v]

            socs.append(current_soc)
            cum_dist_km.append(total_dist)
            cum_time_min.append(total_time)

        charge_lats, charge_lons, charge_texts = [], [], []
        if charging_manager:
            for sid in journey_result.used_station_ids:
                st = charging_manager.get_station(sid)
                if st:
                    charge_lats.append(st.latitude)
                    charge_lons.append(st.longitude)
                    charge_texts.append(
                        f"Charged: {st.title}<br>"
                        f"{st.charging_power} kW  ${st.electricity_price}/kWh"
                    )

        # build animation frames
        frame_indices = list(range(0, len(real_path), max(1, frame_step)))
        if frame_indices[-1] != len(real_path) - 1:
            frame_indices.append(len(real_path) - 1)

        frames = []
        for fi in frame_indices:
            soc_pct = socs[fi] * 100
            dist = cum_dist_km[fi]
            elapsed = cum_time_min[fi]
            real_idx = real_indices[fi]

            # replan events up to this point
            ev_lats, ev_lons, ev_marker_colors, ev_texts = [], [], [], []
            for ev in journey_result.replan_events:
                if ev.path_index <= real_idx:
                    try:
                        ev_lats.append(graph.nodes[ev.node_id]['y'])
                        ev_lons.append(graph.nodes[ev.node_id]['x'])
                        ev_marker_colors.append(trigger_colors.get(ev.trigger, '#AAAAAA'))
                        ev_texts.append(f"{ev.trigger}<br>SOC:{ev.soc*100:.1f}%")
                    except KeyError:
                        pass

            # charging events up to this frame
            c_lats, c_lons, c_texts = [], [], []
            conn_lats, conn_lons = [], []
            for t_node, sid in getattr(journey_result, 'charging_events', []):
                t_idx = None
                for k, n in enumerate(real_path):
                    if n == t_node:
                        t_idx = k
                        break
                if t_idx is None or t_idx > fi:
                    continue
                if charging_manager:
                    st = charging_manager.get_station(sid)
                    if st:
                        c_lats.append(st.latitude)
                        c_lons.append(st.longitude)
                        c_texts.append(
                            f"Charged: {st.title}<br>"
                            f"{st.charging_power} kW  ${st.electricity_price}/kWh"
                        )
                        try:
                            conn_lats += [graph.nodes[t_node]['y'], st.latitude, None]
                            conn_lons += [graph.nodes[t_node]['x'], st.longitude, None]
                        except KeyError:
                            pass

            # incident events up to this frame
            inc_lats, inc_lons, inc_texts, inc_colors = [], [], [], []
            incident_color_map = {
                'road_closure': '#FF0000',
                'congestion': '#FF8C00',
                'station_failure': '#C800FF',
            }
            for ie in getattr(journey_result, 'incident_events', []):
                if ie.path_index <= real_idx:
                    node = ie.node_id
                    if isinstance(node, int) and not graph.nodes[node].get('is_virtual', False):
                        try:
                            inc_lats.append(graph.nodes[node]['y'])
                            inc_lons.append(graph.nodes[node]['x'])
                            inc_texts.append(f"{ie.incident_type}<br>{ie.description}")
                            inc_colors.append(incident_color_map.get(ie.incident_type, '#FF0000'))
                        except KeyError:
                            pass

            frame_traces = [
                go.Scattermapbox(
                    lat=lats[:fi + 1], lon=lons[:fi + 1],
                    mode='lines',
                    line={'width': 4, 'color': '#2196F3'}
                ),
                go.Scattermapbox(
                    lat=[lats[fi]], lon=[lons[fi]],
                    mode='markers',
                    marker={'size': 18, 'color': '#FF3333'},
                    text=[f"SOC: {soc_pct:.1f}%<br>"
                          f"Dist: {dist:.2f} km<br>"
                          f"Time: {elapsed:.1f} min"],
                    hoverinfo='text'
                ),
                go.Scattermapbox(
                    lat=ev_lats or [None],
                    lon=ev_lons or [None],
                    mode='markers',
                    marker={
                        'size': 11,
                        'color': ev_marker_colors or ['rgba(0,0,0,0)']
                    },
                    text=ev_texts or [''],
                    hoverinfo='text'
                ),
                go.Scattermapbox(
                    lat=c_lats or [None],
                    lon=c_lons or [None],
                    mode='markers',
                    marker={'size': 20, 'color': '#FFD700'},
                    text=c_texts or [''],
                    hoverinfo='text'
                ),
                go.Scattermapbox(
                    lat=conn_lats or [None],
                    lon=conn_lons or [None],
                    mode='lines',
                    line={'width': 2, 'color': '#FFD700'},
                    opacity=0.7,
                    hoverinfo='skip'
                ),
                go.Scattermapbox(
                    lat=inc_lats or [None],
                    lon=inc_lons or [None],
                    mode='markers',
                    marker={'size': 14,
                            'color': inc_colors or ['rgba(0,0,0,0)']},
                    text=inc_texts or [''],
                    hoverinfo='text'
                ),
            ]

            frames.append(go.Frame(
                data=frame_traces,
                name=str(fi),
                layout=go.Layout(
                    title_text=(
                        f"Step {fi}/{len(real_path)-1}  |  "
                        f"SOC: {soc_pct:.1f}%  |  "
                        f"Dist: {dist:.2f} km  |  "
                        f"Time: {elapsed:.1f} min"
                    )
                )
            ))

        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

        slider_steps = [
            dict(
                method='animate',
                args=[[str(fi)],
                      {'frame': {'duration': 0, 'redraw': True},
                       'mode': 'immediate', 'transition': {'duration': 0}}],
                label=str(fi)
            )
            for fi in frame_indices
        ]

        fig = go.Figure(
            data=[
                go.Scattermapbox(lat=[], lon=[], mode='lines',
                                 line={'width': 4, 'color': '#2196F3'}, name='Path taken'),
                go.Scattermapbox(lat=[lats[0]], lon=[lons[0]], mode='markers',
                                 marker={'size': 18, 'color': '#FF3333'}, name='Vehicle'),
                go.Scattermapbox(lat=[], lon=[], mode='markers',
                                 marker={'size': 11, 'color': '#FF6B35'}, name='Replan events'),
                go.Scattermapbox(lat=[], lon=[], mode='markers',
                                 marker={'size': 20, 'color': '#FFD700'}, name='Charging stops'),
                go.Scattermapbox(lat=[], lon=[], mode='lines',
                                 line={'width': 2, 'color': '#FFD700'},
                                 opacity=0.7, showlegend=False),
                go.Scattermapbox(lat=[], lon=[], mode='markers',
                                 marker={'size': 14, 'color': '#FF0000'},
                                 name='Incidents'),
                go.Scattermapbox(lat=[lats[0]], lon=[lons[0]], mode='markers',
                                 marker={'size': 16, 'color': 'green', 'symbol': 'circle'},
                                 name='Start'),
                go.Scattermapbox(lat=[lats[-1]], lon=[lons[-1]], mode='markers',
                                 marker={'size': 22, 'color': '#FF2020',
                                         'symbol': 'circle', 'opacity': 1.0},
                                 text=['<b>END</b>'], hoverinfo='text',
                                 name='End'),
            ],
            frames=frames,
            layout=go.Layout(
                mapbox=dict(style='open-street-map',
                            center=dict(lat=center_lat, lon=center_lon), zoom=11),
                height=800,
                title={'text': 'Dynamic Journey Simulation', 'x': 0.5, 'xanchor': 'center'},
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)'),
                updatemenus=[dict(
                    type='buttons',
                    showactive=False,
                    y=-0.05, x=0.5, xanchor='center',
                    buttons=[
                        dict(
                            label='▶  Play',
                            method='animate',
                            args=[None, {
                                'frame': {'duration': 120, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 0}
                            }]
                        ),
                        dict(
                            label='⏸  Pause',
                            method='animate',
                            args=[[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        ),
                    ]
                )],
                sliders=[dict(
                    active=0,
                    steps=slider_steps,
                    x=0.0, y=0.0,
                    len=1.0,
                    pad={'t': 50},
                    currentvalue={'prefix': 'Node: ', 'visible': True, 'xanchor': 'center'},
                    transition={'duration': 0}
                )]
            )
        )

        return fig
