import os
import pandas as pd
import networkx as nx
import numpy as np
import osmnx as ox
from typing import Dict, Optional

from config import ChargingStation


class WinnipegDataLoader:

    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.graph = None
        self.congestion_data = {}
        self.charging_stations = {}
        self.num_time_slots = 288

    def load_graph(self):
        print("Loading Winnipeg road network...")
        filepath = os.path.join(self.data_dir, "winnipeg_road_network.graphml")

        self.graph = ox.load_graphml(filepath)

        # convert to simple digraph (no parallel edges)
        simple_graph = nx.DiGraph()
        simple_graph.graph.update(self.graph.graph)

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_id = f"{u}_{v}_{key}"
            if not simple_graph.has_edge(u, v):
                simple_graph.add_edge(u, v,
                                      edge_id=edge_id,
                                      length=self.to_float(data.get('length', 100)),
                                      speed_limit_kmh=self.to_float(data.get('speed_limit_kmh', 50)),
                                      base_travel_time=self.to_float(data.get('base_travel_time', 60)),
                                      road_name=data.get('road_name', 'Unknown'),
                                      road_type=data.get('road_type', 'unclassified'),
                                      geometry=data.get('geometry', None),
                                      is_virtual=False)

        for node in self.graph.nodes():
            if node in simple_graph:
                simple_graph.nodes[node].update(self.graph.nodes[node])
                simple_graph.nodes[node]['is_virtual'] = False

        self.graph = simple_graph
        print(f"Graph loaded: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        return self.graph

    def add_virtual_charging_nodes(self, charging_stations):
        print("\nAdding virtual charging station nodes to graph...")
        num_virtual_nodes = 0
        num_virtual_edges = 0

        for station in charging_stations.values():
            vnode = station.virtual_node_id
            node_u = station.node_u
            node_v = station.node_v

            if vnode not in self.graph:
                self.graph.add_node(vnode,
                                    y=station.latitude,
                                    x=station.longitude,
                                    is_virtual=True,
                                    is_charging_station=True,
                                    station_id=station.station_id)
                num_virtual_nodes += 1

            # add edges connecting virtual node to both endpoints
            if not self.graph.has_edge(vnode, node_u):
                self.graph.add_edge(vnode, node_u,
                                    length=self.to_float(station.distance_to_u_m),
                                    speed_limit_kmh=30,
                                    base_travel_time=self.to_float(station.distance_to_u_m) / 8.33,
                                    road_name='Virtual Access',
                                    road_type='virtual',
                                    is_virtual=True)
                num_virtual_edges += 1

            if not self.graph.has_edge(node_u, vnode):
                self.graph.add_edge(node_u, vnode,
                                    length=self.to_float(station.distance_to_u_m),
                                    speed_limit_kmh=30,
                                    base_travel_time=self.to_float(station.distance_to_u_m) / 8.33,
                                    road_name='Virtual Access',
                                    road_type='virtual',
                                    is_virtual=True)
                num_virtual_edges += 1

            if not self.graph.has_edge(vnode, node_v):
                self.graph.add_edge(vnode, node_v,
                                    length=self.to_float(station.distance_to_v_m),
                                    speed_limit_kmh=30,
                                    base_travel_time=self.to_float(station.distance_to_v_m) / 8.33,
                                    road_name='Virtual Access',
                                    road_type='virtual',
                                    is_virtual=True)
                num_virtual_edges += 1

            if not self.graph.has_edge(node_v, vnode):
                self.graph.add_edge(node_v, vnode,
                                    length=station.distance_to_v_m,
                                    speed_limit_kmh=30,
                                    base_travel_time=station.distance_to_v_m / 8.33,
                                    road_name='Virtual Access',
                                    road_type='virtual',
                                    is_virtual=True)
                num_virtual_edges += 1

        print(f"  Added {num_virtual_nodes} virtual nodes and {num_virtual_edges} virtual edges")
        print(f"  Total graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

    def load_congestion_data(self):
        print("Loading traffic congestion data...")
        filepath = os.path.join(self.data_dir, "winnipeg_traffic_data.csv")
        df = pd.read_csv(filepath)

        for _, row in df.iterrows():
            edge_id = row['edge_id']
            profile_cols = [f'time_slot_{t}' for t in range(self.num_time_slots)]
            profile = row[profile_cols].values
            self.congestion_data[edge_id] = {
                'node_u': row['node_u'],
                'node_v': row['node_v'],
                'base_travel_time': self.to_float(row['base_travel_time']),
                'congestion_profile': profile
            }

        print(f"Loaded congestion for {len(self.congestion_data)} edges")

    def load_charging_stations(self):
        print("Loading EV charging stations.")
        stations_file = os.path.join(self.data_dir, "ev_stations.csv")
        stations_df = pd.read_csv(stations_file)

        availability_file = os.path.join(self.data_dir, "ev_availability.csv")
        availability_df = pd.read_csv(availability_file)

        if 'station_id' not in stations_df.columns:
            raise ValueError("CSV missing virtual_node_id column")

        for idx, (_, row) in enumerate(stations_df.iterrows()):
            station_id = str(row['station_id'])
            virtual_node_id = str(row['station_id'])

            col_name = f'station_{idx}'
            if col_name in availability_df.columns:
                availability = availability_df[col_name].values
            else:
                # fallback - assume half ports available
                availability = np.full(self.num_time_slots, row['num_ports'] * 0.5)

            station = ChargingStation(
                station_id=station_id,
                virtual_node_id=virtual_node_id,
                node_u=int(row['nearest_edge_u']),
                node_v=int(row['nearest_edge_v']),
                title=row['title'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                distance_to_u_m=self.to_float(row['distance_to_u_m']),
                distance_to_v_m=self.to_float(row['distance_to_v_m']),
                charging_power=self.to_float(row['charging_power_kw']) if not pd.isna(row['charging_power_kw']) else 15.0,
                electricity_price=self.to_float(row['electricity_price_per_kwh']) if not pd.isna(row['electricity_price_per_kwh']) else 0.5,
                charging_efficiency=self.to_float(row['charging_efficiency']) if not pd.isna(row['charging_efficiency']) else 1.0,
                num_ports=int(row['num_ports']),
                availability_schedule=availability
            )
            self.charging_stations[virtual_node_id] = station

        print(f" Loaded {len(self.charging_stations)} charging stations")
        return self.charging_stations

    def get_travel_time(self, u, v, time_slot):
        if not self.graph.has_edge(u, v):
            return float('inf')

        edge_data = self.graph.edges[u, v]
        if edge_data.get('is_virtual', False):
            return edge_data.get('base_travel_time', 60)

        edge_id = edge_data.get('edge_id', f"{u}_{v}_0")
        if edge_id in self.congestion_data:
            time_slot = int(time_slot) % self.num_time_slots
            return self.congestion_data[edge_id]['congestion_profile'][time_slot]

        return edge_data.get('base_travel_time', 60)

    def to_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


class ChargingStationManager:

    def __init__(self, charging_stations, num_time_slots=288):
        self.stations = charging_stations
        self.num_time_slots = num_time_slots
        self.port_usage = {}
        for vid in self.stations:
            self.port_usage[vid] = np.zeros(num_time_slots)

    def is_charging_station(self, node):
        return str(node) in self.stations

    def get_station(self, node):
        return self.stations.get(str(node))

    def get_available_ports(self, node, time_slot):
        if not self.is_charging_station(node):
            return 0
        vid = str(node)
        station = self.stations[vid]
        time_slot = int(time_slot) % self.num_time_slots
        available = station.availability_schedule[time_slot] - self.port_usage[vid][time_slot]
        return max(0, int(available))

    def reserve_port(self, node, start_time_slot, duration_slots):
        if not self.is_charging_station(node):
            return False
        vid = str(node)
        start_time_slot = int(start_time_slot) % self.num_time_slots

        # first check if all slots are available
        for t in range(duration_slots):
            ts = (start_time_slot + t) % self.num_time_slots
            if self.get_available_ports(node, ts) < 1:
                return False

        # then actually reserve
        for t in range(duration_slots):
            ts = (start_time_slot + t) % self.num_time_slots
            self.port_usage[vid][ts] += 1

        return True

    def release_port(self, node, start_time_slot, duration_slots):
        if not self.is_charging_station(node):
            return
        vid = str(node)
        start_time_slot = int(start_time_slot) % self.num_time_slots
        for t in range(duration_slots):
            ts = (start_time_slot + t) % self.num_time_slots
            self.port_usage[vid][ts] = max(0, self.port_usage[vid][ts] - 1)

    def reset_usage(self):
        for node in self.port_usage:
            self.port_usage[node] = np.zeros(self.num_time_slots)
