import random
import networkx as nx
import numpy as np
from typing import List, Optional

from config import OptimizationConfig, PathScores, ChargingStation


class PathGenerator:

    def __init__(self, graph, data_loader=None,
                 charging_manager=None, config=None):
        self.graph = graph
        self.data_loader = data_loader
        self.charging_manager = charging_manager
        self.config = config

    def is_valid_path(self, path, start, end):
        return len(path) > 0 and path[0] == start and path[-1] == end

    def get_distance_to_goal(self, node, goal):
        try:
            x1, y1 = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
            x2, y2 = self.graph.nodes[goal]['x'], self.graph.nodes[goal]['y']
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 111000
        except Exception:
            return float('inf')

    def calculate_edge_weight(self, u, v, time_slot,
                              current_soc, distance_to_goal):
        if not self.graph.has_edge(u, v):
            return float('inf')

        edge_data = self.graph.edges[u, v]
        travel_time = self.data_loader.get_travel_time(u, v, time_slot)
        time_weight = travel_time / 60.0

        length = edge_data.get('length', 100)
        dist_km = length / 1000.0
        battery_consumed = dist_km * self.config.battery_consumption_per_km
        new_soc = current_soc - battery_consumed / self.config.battery_capacity

        # penalize low battery
        if new_soc < self.config.min_soc:
            battery_penalty = 500
        elif new_soc < 0.3:
            battery_penalty = 50 * (0.3 - new_soc)
        else:
            battery_penalty = 0

        if distance_to_goal > 0:
            distance_heuristic = length / distance_to_goal * 10
        else:
            distance_heuristic = 0

        return time_weight + battery_penalty + distance_heuristic

    def generate_intelligent_path(self, start, end, start_time_slot=0):
        if not self.data_loader or not self.config:
            return self.generate_random_path(start, end)

        path = []
        current_node = start
        current_soc = self.config.initial_soc
        current_time_slot = start_time_slot
        max_iterations = 1000

        for _ in range(max_iterations):
            if not path or path[-1] != current_node:
                path.append(current_node)

            if current_node == end:
                return path

            neighbors = list(self.graph.successors(current_node))
            if not neighbors:
                return path

            unvisited = [n for n in neighbors if n not in path]
            if not unvisited:
                return path

            # prefer real nodes over virtual charging nodes
            real_neighbors = [n for n in unvisited
                              if not self.graph.nodes[n].get('is_virtual', False)]
            if not real_neighbors:
                real_neighbors = unvisited

            distance_to_goal = self.get_distance_to_goal(current_node, end)
            neighbor_scores = []

            for neighbor in real_neighbors:
                weight = self.calculate_edge_weight(
                    current_node, neighbor, current_time_slot,
                    current_soc, distance_to_goal
                )
                neighbor_scores.append((neighbor, weight))

            neighbor_scores.sort(key=lambda x: x[1])
            top_k = max(1, len(neighbor_scores) // 3)
            top_neighbors = neighbor_scores[:top_k]
            weights = [1.0 / (w + 1) for _, w in top_neighbors]
            total = sum(weights)
            probs = [w / total for w in weights]
            next_node = np.random.choice([n for n, _ in top_neighbors], p=probs)

            # update battery and time
            edge_data = self.graph.edges[current_node, next_node]
            length = edge_data.get('length', 100)
            dist_km = length / 1000.0
            battery_consumed = dist_km * self.config.battery_consumption_per_km
            current_soc = max(0, current_soc - battery_consumed / self.config.battery_capacity)

            travel_time = self.data_loader.get_travel_time(current_node, next_node, current_time_slot)
            current_time_slot = (current_time_slot + max(1, int(travel_time / 300))) % 288
            current_node = next_node

        return path

    def generate_random_path(self, start, end):
        path = []
        current_node = start
        max_iterations = 1000

        for _ in range(max_iterations):
            if not path or path[-1] != current_node:
                path.append(current_node)

            if current_node == end:
                return path

            neighbors = list(self.graph.successors(current_node))
            if not neighbors:
                return path

            unvisited_neighbors = [n for n in neighbors if n not in path]
            if not unvisited_neighbors:
                return path

            real_neighbors = [n for n in unvisited_neighbors
                              if not self.graph.nodes[n].get('is_virtual', False)]

            if real_neighbors and random.random() > 0.1:
                current_node = random.choice(real_neighbors)
            elif unvisited_neighbors:
                current_node = random.choice(unvisited_neighbors)
            else:
                return path

        return path

    def generate_valid_intelligent_path(self, start, end, start_time_slot=0):
        for _ in range(5):
            path = self.generate_intelligent_path(start, end, start_time_slot)
            if self.is_valid_path(path, start, end):
                return path
        # fall back to shortest path
        try:
            return nx.shortest_path(self.graph, start, end, weight='length')
        except nx.NetworkXNoPath:
            return [start, end]

    def generate_valid_random_path(self, start, end):
        for _ in range(10):
            path = self.generate_random_path(start, end)
            if self.is_valid_path(path, start, end):
                return path
        try:
            return nx.shortest_path(self.graph, start, end, weight='length')
        except nx.NetworkXNoPath:
            return [start, end]

    def generate_via_station_path(self, start, end, station):
        for waypoint in [station.node_u, station.node_v]:
            try:
                seg1 = nx.shortest_path(self.graph, start, waypoint, weight='length')
                seg2 = nx.shortest_path(self.graph, waypoint, end, weight='length')
                return seg1 + seg2[1:]
            except nx.NetworkXNoPath:
                continue
        return []

    def create_initial_population(self, start, end, size, start_time_slot=0):
        population = []

        if self.config and self.config.random_population:
            print(f"  Generating {size} random paths (random_population=True)...")
            for _ in range(size):
                population.append(self.generate_valid_random_path(start, end))
            print(f"  Initial population: {len(population)} paths")
            return population

        intelligent_count = int(size * 0.4)
        via_count = int(size * 0.3)
        random_count = size - intelligent_count - via_count

        print(f"  Generating {intelligent_count} intelligent paths...")
        for i in range(intelligent_count):
            if i % 20 == 0 and i > 0:
                print(f"    Generated {i}/{intelligent_count} intelligent paths...")
            population.append(self.generate_valid_intelligent_path(start, end, start_time_slot))

        print(f"  Generating {via_count} via-station paths for objective diversity...")
        stations_list = list(self.charging_manager.stations.values()) if self.charging_manager else []
        for i in range(via_count):
            if stations_list:
                station = stations_list[i % len(stations_list)]
                path = self.generate_via_station_path(start, end, station)
                if self.is_valid_path(path, start, end):
                    population.append(path)
                    continue
            population.append(self.generate_valid_intelligent_path(start, end, start_time_slot))

        print(f"  Generating {random_count} random paths for diversity...")
        for _ in range(random_count):
            population.append(self.generate_valid_random_path(start, end))

        print(f"  Initial population: {len(population)} paths")
        return population


class PathEvaluator:

    def __init__(self, graph, data_loader, charging_manager, config):
        self.graph = graph
        self.data_loader = data_loader
        self.charging_manager = charging_manager
        self.config = config

    def get_charging_stops_in_path(self, path):
        return [n for n in path if self.graph.nodes[n].get('is_virtual', False)]

    def _find_nearest_station(self, node, visited_stations,
                              time_slot, remaining_path=None):
        candidates = []
        for sid, station in self.charging_manager.stations.items():
            if sid in visited_stations:
                continue
            if not self.config.allow_free_charging and station.electricity_price == 0:
                continue
            if self.charging_manager.get_available_ports(sid, time_slot) <= 0:
                continue
            candidates.append(station)

        if not candidates:
            return None

        # try to pick station closest to remaining route
        route_coords = []
        if remaining_path:
            for n in remaining_path:
                nd = self.graph.nodes[n]
                x, y = nd.get('x'), nd.get('y')
                if x is not None and y is not None:
                    route_coords.append((x, y))

        if route_coords:
            candidates.sort(key=lambda st: min(
                (st.longitude - rx) ** 2 + (st.latitude - ry) ** 2
                for rx, ry in route_coords
            ))
        else:
            try:
                nd = self.graph.nodes[node]
                cx, cy = nd.get('x'), nd.get('y')
            except Exception:
                cx, cy = None, None
            if cx is not None and cy is not None:
                candidates.sort(key=lambda st: (st.longitude - cx) ** 2 + (st.latitude - cy) ** 2)

        return candidates[0]

    def evaluate_path(self, path, start_time_slot=0):
        total_distance = 0
        total_travel_time = 0
        total_charging_time = 0
        total_charging_fee = 0
        num_charging_stops = 0
        visited_stations = set()
        charging_trigger_nodes = []

        current_soc = self.config.initial_soc
        current_time_slot = start_time_slot
        battery_feasible = True

        for i in range(len(path) - 1):
            curr = path[i]
            nxt = path[i + 1]

            if not self.graph.has_edge(curr, nxt):
                battery_feasible = False
                total_charging_fee += 10000
                continue

            edge_data = self.graph.edges[curr, nxt]
            length = edge_data.get('length', 100)
            total_distance += length

            travel_time = self.data_loader.get_travel_time(curr, nxt, current_time_slot)
            total_travel_time += travel_time

            time_elapsed_slots = max(1, int(travel_time / 300))
            current_time_slot = (current_time_slot + time_elapsed_slots) % self.charging_manager.num_time_slots

            dist_km = length / 1000.0
            battery_consumed = dist_km * self.config.battery_consumption_per_km
            current_soc -= battery_consumed / self.config.battery_capacity

            # check if we need to charge
            if current_soc < self.config.charge_trigger_soc:
                remaining_path = path[i:]
                station = self._find_nearest_station(curr, visited_stations,
                                                     current_time_slot, remaining_path)
                if station:
                    target_soc = self.config.charge_target_soc
                    charge_needed = max(0.0, (target_soc - current_soc) * self.config.battery_capacity)
                    if charge_needed > 0:
                        charging_time_hours = charge_needed / station.charging_power
                        total_charging_time += charging_time_hours * 3600
                        actual_charge = charge_needed / station.charging_efficiency
                        total_charging_fee += actual_charge * station.electricity_price
                        num_charging_stops += 1
                        current_soc = target_soc
                        visited_stations.add(station.virtual_node_id)
                        charging_trigger_nodes.append((curr, station.virtual_node_id))
                elif current_soc < self.config.min_soc:
                    battery_feasible = False
                    total_charging_fee += 10000

        total_time_min = (total_travel_time + total_charging_time) / 60.0

        return PathScores(
            total_time_minutes=total_time_min,
            total_fee=total_charging_fee,
            distance_km=total_distance / 1000.0,
            num_charging_stops=num_charging_stops,
            battery_feasible=battery_feasible,
            used_station_ids=list(visited_stations),
            charging_trigger_nodes=charging_trigger_nodes
        )

    def evaluate_population(self, population, start_time_slot=0):
        scores = np.zeros((len(population), 2))
        self.charging_manager.reset_usage()

        for i, path in enumerate(population):
            path_scores = self.evaluate_path(path, start_time_slot)
            scores[i, :] = path_scores.as_array()

        self.charging_manager.reset_usage()
        return scores
