import random as _random
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set

from config import OptimizationConfig, DynamicRoutingConfig, IncidentConfig, Incident, ChargingStation
from route_planner import PathGenerator, PathEvaluator
from optimizer import NSGAIIOptimizer


@dataclass
class IncidentEvent:
    incident_type: str
    node_id: int
    time_slot: int
    path_index: int
    description: str


@dataclass
class ReplanEvent:
    trigger: str
    node_id: int
    time_slot: int
    soc: float
    path_index: int


@dataclass
class JourneyResult:
    actual_path: List
    initial_plan: List
    replan_events: List[ReplanEvent]
    total_distance_km: float
    total_time_minutes: float
    total_fee: float
    num_charging_stops: int
    battery_feasible: bool
    used_station_ids: List[str] = field(default_factory=list)
    charging_events: List[Tuple[int, str]] = field(default_factory=list)
    incident_events: List[IncidentEvent] = field(default_factory=list)

    def summary(self):
        trigger_counts = {}
        for ev in self.replan_events:
            trigger_counts[ev.trigger] = trigger_counts.get(ev.trigger, 0) + 1
        lines = [
            f"  Path nodes       : {len(self.actual_path)}",
            f"  Distance         : {self.total_distance_km:.2f} km",
            f"  Total time       : {self.total_time_minutes:.2f} min",
            f"  Charging fee     : ${self.total_fee:.2f}",
            f"  Charging stops   : {self.num_charging_stops}",
            f"  Battery feasible : {'Yes' if self.battery_feasible else 'No'}",
            f"  Replan events    : {len(self.replan_events)}",
            f"  Incident events  : {len(self.incident_events)}",
        ]
        for t, c in sorted(trigger_counts.items()):
            lines.append(f"    - {t:<20}: {c}")
        return "\n".join(lines)


class DynamicRouter:

    def __init__(self, graph, data_loader, charging_manager,
                 config, dynamic_config=None, incident_config=None):
        self.graph = graph
        self.data_loader = data_loader
        self.charging_manager = charging_manager
        self.config = config
        self.dynamic_config = dynamic_config or DynamicRoutingConfig()
        self.incident_config = incident_config or IncidentConfig()
        self.path_evaluator = PathEvaluator(graph, data_loader, charging_manager, config)
        self._incidents = self._prepare_incidents()

    def simulate_journey(self, start, end, start_time_slot=0):
        print("\n[DynamicRouter] Running initial NSGA-II optimization...")
        solutions, scores = self._run_optimizer(
            start, end, start_time_slot,
            self.config.initial_soc, self.config
        )
        if not solutions:
            print("[DynamicRouter] Initial optimization failed, using shortest path")
            try:
                fallback = nx.shortest_path(self.graph, start, end, weight='length')
            except nx.NetworkXNoPath:
                fallback = [start, end]
            solutions, scores = [fallback], np.zeros((1, 2))

        current_plan = self._select_best_path(solutions, scores)
        initial_plan = list(current_plan)
        print(f"[DynamicRouter] Initial plan has {len(initial_plan)} nodes. Starting sim...")

        actual_path = [start]
        current_node = start
        current_soc = self.config.initial_soc
        current_time_slot = start_time_slot
        total_dist_m = 0.0
        total_travel_s = 0.0
        total_charging_s = 0.0
        total_fee = 0.0
        num_charging_stops = 0
        battery_feasible = True
        visited_stations = set()
        replan_events = []
        charging_events = []
        incident_events = []

        if self._incidents:
            print(f"[DynamicRouter] {len(self._incidents)} incident(s) active in simulation.")

        last_replan_slot = start_time_slot
        last_replan_road = ''
        last_replan_dist_m = 0.0
        plan_ptr = 0
        turned_onto_new_road = False

        # main simulation loop
        while current_node != end:
            if plan_ptr >= len(current_plan) - 1:
                new_plan = self._full_replan(current_node, end,
                                             current_time_slot, current_soc)
                if new_plan and len(new_plan) > 1:
                    current_plan = new_plan
                    plan_ptr = 0
                else:
                    break

            next_node = current_plan[plan_ptr + 1]

            # skip virtual nodes
            if self.graph.nodes[next_node].get('is_virtual', False):
                plan_ptr += 1
                continue

            # check for road closure incidents
            if self.incident_config.enabled:
                blocked = self._blocked_edges(current_time_slot)
                if (current_node, next_node) in blocked:
                    print(f"[Incident] Road closure on edge {current_node}->{next_node} "
                          f"at slot {current_time_slot}. Re-planning...")
                    incident_events.append(IncidentEvent(
                        incident_type='road_closure',
                        node_id=current_node,
                        time_slot=current_time_slot,
                        path_index=len(actual_path) - 1,
                        description=f"Road closure: edge {current_node}->{next_node}"
                    ))
                    new_plan = self._replan_avoiding_incidents(
                        current_node, end, current_time_slot, current_soc, full=True)
                    if new_plan and len(new_plan) > 1:
                        current_plan = new_plan
                        plan_ptr = 0
                        last_replan_slot = current_time_slot
                        replan_events.append(ReplanEvent(
                            trigger='incident',
                            node_id=current_node,
                            time_slot=current_time_slot,
                            soc=current_soc,
                            path_index=len(actual_path) - 1
                        ))
                    continue

            if not self.graph.has_edge(current_node, next_node):
                new_plan = self._full_replan(current_node, end,
                                             current_time_slot, current_soc)
                if new_plan and len(new_plan) > 1:
                    current_plan = new_plan
                    plan_ptr = 0
                    next_node = current_plan[1]
                else:
                    battery_feasible = False
                    break

            edge_data = self.graph.edges[current_node, next_node]
            length = edge_data.get('length', 100)
            travel_time = self.data_loader.get_travel_time(
                current_node, next_node, current_time_slot)

            # apply congestion penalty from incidents
            if self.incident_config.enabled:
                for cu, cv, sev in self._congested_edges(current_time_slot):
                    if cu == current_node and cv == next_node:
                        travel_time *= sev
                        incident_events.append(IncidentEvent(
                            incident_type='congestion',
                            node_id=current_node,
                            time_slot=current_time_slot,
                            path_index=len(actual_path) - 1,
                            description=f"Congestion (x{sev:.1f}) on edge {current_node}->{next_node}"
                        ))
                        break

            dist_km = length / 1000.0
            battery_consumed = dist_km * self.config.battery_consumption_per_km
            current_soc = max(0.0, current_soc - battery_consumed / self.config.battery_capacity)

            slots_elapsed = max(1, int(travel_time / 300))
            current_time_slot = (current_time_slot + slots_elapsed) % 288
            total_dist_m += length
            total_travel_s += travel_time

            current_node = next_node
            actual_path.append(current_node)
            plan_ptr += 1

            if current_node == end:
                break

            # handle charging if battery low
            if current_soc < self.config.charge_trigger_soc:
                failed = self._failed_station_ids(current_time_slot) if self.incident_config.enabled else set()
                if failed:
                    incident_events.append(IncidentEvent(
                        incident_type='station_failure',
                        node_id=current_node,
                        time_slot=current_time_slot,
                        path_index=len(actual_path) - 1,
                        description=f"Station(s) offline: {', '.join(failed)}"
                    ))
                station = self.path_evaluator._find_nearest_station(
                    current_node, visited_stations | failed, current_time_slot,
                    remaining_path=current_plan[plan_ptr:])

                if station:
                    charge_needed = max(
                        0.0,
                        (self.config.charge_target_soc - current_soc)
                        * self.config.battery_capacity
                    )
                    if charge_needed > 0:
                        charging_h = charge_needed / station.charging_power
                        total_charging_s += charging_h * 3600
                        actual_charge = charge_needed / station.charging_efficiency
                        total_fee += actual_charge * station.electricity_price
                        num_charging_stops += 1
                        current_soc = self.config.charge_target_soc
                        visited_stations.add(station.virtual_node_id)
                        slots_charging = max(1, int(charging_h * 12))
                        current_time_slot = (current_time_slot + slots_charging) % 288
                        charging_events.append((current_node, station.virtual_node_id))
                    replan_trigger = 'post_charging'
                else:
                    if current_soc < self.config.min_soc:
                        battery_feasible = False
                        total_fee += 10000
                    replan_trigger = 'soc_threshold'

                new_plan = self._full_replan(current_node, end,
                                             current_time_slot, current_soc)
                if new_plan:
                    current_plan = new_plan
                    plan_ptr = 0
                    last_replan_slot = current_time_slot
                    last_replan_dist_m = total_dist_m
                replan_events.append(ReplanEvent(
                    trigger=replan_trigger,
                    node_id=current_node,
                    time_slot=current_time_slot,
                    soc=current_soc,
                    path_index=len(actual_path) - 1
                ))
                continue

            # periodic scheduled replan check
            slots_since_replan = (current_time_slot - last_replan_slot) % 288
            remaining_m = self._remaining_distance_m(current_node, end)
            if slots_since_replan >= self.dynamic_config.replan_interval_slots:
                last_replan_slot = current_time_slot
                traveled_since = total_dist_m - last_replan_dist_m
                curr_road_name = ''
                if len(actual_path) >= 2:
                    e = self.graph.edges[actual_path[-2], actual_path[-1]]
                    curr_road_name = e.get('road_name') or e.get('name', '')
                same_road = bool(curr_road_name and curr_road_name == last_replan_road)
                enough_traveled = traveled_since >= self.dynamic_config.min_replan_traveled_m
                if (remaining_m >= self.dynamic_config.min_replan_distance_m
                        and not same_road
                        and enough_traveled):
                    last_replan_road = curr_road_name
                    last_replan_dist_m = total_dist_m
                    new_plan = self._full_replan(current_node, end,
                                                 current_time_slot, current_soc)
                    if new_plan:
                        current_plan = new_plan
                        plan_ptr = 0
                    replan_events.append(ReplanEvent(
                        trigger='scheduled',
                        node_id=current_node,
                        time_slot=current_time_slot,
                        soc=current_soc,
                        path_index=len(actual_path) - 1
                    ))
                    continue

            # intersection-based replan
            if len(actual_path) >= 2:
                entered_edge = self.graph.edges[actual_path[-2], current_node]
                prev_road = ''
                if len(actual_path) >= 3:
                    prev_edge = self.graph.edges[actual_path[-3], actual_path[-2]]
                    prev_road = prev_edge.get('road_name') or prev_edge.get('name', '')
                curr_road = entered_edge.get('road_name') or entered_edge.get('name', '')
                turned_onto_new_road = (curr_road != prev_road)

            if (len(actual_path) >= 2
                    and turned_onto_new_road
                    and remaining_m >= self.dynamic_config.min_replan_distance_m):
                new_plan = self._intersection_replan(current_node, end,
                                                     current_time_slot, current_soc)
                if new_plan:
                    current_plan = new_plan
                    plan_ptr = 0
                replan_events.append(ReplanEvent(
                    trigger='intersection',
                    node_id=current_node,
                    time_slot=current_time_slot,
                    soc=current_soc,
                    path_index=len(actual_path) - 1
                ))

        total_time_minutes = (total_travel_s + total_charging_s) / 60.0
        print(f"[DynamicRouter] Done. {len(actual_path)} nodes, "
              f"{len(replan_events)} replan events "
              f"({sum(1 for e in replan_events if e.trigger == 'intersection')} intersection, "
              f"{sum(1 for e in replan_events if e.trigger == 'scheduled')} scheduled, "
              f"{sum(1 for e in replan_events if e.trigger in ('soc_threshold', 'post_charging'))} soc/charge).")

        return JourneyResult(
            actual_path=actual_path,
            initial_plan=initial_plan,
            replan_events=replan_events,
            total_distance_km=total_dist_m / 1000.0,
            total_time_minutes=total_time_minutes,
            total_fee=total_fee,
            num_charging_stops=num_charging_stops,
            battery_feasible=battery_feasible,
            used_station_ids=list(visited_stations),
            charging_events=charging_events,
            incident_events=incident_events
        )

    def _prepare_incidents(self):
        ic = self.incident_config
        if not ic.enabled:
            return []
        if ic.incidents:
            return list(ic.incidents)
        if not ic.random_incidents:
            return []

        rng = _random.Random(ic.random_seed)
        incidents = []

        real_edges = [
            (u, v) for u, v in self.graph.edges()
            if not self.graph.nodes[u].get('is_virtual', False)
            and not self.graph.nodes[v].get('is_virtual', False)
        ]
        if real_edges:
            chosen = rng.sample(real_edges, min(ic.num_random_road_closures, len(real_edges)))
            for u, v in chosen:
                start_slot = rng.randint(0, 144)
                incidents.append(Incident(
                    incident_type='road_closure',
                    node_u=u, node_v=v,
                    start_time_slot=start_slot,
                    end_time_slot=start_slot + rng.randint(12, 72)
                ))

        station_ids = list(self.charging_manager.stations.keys())
        if station_ids:
            chosen_sids = rng.sample(station_ids, min(ic.num_random_station_failures, len(station_ids)))
            for sid in chosen_sids:
                start_slot = rng.randint(0, 144)
                incidents.append(Incident(
                    incident_type='station_failure',
                    station_id=sid,
                    start_time_slot=start_slot,
                    end_time_slot=start_slot + rng.randint(24, 96)
                ))

        return incidents

    def _active_incidents(self, time_slot):
        return [i for i in self._incidents
                if i.start_time_slot <= time_slot < i.end_time_slot]

    def _blocked_edges(self, time_slot):
        return {(i.node_u, i.node_v) for i in self._active_incidents(time_slot)
                if i.incident_type == 'road_closure'}

    def _failed_station_ids(self, time_slot):
        return {i.station_id for i in self._active_incidents(time_slot)
                if i.incident_type == 'station_failure' and i.station_id}

    def _congested_edges(self, time_slot):
        return [(i.node_u, i.node_v, i.severity)
                for i in self._active_incidents(time_slot)
                if i.incident_type == 'congestion']

    def _replan_avoiding_incidents(self, start, end, time_slot,
                                   current_soc, full=True):
        blocked = self._blocked_edges(time_slot)
        congested = self._congested_edges(time_slot)
        modified = []

        # temporarily jack up edge weights for blocked/congested edges
        for u, v in blocked:
            if self.graph.has_edge(u, v):
                orig = self.graph.edges[u, v].get('length', 100)
                self.graph.edges[u, v]['length'] = 1e9
                modified.append((u, v, orig))

        for u, v, sev in congested:
            if self.graph.has_edge(u, v):
                orig = self.graph.edges[u, v].get('length', 100)
                self.graph.edges[u, v]['length'] = orig * sev
                modified.append((u, v, orig))

        result = (self._full_replan(start, end, time_slot, current_soc) if full
                  else self._intersection_replan(start, end, time_slot, current_soc))

        # restore original weights
        for u, v, orig in modified:
            if self.graph.has_edge(u, v):
                self.graph.edges[u, v]['length'] = orig

        return result

    def _remaining_distance_m(self, node, end):
        try:
            x1 = self.graph.nodes[node]['x']
            y1 = self.graph.nodes[node]['y']
            x2 = self.graph.nodes[end]['x']
            y2 = self.graph.nodes[end]['y']
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 111000
        except Exception:
            return float('inf')

    def _full_replan(self, start, end, time_slot, current_soc):
        cfg = self._make_config(current_soc,
                                self.dynamic_config.full_replan_pop_size,
                                self.dynamic_config.full_replan_generations)
        solutions, scores = self._run_optimizer(start, end, time_slot, current_soc, cfg)
        return self._select_best_path(solutions, scores) if solutions else None

    def _intersection_replan(self, start, end, time_slot, current_soc):
        cfg = self._make_config(current_soc,
                                self.dynamic_config.intersection_pop_size,
                                self.dynamic_config.intersection_generations)
        solutions, scores = self._run_optimizer(start, end, time_slot, current_soc, cfg)
        return self._select_best_path(solutions, scores) if solutions else None

    def _make_config(self, current_soc, pop_size, num_gen):
        return OptimizationConfig(
            init_population_size=pop_size,
            num_generations=num_gen,
            tournament_size=self.config.tournament_size,
            mutation_type=self.config.mutation_type,
            mutation_probability=self.config.mutation_probability,
            battery_capacity=self.config.battery_capacity,
            battery_consumption_per_km=self.config.battery_consumption_per_km,
            min_soc=self.config.min_soc,
            initial_soc=current_soc,
            charge_trigger_soc=self.config.charge_trigger_soc,
            charge_target_soc=self.config.charge_target_soc,
            allow_free_charging=self.config.allow_free_charging,
        )

    def _run_optimizer(self, start, end, time_slot, initial_soc, config):
        generator = PathGenerator(self.graph, self.data_loader,
                                  self.charging_manager, config)
        evaluator = PathEvaluator(self.graph, self.data_loader,
                                  self.charging_manager, config)
        opt = NSGAIIOptimizer(config, generator, evaluator)
        return opt.optimize(start, end, time_slot)

    def _select_best_path(self, solutions, scores):
        if not solutions:
            return None
        if len(solutions) == 1:
            return solutions[0]
        # normalize and pick lowest combined score
        max_time = scores[:, 0].max() + 1e-9
        max_fee = scores[:, 1].max() + 1e-9
        combined = scores[:, 0] / max_time + scores[:, 1] / max_fee
        return solutions[int(np.argmin(combined))]
