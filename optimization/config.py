import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Set
from enum import Enum


class MutationType(Enum):
    NONE = 0
    ENABLED = 1


class DominationResult(Enum):
    FIRST_DOMINATES = 1
    SECOND_DOMINATES = 2
    NO_DOMINATION = 0


# predefined routes for winnipeg - (start, end, description, distance_km)
PREDEFINED_ROUTES: List[Tuple] = [
    (288981953,   2566870611, "West to East (longest)",            33.5),
    (367131619,   1688849660, "Southwest to Northeast",            31.5),
    (9666839223,  1688848613, "West to East (central)",            30.2),
    (8151998098,  4410522479, "East to West",                      29.8),
    (1152186186,  1688786938, "Southwest to Northeast (alt)",      29.7),
]

ACTIVE_ROUTE_INDEX: int = 0

START_TIME_SLOT: int = 96


@dataclass
class OptimizationConfig:
    init_population_size: int = 100
    num_generations: int = 50
    tournament_size: int = 2
    mutation_type: MutationType = MutationType.ENABLED
    mutation_probability: float = 0.3
    battery_capacity: float = 10.0
    battery_consumption_per_km: float = 0.18
    min_soc: float = 0.10
    initial_soc: float = 0.80
    charge_trigger_soc: float = 0.25
    charge_target_soc: float = 0.55
    allow_free_charging: bool = False
    random_population: bool = False


@dataclass
class DynamicRoutingConfig:
    replan_interval_slots: int = 4
    min_replan_distance_m: float = 3000.0
    min_replan_traveled_m: float = 5000.0
    intersection_pop_size: int = 20
    intersection_generations: int = 5
    full_replan_pop_size: int = 50
    full_replan_generations: int = 15


@dataclass
class Incident:
    incident_type: str
    start_time_slot: int = 0
    end_time_slot: int = 288
    node_u: int = None
    node_v: int = None
    station_id: str = None
    severity: float = 3.0


@dataclass
class IncidentConfig:
    enabled: bool = False
    incidents: List[Incident] = field(default_factory=list)
    random_incidents: bool = True
    num_random_road_closures: int = 2
    num_random_station_failures: int = 1
    random_seed: int = 42


@dataclass
class PathScores:
    total_time_minutes: float
    total_fee: float
    distance_km: float
    num_charging_stops: int
    battery_feasible: bool = True
    used_station_ids: List[str] = field(default_factory=list)
    charging_trigger_nodes: List[Tuple] = field(default_factory=list)

    def as_array(self):
        return np.array([self.total_time_minutes, self.total_fee])


@dataclass
class ChargingStation:
    station_id: str
    virtual_node_id: str
    node_u: int
    node_v: int
    title: str
    latitude: float
    longitude: float
    distance_to_u_m: float
    distance_to_v_m: float
    charging_power: float
    electricity_price: float
    charging_efficiency: float
    num_ports: int
    availability_schedule: np.ndarray
