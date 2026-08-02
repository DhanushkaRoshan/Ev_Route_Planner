import os
import sys
import csv
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, ROOT)

from config import OptimizationConfig, DynamicRoutingConfig, IncidentConfig, PREDEFINED_ROUTES
from data_processor import WinnipegDataLoader, ChargingStationManager
from route_planner import PathGenerator, PathEvaluator
from optimizer import NSGAIIOptimizer
from dynamic_router import DynamicRouter

DATA_DIR = os.path.join(ROOT, 'data')
RUNS_CSV = os.path.join(SCRIPT_DIR, 'variance_runs.csv')
SUMM_CSV = os.path.join(SCRIPT_DIR, 'variance_summary.csv')
OUT_FIG = os.path.join(SCRIPT_DIR, 'variance_box.png')

N_REPS = 5
ROUTE_IDX = 0
START_SLOT = 96


def main():
    print('[load] graph + congestion + stations')
    loader = WinnipegDataLoader(DATA_DIR)
    graph = loader.load_graph()
    loader.load_congestion_data()
    stations = loader.load_charging_stations()
    loader.add_virtual_charging_nodes(stations)
    manager = ChargingStationManager(stations, 288)

    route = PREDEFINED_ROUTES[ROUTE_IDX]
    start, end = route[0], route[1]

    rows = []
    for rep in range(N_REPS):
        print(f'\n========== rep {rep + 1}/{N_REPS} ==========')
        cfg = OptimizationConfig()

        gen = PathGenerator(graph, loader, manager, cfg)
        ev = PathEvaluator(graph, loader, manager, cfg)
        opt = NSGAIIOptimizer(cfg, gen, ev)
        t0 = time.time()
        solutions, scores = opt.optimize(start, end, START_SLOT)
        st_rt = time.time() - t0

        if not solutions:
            print('  no static solutions; skipping')
            continue

        pareto_size = len(solutions)
        best_time = float(scores[:, 0].min())
        best_fee = float(scores[:, 1].min())
        mean_time = float(scores[:, 0].mean())
        mean_fee = float(scores[:, 1].mean())

        router = DynamicRouter(graph, loader, manager, cfg,
                                DynamicRoutingConfig(), IncidentConfig())
        t0 = time.time()
        journey = router.simulate_journey(start, end, START_SLOT)
        dy_rt = time.time() - t0
        trig_count = {'scheduled': 0, 'intersection': 0,
                      'soc_threshold': 0, 'post_charging': 0, 'incident': 0}
        for ev_ in journey.replan_events:
            trig_count[ev_.trigger] = trig_count.get(ev_.trigger, 0) + 1

        rows.append({
            'rep': rep + 1,
            'pareto_size': pareto_size,
            'best_time': best_time,
            'best_fee': best_fee,
            'mean_pareto_time': mean_time,
            'mean_pareto_fee': mean_fee,
            'static_runtime_s': st_rt,
            'dyn_time': journey.total_time_minutes,
            'dyn_fee': journey.total_fee,
            'dyn_stops': journey.num_charging_stops,
            'dyn_replans': len(journey.replan_events),
            'dyn_intersection': trig_count['intersection'],
            'dyn_scheduled': trig_count['scheduled'],
            'dyn_runtime_s': dy_rt,
        })

    with open(RUNS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\n[csv] {RUNS_CSV}')

    metrics = [k for k in rows[0].keys() if k != 'rep']
    summary = []
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        cv = (std / mean * 100) if mean != 0 else 0.0
        summary.append({
            'metric': m,
            'mean': mean,
            'std': std,
            'min': float(vals.min()),
            'max': float(vals.max()),
            'cv_pct': cv,
        })

    with open(SUMM_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'mean', 'std', 'min', 'max', 'cv_pct'])
        w.writeheader()
        for s in summary:
            w.writerow(s)
    print(f'[csv] {SUMM_CSV}')

    print('\n=== Variance summary across 5 reps ===')
    print(f'{"metric":<22} {"mean":>10} {"std":>10} {"min":>10} {"max":>10} {"cv%":>7}')
    for s in summary:
        print(f'{s["metric"]:<22} {s["mean"]:>10.3f} {s["std"]:>10.3f} '
              f'{s["min"]:>10.3f} {s["max"]:>10.3f} {s["cv_pct"]:>6.1f}%')

    plot_metrics = ['best_time', 'dyn_time', 'best_fee', 'dyn_fee',
                    'pareto_size', 'dyn_replans']
    data = [[r[m] for r in rows] for m in plot_metrics]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(data, tick_labels=plot_metrics)
    ax.set_title(f'Distribution across {len(rows)} reps (baseline config)')
    ax.set_ylabel('value (mixed units)')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()
    print(f'[fig] {OUT_FIG}')


if __name__ == '__main__':
    main()
