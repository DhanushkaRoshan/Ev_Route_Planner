import os
import sys
import csv
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
OUT_CSV = os.path.join(SCRIPT_DIR, 'matched_comparison.csv')
OUT_FIG = os.path.join(SCRIPT_DIR, 'matched_comparison.png')


def normalized_pick(scores):
    max_t = scores[:, 0].max() + 1e-9
    max_f = scores[:, 1].max() + 1e-9
    combined = scores[:, 0] / max_t + scores[:, 1] / max_f
    return int(np.argmin(combined))


def run(loader, graph, manager, route_idx=0, start_slot=96, reps=3):
    route = PREDEFINED_ROUTES[route_idx]
    start, end = route[0], route[1]

    rows = []
    for rep in range(reps):
        print(f'\n--- rep {rep + 1}/{reps} ---')
        cfg = OptimizationConfig()
        gen = PathGenerator(graph, loader, manager, cfg)
        ev = PathEvaluator(graph, loader, manager, cfg)
        opt = NSGAIIOptimizer(cfg, gen, ev)
        solutions, scores = opt.optimize(start, end, start_slot)
        if not solutions:
            print('  no static solutions, skipping rep')
            continue

        rules = {
            'best_time': int(np.argmin(scores[:, 0])),
            'best_fee': int(np.argmin(scores[:, 1])),
            'normalized': normalized_pick(scores),
        }

        router = DynamicRouter(graph, loader, manager, cfg,
                                DynamicRoutingConfig(), IncidentConfig())
        journey = router.simulate_journey(start, end, start_slot)

        for rule_name, idx in rules.items():
            static_t = float(scores[idx, 0])
            static_f = float(scores[idx, 1])
            rows.append({
                'rep': rep + 1,
                'rule': rule_name,
                'static_time': static_t,
                'static_fee': static_f,
                'dyn_time': journey.total_time_minutes,
                'dyn_fee': journey.total_fee,
                'delta_time': journey.total_time_minutes - static_t,
                'delta_fee': journey.total_fee - static_f,
            })
            print(f'  {rule_name:10}  static=({static_t:6.2f}, '
                  f'{static_f:5.2f})  dyn=({journey.total_time_minutes:6.2f}, '
                  f'{journey.total_fee:5.2f})')
    return rows


def make_figure(rows):
    rules = ['best_time', 'best_fee', 'normalized']
    static_t = [np.mean([r['static_time'] for r in rows if r['rule'] == k]) for k in rules]
    dyn_t = [np.mean([r['dyn_time'] for r in rows if r['rule'] == k]) for k in rules]
    static_f = [np.mean([r['static_fee'] for r in rows if r['rule'] == k]) for k in rules]
    dyn_f = [np.mean([r['dyn_fee'] for r in rows if r['rule'] == k]) for k in rules]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(rules))
    w = 0.36

    axes[0].bar(x - w / 2, static_t, w, label='Static plan', color='#1f77b4')
    axes[0].bar(x + w / 2, dyn_t, w, label='Dynamic execution', color='#ff7f0e')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rules)
    axes[0].set_ylabel('Trip time (min)')
    axes[0].set_title('Time')
    axes[0].legend(frameon=False, fontsize=9)
    for i, (s, d) in enumerate(zip(static_t, dyn_t)):
        axes[0].text(i - w / 2, s + 0.3, f'{s:.1f}', ha='center', fontsize=8)
        axes[0].text(i + w / 2, d + 0.3, f'{d:.1f}', ha='center', fontsize=8)

    axes[1].bar(x - w / 2, static_f, w, label='Static plan', color='#1f77b4')
    axes[1].bar(x + w / 2, dyn_f, w, label='Dynamic execution', color='#ff7f0e')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rules)
    axes[1].set_ylabel('Fee (\\$)')
    axes[1].set_title('Fee')
    axes[1].legend(frameon=False, fontsize=9)
    for i, (s, d) in enumerate(zip(static_f, dyn_f)):
        axes[1].text(i - w / 2, s + 0.03, f'{s:.2f}', ha='center', fontsize=8)
        axes[1].text(i + w / 2, d + 0.03, f'{d:.2f}', ha='center', fontsize=8)

    fig.suptitle('Static plan vs dynamic execution under matched plan-selection rules')
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()


def main():
    print('[load] graph + congestion + stations')
    loader = WinnipegDataLoader(DATA_DIR)
    graph = loader.load_graph()
    loader.load_congestion_data()
    stations = loader.load_charging_stations()
    loader.add_virtual_charging_nodes(stations)
    manager = ChargingStationManager(stations, 288)

    rows = run(loader, graph, manager, route_idx=0, start_slot=96, reps=3)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\n[csv] {OUT_CSV}')

    make_figure(rows)
    print(f'[fig] {OUT_FIG}')

    print('\n=== Matched comparison (mean across reps) ===')
    print(f'{"rule":<12} {"static_t":>9} {"dyn_t":>9} {"delta_t":>9} '
          f'{"static_f":>9} {"dyn_f":>9} {"delta_f":>9}')
    for rule in ['best_time', 'best_fee', 'normalized']:
        sub = [r for r in rows if r['rule'] == rule]
        if not sub:
            continue
        st = np.mean([r['static_time'] for r in sub])
        dt = np.mean([r['dyn_time'] for r in sub])
        sf = np.mean([r['static_fee'] for r in sub])
        df = np.mean([r['dyn_fee'] for r in sub])
        print(f'{rule:<12} {st:>9.2f} {dt:>9.2f} {dt - st:>+9.2f} '
              f'{sf:>9.2f} {df:>9.2f} {df - sf:>+9.2f}')


if __name__ == '__main__':
    main()
