import os
import sys
import csv
import time
import inspect

import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, ROOT)

from config import OptimizationConfig, DynamicRoutingConfig, IncidentConfig, PREDEFINED_ROUTES
from data_processor import WinnipegDataLoader, ChargingStationManager
from dynamic_router import DynamicRouter

DATA_DIR = os.path.join(ROOT, 'data')
OUT_CSV = os.path.join(SCRIPT_DIR, 'intersection_noop.csv')


def instrument(router):
    log = []
    original = router._intersection_replan

    def wrapped(start, end, time_slot, current_soc):
        caller = inspect.currentframe().f_back
        before_plan = caller.f_locals.get('current_plan')
        plan_ptr = caller.f_locals.get('plan_ptr')
        before_next = None
        if before_plan and plan_ptr is not None and plan_ptr + 1 < len(before_plan):
            before_next = before_plan[plan_ptr + 1]

        t = time.time()
        new_plan = original(start, end, time_slot, current_soc)
        dt = time.time() - t

        after_next = new_plan[1] if new_plan and len(new_plan) > 1 else None
        log.append({
            'time_slot': time_slot,
            'soc': current_soc,
            'before_next': before_next,
            'after_next': after_next,
            'changed': bool(after_next != before_next and after_next is not None),
            'new_plan_len': len(new_plan) if new_plan else 0,
            'runtime_s': dt,
        })
        return new_plan

    router._intersection_replan = wrapped
    return log


def main():
    print('[load] graph + congestion + stations')
    loader = WinnipegDataLoader(DATA_DIR)
    graph = loader.load_graph()
    loader.load_congestion_data()
    stations = loader.load_charging_stations()
    loader.add_virtual_charging_nodes(stations)
    manager = ChargingStationManager(stations, 288)

    cfg = OptimizationConfig()
    dyn_cfg = DynamicRoutingConfig()
    inc_cfg = IncidentConfig()

    route = PREDEFINED_ROUTES[0]
    start, end = route[0], route[1]
    start_slot = 96

    print(f'[run] route 0, slot {start_slot}, default config')
    router = DynamicRouter(graph, loader, manager, cfg, dyn_cfg, inc_cfg)
    log = instrument(router)
    journey = router.simulate_journey(start, end, start_slot)

    n_total = len(log)
    n_changed = sum(1 for r in log if r['changed'])
    n_noop = n_total - n_changed
    total_runtime = sum(r['runtime_s'] for r in log)

    print('\n=== Intersection replan analysis ===')
    print(f'  Total intersection replans : {n_total}')
    print(f'  Plan-changing replans      : {n_changed} ({n_changed / max(n_total, 1) * 100:.1f}%)')
    print(f'  No-op replans              : {n_noop} ({n_noop / max(n_total, 1) * 100:.1f}%)')
    print(f'  Total compute spent        : {total_runtime:.2f} s')
    print(f'  Mean per replan            : {total_runtime / max(n_total, 1):.3f} s')
    print(f'  Compute wasted on no-ops   : {total_runtime * n_noop / max(n_total, 1):.2f} s')

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()) if log else [])
        w.writeheader()
        for row in log:
            w.writerow(row)
    print(f'\n[csv] {OUT_CSV}')

    summary_path = os.path.join(SCRIPT_DIR, 'intersection_noop_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['total', 'changed', 'noop', 'pct_noop',
                    'total_runtime_s', 'wasted_runtime_s'])
        w.writerow([n_total, n_changed, n_noop,
                    n_noop / max(n_total, 1) * 100,
                    total_runtime,
                    total_runtime * n_noop / max(n_total, 1)])
    print(f'[csv] {summary_path}')


if __name__ == '__main__':
    main()
