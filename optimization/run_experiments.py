import os
import csv
import json
import time
import argparse

import matplotlib
matplotlib.use('Agg')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import (OptimizationConfig, DynamicRoutingConfig, IncidentConfig,
                    PREDEFINED_ROUTES, START_TIME_SLOT)
from data_processor import WinnipegDataLoader, ChargingStationManager
from route_planner import PathGenerator, PathEvaluator
from optimizer import NSGAIIOptimizer
from dynamic_router import DynamicRouter
from visualizer import WinnipegVisualizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUT_DIR = os.path.join(SCRIPT_DIR, 'results_experiments')
THESIS_FIG = os.path.join(SCRIPT_DIR, 'figures')


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def load_world():
    print('[load] graph + congestion + stations')
    loader = WinnipegDataLoader(DATA_DIR)
    graph = loader.load_graph()
    loader.load_congestion_data()
    stations = loader.load_charging_stations()
    loader.add_virtual_charging_nodes(stations)
    manager = ChargingStationManager(stations, 288)
    return loader, graph, manager, stations


def run_static(loader, graph, manager, config, start, end, t0):
    generator = PathGenerator(graph, loader, manager, config)
    evaluator = PathEvaluator(graph, loader, manager, config)
    optimizer = NSGAIIOptimizer(config, generator, evaluator)
    t = time.time()
    solutions, scores = optimizer.optimize(start, end, t0)
    elapsed = time.time() - t
    details = [evaluator.evaluate_path(s, t0) for s in solutions] if solutions else []
    return solutions, scores, details, elapsed, optimizer.convergence_history, evaluator


def run_dynamic(loader, graph, manager, config, dyn_cfg, inc_cfg, start, end, t0):
    router = DynamicRouter(graph, loader, manager, config, dyn_cfg, inc_cfg)
    t = time.time()
    journey = router.simulate_journey(start, end, t0)
    elapsed = time.time() - t
    return journey, elapsed


def astar_baseline(graph, evaluator, start, end, t0):
    try:
        path = nx.astar_path(graph, start, end, weight='length')
        return path, evaluator.evaluate_path(path, t0)
    except Exception as e:
        print(f'[astar] failed: {e}')
        return [start, end], evaluator.evaluate_path([start, end], t0)


def save_static_csv(solutions, scores, details, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'time_min', 'fee_usd', 'distance_km',
                    'num_charging_stops', 'battery_feasible', 'path_len'])
        for i, (s, d) in enumerate(zip(solutions, details)):
            w.writerow([i + 1, d.total_time_minutes, d.total_fee,
                        d.distance_km, d.num_charging_stops,
                        d.battery_feasible, len(s)])


def save_dynamic_csv(journey, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_min', 'fee_usd', 'distance_km',
                    'num_charging_stops', 'battery_feasible',
                    'num_replans', 'num_incidents', 'path_len'])
        w.writerow([journey.total_time_minutes, journey.total_fee,
                    journey.total_distance_km, journey.num_charging_stops,
                    journey.battery_feasible, len(journey.replan_events),
                    len(journey.incident_events), len(journey.actual_path)])
        w.writerow([])
        w.writerow(['trigger', 'node', 'slot', 'soc', 'path_idx'])
        for ev in journey.replan_events:
            w.writerow([ev.trigger, ev.node_id, ev.time_slot,
                        f'{ev.soc:.3f}', ev.path_index])


def trigger_breakdown(journey):
    counts = {}
    for ev in journey.replan_events:
        counts[ev.trigger] = counts.get(ev.trigger, 0) + 1
    return counts


def pareto_stats(scores, astar_scores):
    if scores is None or len(scores) == 0:
        return {}
    astar_t = astar_scores.total_time_minutes
    astar_f = astar_scores.total_fee
    dom = 0
    for i in range(len(scores)):
        t, fe = scores[i]
        if t <= astar_t and fe <= astar_f and (t < astar_t or fe < astar_f):
            dom += 1
    return {
        'pareto_size': int(len(scores)),
        'best_time': float(scores[:, 0].min()),
        'best_fee': float(scores[:, 1].min()),
        'mean_time': float(scores[:, 0].mean()),
        'mean_fee': float(scores[:, 1].mean()),
        'astar_time': float(astar_t),
        'astar_fee': float(astar_f),
        'dominating_astar': int(dom),
    }


def fig_pareto_real(scores, astar_scores, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    # find dominated solutions
    is_dom = np.zeros(len(scores), dtype=bool)
    for i in range(len(scores)):
        for j in range(len(scores)):
            if i == j:
                continue
            if (scores[j, 0] <= scores[i, 0] and scores[j, 1] <= scores[i, 1] and
                    (scores[j, 0] < scores[i, 0] or scores[j, 1] < scores[i, 1])):
                is_dom[i] = True
                break
    pareto = scores[~is_dom]
    if len(pareto):
        order = np.argsort(pareto[:, 0])
        ax.plot(pareto[order, 0], pareto[order, 1], '-o',
                color='#1f77b4', ms=6, label=f'Pareto front (n={len(pareto)})')
    if is_dom.any():
        ax.scatter(scores[is_dom, 0], scores[is_dom, 1],
                   s=24, c='#9aa0a6', alpha=0.6, label='dominated')
    ax.scatter([astar_scores.total_time_minutes], [astar_scores.total_fee],
               marker='*', s=200, c='#d62728', label='A* baseline',
               zorder=5, edgecolors='black', linewidth=0.6)
    ax.set_xlabel('Total time (min)')
    ax.set_ylabel('Total charging fee (\\$)')
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_convergence_real(history, out_path):
    if not history:
        return
    gens = [h['generation'] for h in history]
    bt = [h['best_time'] for h in history]
    bf = [h['best_fee'] for h in history]
    ps = [h['pareto_size'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].plot(gens, bt, color='#1f77b4')
    axes[0].set_title('Best time (min)')
    axes[0].set_xlabel('Generation')
    axes[1].plot(gens, bf, color='#2ca02c')
    axes[1].set_title('Best fee (\\$)')
    axes[1].set_xlabel('Generation')
    axes[2].plot(gens, ps, color='#9467bd')
    axes[2].set_title('Pareto front size')
    axes[2].set_xlabel('Generation')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_replan_timeline_real(journey, out_path):
    if not journey.replan_events:
        return
    colors = {'scheduled': '#1f77b4', 'intersection': '#2ca02c',
              'soc_threshold': '#d62728', 'post_charging': '#9467bd',
              'incident': '#ff7f0e'}
    fig, ax = plt.subplots(figsize=(8, 2.6))
    slots = [ev.time_slot for ev in journey.replan_events]
    base_slot = slots[0]
    minutes = [((s - base_slot) % 288) * 5 for s in slots]
    ax.hlines(0, -2, max(minutes) + 6 if minutes else 10, color='#aaa', lw=2)
    seen_labels = set()
    for ev, x in zip(journey.replan_events, minutes):
        c = colors.get(ev.trigger, '#888')
        label = ev.trigger if ev.trigger not in seen_labels else None
        ax.scatter([x], [0], s=140, c=c, zorder=3,
                   edgecolors='black', linewidth=0.5, label=label)
        seen_labels.add(ev.trigger)
    ax.set_yticks([])
    ax.set_xlabel('Time since first replan (min)')
    ax.set_title(f'Replan events ({len(journey.replan_events)} total)')
    ax.legend(frameon=False, loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_static_vs_dynamic_real(static_scores, journey, out_path):
    if static_scores is None or len(static_scores) == 0:
        return
    # pick the knee point from the static front
    knee_idx = int(np.argmin(static_scores[:, 0] / (static_scores[:, 0].max() + 1e-9)
                             + static_scores[:, 1] / (static_scores[:, 1].max() + 1e-9)))
    static_t = static_scores[knee_idx, 0]
    static_f = static_scores[knee_idx, 1]
    cats = ['Time (min)', 'Fee (\\$)', 'Stops', 'Replans']
    static_vals = [static_t, static_f, 0, 0]
    dyn_vals = [journey.total_time_minutes, journey.total_fee,
                journey.num_charging_stops, len(journey.replan_events)]
    x = np.arange(len(cats))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(x - w / 2, static_vals, w, label='Initial Route', color='#1f77b4')
    ax.bar(x + w / 2, dyn_vals, w, label='Re-Evaluated Route', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.legend(frameon=False)
    for i, (s, d) in enumerate(zip(static_vals, dyn_vals)):
        ax.text(i - w / 2, s + max(static_vals + dyn_vals) * 0.02,
                f'{s:.2f}' if isinstance(s, float) else f'{s}',
                ha='center', fontsize=8)
        ax.text(i + w / 2, d + max(static_vals + dyn_vals) * 0.02,
                f'{d:.2f}' if isinstance(d, float) else f'{d}',
                ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_soc_real(graph, loader, manager, config, path, t0, out_path):
    if len(path) < 2:
        return
    dist_acc = [0.0]
    soc = [config.initial_soc]
    cur_soc = config.initial_soc
    charge_at = []
    cur_slot = t0
    visited = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not graph.has_edge(u, v):
            continue
        edge = graph.edges[u, v]
        length_km = edge.get('length', 100) / 1000.0
        cur_soc -= length_km * config.battery_consumption_per_km / config.battery_capacity
        cur_soc = max(0.0, cur_soc)
        tt = loader.get_travel_time(u, v, cur_slot)
        cur_slot = (cur_slot + max(1, int(tt / 300))) % 288
        if cur_soc < config.charge_trigger_soc:
            for sid, st in manager.stations.items():
                if sid in visited:
                    continue
                if not config.allow_free_charging and st.electricity_price == 0:
                    continue
                if manager.get_available_ports(sid, cur_slot) <= 0:
                    continue
                visited.add(sid)
                charge_at.append(dist_acc[-1] + length_km)
                cur_soc = config.charge_target_soc
                break
        dist_acc.append(dist_acc[-1] + length_km)
        soc.append(cur_soc)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(dist_acc, soc, lw=2, color='#1f77b4', label='SOC trace')
    ax.axhline(config.charge_trigger_soc, ls='--', color='#d62728', alpha=0.6,
               label=f'charge trigger ({int(config.charge_trigger_soc*100)}\\%)')
    ax.axhline(config.min_soc, ls=':', color='black', alpha=0.6,
               label=f'hard min ({int(config.min_soc*100)}\\%)')
    for c in charge_at:
        ax.axvline(c, color='#2ca02c', alpha=0.25, lw=4)
    ax.set_xlabel('Distance travelled (km)')
    ax.set_ylabel('State of charge')
    ax.set_ylim(0, max(0.9, max(soc) + 0.05))
    ax.legend(frameon=False, loc='lower left')
    ax.set_title(f'SOC profile ({len(charge_at)} charging stops)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def experiment_configs():
    # each tuple: (name, opt_overrides, dyn_overrides, incident_overrides, route_idx, start_slot, repeats)
    return [
        ('baseline',         {},                          {}, {}, 0, 96, 3),
        ('midday',           {},                          {}, {}, 0, 144, 2),
        ('evening_rush',     {},                          {}, {}, 0, 204, 2),
        ('random_pop',       {'random_population': True}, {}, {}, 0, 96, 2),
        ('high_target_70',   {'charge_target_soc': 0.70}, {}, {}, 0, 96, 2),
        ('incidents_on',     {}, {}, {'enabled': True, 'random_incidents': True,
                                      'num_random_road_closures': 2,
                                      'num_random_station_failures': 1,
                                      'random_seed': 42}, 0, 96, 2),
    ]


def make_config(overrides):
    base = OptimizationConfig()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def make_dyn(overrides):
    base = DynamicRoutingConfig()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def make_inc(overrides):
    base = IncidentConfig()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Run only the baseline once for smoke test')
    args = parser.parse_args()

    ensure_dir(OUT_DIR)
    ensure_dir(THESIS_FIG)

    loader, graph, manager, stations = load_world()

    summary_rows = []
    summary_header = [
        'experiment', 'repeat', 'route_idx', 'start_slot',
        'astar_time_min', 'astar_fee', 'astar_distance_km',
        'pareto_size', 'pareto_best_time', 'pareto_best_fee',
        'pareto_mean_time', 'pareto_mean_fee', 'dominating_astar',
        'static_runtime_s',
        'dyn_time_min', 'dyn_fee', 'dyn_distance_km',
        'dyn_stops', 'dyn_replans', 'dyn_incidents', 'dyn_feasible',
        'trig_scheduled', 'trig_intersection', 'trig_soc',
        'trig_post_charging', 'trig_incident',
        'dyn_runtime_s',
    ]

    configs = experiment_configs()
    if args.quick:
        configs = [configs[0]]

    for exp_name, opt_over, dyn_over, inc_over, route_idx, start_slot, repeats in configs:
        exp_dir = ensure_dir(os.path.join(OUT_DIR, exp_name))
        for rep in range(repeats):
            print(f'\n========== {exp_name} (rep {rep + 1}/{repeats}) ==========')
            route = PREDEFINED_ROUTES[route_idx]
            start, end = route[0], route[1]

            cfg = make_config(opt_over)
            dyn_cfg = make_dyn(dyn_over)
            inc_cfg = make_inc(inc_over)

            sol, scores, details, st_time, history, evaluator = run_static(
                loader, graph, manager, cfg, start, end, start_slot)

            astar_path, astar_sc = astar_baseline(graph, evaluator, start, end, start_slot)
            pst = pareto_stats(scores, astar_sc)

            journey, dy_time = run_dynamic(loader, graph, manager, cfg,
                                            dyn_cfg, inc_cfg, start, end, start_slot)
            trig = trigger_breakdown(journey)

            row = [
                exp_name, rep + 1, route_idx, start_slot,
                astar_sc.total_time_minutes, astar_sc.total_fee, astar_sc.distance_km,
                pst.get('pareto_size', 0),
                pst.get('best_time', 0), pst.get('best_fee', 0),
                pst.get('mean_time', 0), pst.get('mean_fee', 0),
                pst.get('dominating_astar', 0),
                st_time,
                journey.total_time_minutes, journey.total_fee,
                journey.total_distance_km, journey.num_charging_stops,
                len(journey.replan_events), len(journey.incident_events),
                int(journey.battery_feasible),
                trig.get('scheduled', 0), trig.get('intersection', 0),
                trig.get('soc_threshold', 0), trig.get('post_charging', 0),
                trig.get('incident', 0),
                dy_time,
            ]
            summary_rows.append(row)

            save_static_csv(sol, scores, details,
                            os.path.join(exp_dir, f'static_rep{rep + 1}.csv'))
            save_dynamic_csv(journey,
                             os.path.join(exp_dir, f'dynamic_rep{rep + 1}.csv'))

            # save thesis figures on the first baseline run
            if exp_name == 'baseline' and rep == 0:
                fig_pareto_real(scores, astar_sc,
                                os.path.join(THESIS_FIG, 'pareto_front.png'),
                                f'Pareto front (route {route_idx}, slot {start_slot})')
                fig_convergence_real(history,
                                     os.path.join(THESIS_FIG, 'convergence.png'))
                fig_replan_timeline_real(journey,
                                         os.path.join(THESIS_FIG, 'replan_timeline.png'))
                fig_static_vs_dynamic_real(scores, journey,
                                           os.path.join(THESIS_FIG, 'static_vs_dynamic.png'))
                if sol:
                    best_time_idx = int(np.argmin(scores[:, 0]))
                    fig_soc_real(graph, loader, manager, cfg, sol[best_time_idx],
                                 start_slot,
                                 os.path.join(THESIS_FIG, 'soc_profile.png'))

    # write out summary csv and json
    summary_path = os.path.join(OUT_DIR, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(summary_header)
        for row in summary_rows:
            w.writerow(row)
    print(f'\n[summary] wrote {summary_path}')

    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump([dict(zip(summary_header, r)) for r in summary_rows], f, indent=2)

    print('\nDone.')


if __name__ == '__main__':
    main()
