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

from run_experiments import (
    load_world, run_static, run_dynamic, astar_baseline,
    pareto_stats, trigger_breakdown,
    fig_pareto_real, fig_static_vs_dynamic_real, fig_soc_real,
    fig_convergence_real,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'results_routes')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures_routes')


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def save_static_csv(solutions, details, path):
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


def knee_point(scores):
    # find the "knee" - normalized sum closest to zero
    t = scores[:, 0] / (scores[:, 0].max() + 1e-9)
    f = scores[:, 1] / (scores[:, 1].max() + 1e-9)
    idx = int(np.argmin(t + f))
    return scores[idx, 0], scores[idx, 1]


def fig_routes_time_fee(rows, out_path):
    labels = [f"R{r['route_idx']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.27

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    st_bt = [r['pareto_best_time'] for r in rows]
    dyn_t = [r['dyn_time_min'] for r in rows]
    astar_t = [r['astar_time_min'] for r in rows]
    ax1.bar(x - w, st_bt, w, label='Static best-time', color='#1f77b4')
    ax1.bar(x, dyn_t, w, label='Dynamic', color='#ff7f0e')
    ax1.bar(x + w, astar_t, w, label='A* baseline', color='#9aa0a6')
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel('Total time (min)')
    ax1.set_title('Travel time by route')
    ax1.legend(frameon=False, fontsize=8)

    st_bf = [r['pareto_best_fee'] for r in rows]
    dyn_f = [r['dyn_fee'] for r in rows]
    ax2.bar(x - w / 2, st_bf, w, label='Static best-fee', color='#2ca02c')
    ax2.bar(x + w / 2, dyn_f, w, label='Dynamic', color='#ff7f0e')
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel('Charging fee (\\$)')
    ax2.set_title('Charging fee by route')
    ax2.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_pareto_scatter(scores, astar_scores, out_path, title):
    # check which solutions are dominated
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
    fig, ax = plt.subplots(figsize=(6, 4.2))
    if len(pareto):
        ax.scatter(pareto[:, 0], pareto[:, 1], s=55, color='#1f77b4',
                   edgecolors='black', linewidth=0.5, zorder=4,
                   label=f'Pareto front (n={len(pareto)})')
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


def fig_routes_fronts(all_fronts, out_path):
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for k, (idx, scores) in enumerate(all_fronts):
        if scores is None or len(scores) == 0:
            continue
        ax.scatter(scores[:, 0], scores[:, 1], s=45, marker='o',
                   color=colors[idx % 10], edgecolors='black', linewidth=0.4,
                   label=f'Route {idx}')
    ax.set_xlabel('Total time (min)')
    ax.set_ylabel('Total charging fee (\\$)')
    ax.set_title('Pareto fronts across all routes')
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_routes_replans(rows, out_path):
    labels = [f"R{r['route_idx']}" for r in rows]
    x = np.arange(len(rows))
    inter = [r['trig_intersection'] for r in rows]
    sched = [r['trig_scheduled'] for r in rows]
    post = [r['trig_post_charging'] for r in rows]
    soc = [r['trig_soc'] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x, inter, label='intersection', color='#2ca02c')
    bottom = np.array(inter, dtype=float)
    ax.bar(x, sched, bottom=bottom, label='scheduled', color='#1f77b4')
    bottom += np.array(sched, dtype=float)
    ax.bar(x, post, bottom=bottom, label='post_charging', color='#9467bd')
    bottom += np.array(post, dtype=float)
    ax.bar(x, soc, bottom=bottom, label='soc_threshold', color='#d62728')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Replan events')
    ax.set_title('Replan triggers by route')
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Run only the first route (smoke test)')
    args = parser.parse_args()

    ensure_dir(OUT_DIR)
    ensure_dir(FIG_DIR)

    loader, graph, manager, stations = load_world()

    start_slot = START_TIME_SLOT
    routes = list(enumerate(PREDEFINED_ROUTES))
    if args.quick:
        routes = routes[:1]

    summary_header = [
        'route_idx', 'start_node', 'end_node', 'description', 'approx_km',
        'start_slot',
        'astar_time_min', 'astar_fee', 'astar_distance_km',
        'pareto_size', 'pareto_best_time', 'pareto_best_fee',
        'pareto_mean_time', 'pareto_mean_fee', 'dominating_astar',
        'knee_time', 'knee_fee',
        'static_runtime_s',
        'dyn_time_min', 'dyn_fee', 'dyn_distance_km',
        'dyn_stops', 'dyn_replans', 'dyn_incidents', 'dyn_feasible',
        'trig_scheduled', 'trig_intersection', 'trig_soc',
        'trig_post_charging', 'trig_incident',
        'dyn_runtime_s',
    ]
    summary_rows = []
    all_fronts = []

    for route_idx, route in routes:
        start, end, desc, approx_km = route[0], route[1], route[2], route[3]
        print(f'\n========== Route {route_idx}: {desc} '
              f'({start} -> {end}, ~{approx_km} km) ==========')
        rdir = ensure_dir(os.path.join(OUT_DIR, f'{route_idx}_{start}_{end}'))

        cfg = OptimizationConfig()
        dyn_cfg = DynamicRoutingConfig()
        inc_cfg = IncidentConfig()

        sol, scores, details, st_time, history, evaluator = run_static(
            loader, graph, manager, cfg, start, end, start_slot)
        astar_path, astar_sc = astar_baseline(graph, evaluator, start, end, start_slot)
        pst = pareto_stats(scores, astar_sc)

        journey, dy_time = run_dynamic(loader, graph, manager, cfg,
                                       dyn_cfg, inc_cfg, start, end, start_slot)
        trig = trigger_breakdown(journey)

        knee_t, knee_f = (knee_point(scores) if scores is not None and len(scores)
                          else (0.0, 0.0))

        row = {
            'route_idx': route_idx, 'start_node': start, 'end_node': end,
            'description': desc, 'approx_km': approx_km, 'start_slot': start_slot,
            'astar_time_min': astar_sc.total_time_minutes,
            'astar_fee': astar_sc.total_fee,
            'astar_distance_km': astar_sc.distance_km,
            'pareto_size': pst.get('pareto_size', 0),
            'pareto_best_time': pst.get('best_time', 0),
            'pareto_best_fee': pst.get('best_fee', 0),
            'pareto_mean_time': pst.get('mean_time', 0),
            'pareto_mean_fee': pst.get('mean_fee', 0),
            'dominating_astar': pst.get('dominating_astar', 0),
            'knee_time': knee_t, 'knee_fee': knee_f,
            'static_runtime_s': st_time,
            'dyn_time_min': journey.total_time_minutes,
            'dyn_fee': journey.total_fee,
            'dyn_distance_km': journey.total_distance_km,
            'dyn_stops': journey.num_charging_stops,
            'dyn_replans': len(journey.replan_events),
            'dyn_incidents': len(journey.incident_events),
            'dyn_feasible': int(journey.battery_feasible),
            'trig_scheduled': trig.get('scheduled', 0),
            'trig_intersection': trig.get('intersection', 0),
            'trig_soc': trig.get('soc_threshold', 0),
            'trig_post_charging': trig.get('post_charging', 0),
            'trig_incident': trig.get('incident', 0),
            'dyn_runtime_s': dy_time,
        }
        summary_rows.append(row)
        all_fronts.append((route_idx, scores))

        save_static_csv(sol, details, os.path.join(rdir, 'static.csv'))
        save_dynamic_csv(journey, os.path.join(rdir, 'dynamic.csv'))

        # generate per-route figures
        if scores is not None and len(scores):
            fig_pareto_scatter(scores, astar_sc,
                               os.path.join(FIG_DIR, f'pareto_r{route_idx}.png'),
                               f'Route {route_idx}: {desc}')
            fig_static_vs_dynamic_real(
                scores, journey,
                os.path.join(FIG_DIR, f'static_vs_dynamic_r{route_idx}.png'))
            best_time_idx = int(np.argmin(scores[:, 0]))
            best_fee_idx = int(np.argmin(scores[:, 1]))
            fig_soc_real(graph, loader, manager, cfg, sol[best_time_idx],
                         start_slot, os.path.join(FIG_DIR, f'soc_r{route_idx}.png'))

            fig_convergence_real(history,
                                 os.path.join(FIG_DIR, f'conv_r{route_idx}.png'))

            routes_to_draw = [astar_path,
                              sol[best_time_idx],
                              sol[best_fee_idx],
                              journey.actual_path]
            route_details = [astar_sc,
                             details[best_time_idx],
                             details[best_fee_idx],
                             journey]
            route_labels = ['A* baseline', 'Initial best time',
                            'Initial best fee', 'Re-Evaluated Route']
            fig_map = WinnipegVisualizer.plot_route_networkx(
                graph, routes_to_draw, start, end,
                labels=route_labels, path_details=route_details,
                charging_manager=manager,
                output_path=os.path.join(FIG_DIR, f'map_r{route_idx}.png'))
            plt.close(fig_map)

        print(f'  static: {pst.get("pareto_size",0)} sols, best_time '
              f'{pst.get("best_time",0):.2f} min, best_fee ${pst.get("best_fee",0):.3f}')
        print(f'  dynamic: {journey.total_time_minutes:.2f} min, '
              f'${journey.total_fee:.2f}, {len(journey.replan_events)} replans')

    # cross-route summary figures
    if len(summary_rows) > 1:
        fig_routes_time_fee(summary_rows, os.path.join(FIG_DIR, 'routes_time_fee.png'))
        fig_routes_fronts(all_fronts, os.path.join(FIG_DIR, 'routes_fronts.png'))
        fig_routes_replans(summary_rows, os.path.join(FIG_DIR, 'routes_replans.png'))

    summary_path = os.path.join(OUT_DIR, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(summary_header)
        for r in summary_rows:
            w.writerow([r[k] for k in summary_header])
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary_rows, f, indent=2)
    print(f'\n[summary] wrote {summary_path}')
    print('Done.')


if __name__ == '__main__':
    main()
