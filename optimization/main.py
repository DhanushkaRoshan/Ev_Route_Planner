import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import OptimizationConfig, DynamicRoutingConfig, IncidentConfig, PREDEFINED_ROUTES, ACTIVE_ROUTE_INDEX, START_TIME_SLOT
from data_processor import WinnipegDataLoader, ChargingStationManager
from route_planner import PathGenerator, PathEvaluator
from optimizer import NSGAIIOptimizer
from dynamic_router import DynamicRouter
from visualizer import WinnipegVisualizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUT_DIR = os.path.join(SCRIPT_DIR, 'results_dynamic')


def route_out_dir(start, end):
    folder = os.path.join(OUT_DIR, f"{start}_{end}")
    os.makedirs(folder, exist_ok=True)
    return folder


def out(folder, filename):
    return os.path.join(folder, filename)


def analyze_charging_usage(path_details, charging_manager):
    total_paths = len(path_details)
    paths_with_charging = sum(1 for ps in path_details if ps.num_charging_stops > 0)
    all_station_ids = set(sid for ps in path_details for sid in ps.used_station_ids)

    print(f"\nTotal Pareto-optimal paths: {total_paths}")
    print(f"Paths using charging stations: {paths_with_charging} ({paths_with_charging / total_paths * 100:.1f}%)")
    print(f"Unique charging stations used: {len(all_station_ids)}")

    if all_station_ids:
        print(f"\nCharging stations in optimal routes:")
        for sid in list(all_station_ids)[:5]:
            station = charging_manager.get_station(sid)
            if station:
                print(f"  - {station.title} (ID: {sid}, {station.charging_power}kW, ${station.electricity_price}/kWh)")


def save_results_to_csv(solutions, scores, path_evaluator, start_time_slot, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['solution_id', 'total_time_min', 'total_fee_usd',
                         'distance_km', 'num_charging_stops', 'battery_feasible',
                         'path_length', 'path_nodes'])
        for i, (solution, scores_row) in enumerate(zip(solutions, scores)):
            path_detail = path_evaluator.evaluate_path(solution, start_time_slot)
            writer.writerow([
                i + 1,
                scores_row[0],
                scores_row[1],
                path_detail.distance_km,
                path_detail.num_charging_stops,
                path_detail.battery_feasible,
                len(solution),
                ' -> '.join(map(str, solution))
            ])


def save_dynamic_result_to_csv(journey_result, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['total_time_min', 'total_fee_usd', 'distance_km',
                         'num_charging_stops', 'battery_feasible',
                         'num_replan_events', 'path_length'])
        writer.writerow([
            journey_result.total_time_minutes,
            journey_result.total_fee,
            journey_result.total_distance_km,
            journey_result.num_charging_stops,
            journey_result.battery_feasible,
            len(journey_result.replan_events),
            len(journey_result.actual_path)
        ])
        writer.writerow([])
        writer.writerow(['Replan Events'])
        writer.writerow(['index', 'trigger', 'node_id', 'time_slot', 'soc', 'path_index'])
        for i, ev in enumerate(journey_result.replan_events):
            writer.writerow([i + 1, ev.trigger, ev.node_id,
                             ev.time_slot, f"{ev.soc:.3f}", ev.path_index])


def compare_with_baseline(astar_scores, pareto_scores):
    astar_time = astar_scores.total_time_minutes
    astar_fee = astar_scores.total_fee

    print(f"\nA* Baseline:")
    print(f"  Time: {astar_time:.2f} minutes")
    print(f"  Fee: ${astar_fee:.2f}")

    dominating_solutions = 0
    for i in range(len(pareto_scores)):
        if (pareto_scores[i, 0] <= astar_time and pareto_scores[i, 1] <= astar_fee and
                (pareto_scores[i, 0] < astar_time or pareto_scores[i, 1] < astar_fee)):
            dominating_solutions += 1

    print(f"\n{dominating_solutions} out of {len(pareto_scores)} Pareto solutions dominate A*")

    if len(pareto_scores) > 0:
        avg_time = pareto_scores[:, 0].mean()
        avg_fee = pareto_scores[:, 1].mean()

        time_improvement = ((astar_time - avg_time) / astar_time) * 100 if astar_time > 0 else 0
        fee_str = (f"{((astar_fee - avg_fee) / astar_fee) * 100:+.1f}%"
                   if astar_fee > 0 else "N/A (A* baseline is free)")
        print(f"\nAverage improvements over A*:")
        print(f"  Time: {time_improvement:+.1f}%")
        print(f"  Fee: {fee_str}")

        best_time_idx = np.argmin(pareto_scores[:, 0])
        best_fee_idx = np.argmin(pareto_scores[:, 1])

        best_time_pct = (f"({((astar_time - pareto_scores[best_time_idx, 0]) / astar_time * 100):+.1f}%)"
                         if astar_time > 0 else "")
        best_fee_pct = (f"({((astar_fee - pareto_scores[best_fee_idx, 1]) / astar_fee * 100):+.1f}%)"
                        if astar_fee > 0 else "")

        print(f"\nBest Time Solution:")
        print(f"  Time: {pareto_scores[best_time_idx, 0]:.2f} min {best_time_pct}")
        print(f"  Fee: ${pareto_scores[best_time_idx, 1]:.2f}")

        print(f"\nBest Fee Solution:")
        print(f"  Time: {pareto_scores[best_fee_idx, 0]:.2f} min")
        print(f"  Fee: ${pareto_scores[best_fee_idx, 1]:.2f} {best_fee_pct}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # load everything
    data_loader = WinnipegDataLoader(DATA_DIR)
    graph = data_loader.load_graph()
    data_loader.load_congestion_data()
    charging_stations_dict = data_loader.load_charging_stations()
    data_loader.add_virtual_charging_nodes(charging_stations_dict)

    config = OptimizationConfig()
    charging_manager = ChargingStationManager(charging_stations_dict, 288)
    path_generator = PathGenerator(graph, data_loader=data_loader,
                                   charging_manager=charging_manager, config=config)
    path_evaluator = PathEvaluator(graph, data_loader, charging_manager, config)
    optimizer = NSGAIIOptimizer(config, path_generator, path_evaluator)

    real_nodes = [n for n in graph.nodes() if not graph.nodes[n].get('is_virtual', False)]
    print(f"\nTotal nodes: {len(graph.nodes())} ({len(real_nodes)} real, "
          f"{len(graph.nodes()) - len(real_nodes)} virtual)")
    print(f"EV Stations: {len(charging_stations_dict)}")

    route = PREDEFINED_ROUTES[ACTIVE_ROUTE_INDEX]
    start_node, end_node, route_desc, route_dist_km = route[0], route[1], route[2], route[3]
    start_time_slot = START_TIME_SLOT
    start_hour = (start_time_slot * 5) / 60
    print(f"\nRoute [{ACTIVE_ROUTE_INDEX}]: {route_desc}")
    print(f"  {start_node} -> {end_node}  (~{route_dist_km:.1f} km)")
    print(f"Start Time: {start_hour:.0f}:00 (Time Slot {start_time_slot})")

    run_dir = route_out_dir(start_node, end_node)
    print(f"Saving outputs to: {run_dir}")

    # get A* baseline first
    print("\nComputing A* baseline...")
    try:
        astar_path = nx.astar_path(graph, start_node, end_node, weight='length')
        astar_scores = path_evaluator.evaluate_path(astar_path, start_time_slot)
        print(f"A* path length: {len(astar_path)} nodes")
        print(f"  Time: {astar_scores.total_time_minutes:.2f} minutes")
        print(f"  Fee: ${astar_scores.total_fee:.2f}")
        print(f"  Distance: {astar_scores.distance_km:.2f} km")
        print(f"  Charging Stops: {astar_scores.num_charging_stops}")
        print(f"  Battery Feasible: {astar_scores.battery_feasible}")
    except Exception as e:
        print(f"Could not compute A* baseline: {e}")
        astar_path = [start_node, end_node]
        astar_scores = path_evaluator.evaluate_path(astar_path, start_time_slot)

    # run NSGA-II
    print("\n=== Static NSGA-II Optimisation ===")
    final_solutions, final_scores = optimizer.optimize(start_node, end_node, start_time_slot)

    if not final_solutions:
        print("No solutions found!")
        return None

    print(f"\nFound {len(final_solutions)} Pareto-optimal solutions")
    path_details = [path_evaluator.evaluate_path(s, start_time_slot) for s in final_solutions]

    for i, (solution, ps) in enumerate(zip(final_solutions[:5], path_details[:5])):
        print(f"\n--- Solution {i + 1} ---")
        print(f"  Path: {len(solution)} nodes | Charging Stops: {ps.num_charging_stops}")
        print(f"  Time: {ps.total_time_minutes:.2f} min | Fee: ${ps.total_fee:.2f}")
        print(f"  Distance: {ps.distance_km:.2f} km | Battery Feasible: {'Yes' if ps.battery_feasible else 'No'}")
        if ps.used_station_ids:
            for sid in ps.used_station_ids:
                st = charging_manager.get_station(sid)
                if st:
                    print(f"  Charged at: {st.title} ({st.charging_power}kW, ${st.electricity_price}/kWh)")

    analyze_charging_usage(path_details, charging_manager)
    compare_with_baseline(astar_scores, final_scores)

    save_results_to_csv(final_solutions, final_scores, path_evaluator,
                        start_time_slot, out(run_dir, 'pareto_solutions.csv'))

    best_time_idx = int(np.argmin(final_scores[:, 0]))
    best_fee_idx = int(np.argmin(final_scores[:, 1]))

    print(f"\nBest by time : {path_details[best_time_idx].total_time_minutes:.2f} min, "
          f"${path_details[best_time_idx].total_fee:.2f} fee")
    print(f"Best by fee  : {path_details[best_fee_idx].total_time_minutes:.2f} min, "
          f"${path_details[best_fee_idx].total_fee:.2f} fee")

    # now run the dynamic router
    print("\n=== Dynamic Rolling-Horizon Optimisation ===")
    dynamic_config = DynamicRoutingConfig()
    incident_config = IncidentConfig()
    dynamic_router = DynamicRouter(graph, data_loader, charging_manager, config,
                                   dynamic_config, incident_config)
    journey = dynamic_router.simulate_journey(start_node, end_node, start_time_slot)

    print("\nDynamic Journey Result:")
    print(journey.summary())

    save_dynamic_result_to_csv(journey, out(run_dir, 'dynamic_journey.csv'))

    # generate all the plots
    pareto_fig = WinnipegVisualizer.plot_pareto_front(
        final_scores,
        title=f"Start Time: {start_hour:.0f}:00",
        start_node=start_node, end_node=end_node
    )
    plt.savefig(out(run_dir, 'pareto_front.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pareto_front.png")

    fig_3d = WinnipegVisualizer.plot_3d_pareto_front(
        path_details,
        title="NSGA-II Optimisation", start_node=start_node, end_node=end_node
    )
    plt.savefig(out(run_dir, 'pareto_front_3d.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pareto_front_3d.png")

    para_fig = WinnipegVisualizer.plot_parallel_coordinates(
        path_details, title="Winnipeg EV Route Objectives"
    )
    para_fig.write_image(out(run_dir, 'parallel_coordinates.png'))
    para_fig.show()
    print(f"Saved parallel_coordinates.png")

    selected = [final_solutions[best_time_idx], final_solutions[best_fee_idx]]
    selected_details = [path_details[best_time_idx], path_details[best_fee_idx]]
    labels = ['Fastest Time', 'Lowest Fee']

    if astar_path:
        selected.append(astar_path)
        selected_details.append(astar_scores)
        labels.append('A* Baseline')

    route_fig = WinnipegVisualizer.plot_route_comparison(
        graph, selected, selected_details, charging_stations_dict,
        charging_manager=charging_manager, labels=labels
    )
    route_fig.write_html(out(run_dir, 'route_comparison.html'))
    route_fig.show()
    print(f"Saved route_comparison.html")

    summary_txt = WinnipegVisualizer.create_route_summary(
        graph, final_solutions[best_time_idx],
        path_details[best_time_idx], charging_manager, route_name="Best Time Route"
    )
    print(summary_txt)
    with open(out(run_dir, 'best_time_route_summary.txt'), 'w') as f:
        f.write(summary_txt)
    print(f"Saved best_time_route_summary.txt")

    WinnipegVisualizer.plot_route_networkx(
        graph, selected, start_node, end_node, labels=labels,
        path_details=selected_details, charging_manager=charging_manager,
        output_path=out(run_dir, 'route_networkx.png')
    )
    plt.show()
    print(f"Saved route_networkx.png")

    conv_fig = WinnipegVisualizer.plot_convergence(optimizer.convergence_history)
    if conv_fig:
        plt.savefig(out(run_dir, 'convergence.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved convergence.png")

    soc_fig = WinnipegVisualizer.plot_soc_profile(
        graph, selected, selected_details,
        data_loader, config, charging_manager,
        labels=labels, start_time_slot=start_time_slot
    )
    plt.savefig(out(run_dir, 'soc_profile.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved soc_profile.png")

    station_fig = WinnipegVisualizer.plot_station_usage_analysis(
        selected_details, charging_manager, labels=labels
    )
    if station_fig:
        plt.savefig(out(run_dir, 'station_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved station_analysis.png")

    net_fig = WinnipegVisualizer.plot_network(
        graph, charging_stations_dict, title="Winnipeg Road Network & EV Stations"
    )
    net_fig.write_html(out(OUT_DIR, 'network.html'))
    net_fig.show()
    print(f"Saved network.html")

    dyn_map = WinnipegVisualizer.plot_dynamic_journey(
        graph, journey, charging_stations_dict, charging_manager
    )
    dyn_map.write_html(out(run_dir, 'dynamic_journey_map.html'))
    dyn_map.show()
    print(f"Saved dynamic_journey_map.html")

    anim_fig = WinnipegVisualizer.plot_journey_animation(
        graph, journey, config, data_loader,
        charging_manager=charging_manager,
        start_time_slot=start_time_slot,
        frame_step=2
    )
    if anim_fig:
        anim_fig.write_html(out(run_dir, 'journey_animation.html'))
        anim_fig.show()
        print(f"Saved journey_animation.html")

    replan_fig = WinnipegVisualizer.plot_replan_timeline(journey)
    if replan_fig:
        plt.savefig(out(run_dir, 'replan_timeline.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved replan_timeline.png")

    compare_fig = WinnipegVisualizer.plot_static_vs_dynamic(final_scores, journey)
    plt.savefig(out(run_dir, 'static_vs_dynamic.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved static_vs_dynamic.png")

    print("\nDone!")
    return final_solutions, final_scores, journey


if __name__ == '__main__':
    main()
