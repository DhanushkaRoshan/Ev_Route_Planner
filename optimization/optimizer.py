import random
import copy
import numpy as np
import networkx as nx
from typing import List, Tuple

from config import OptimizationConfig, DominationResult, MutationType
from route_planner import PathGenerator, PathEvaluator


class NSGAIIOptimizer:

    def __init__(self, config, path_generator, path_evaluator):
        self.config = config
        self.path_generator = path_generator
        self.path_evaluator = path_evaluator
        self.convergence_history = []

    def dominates(self, scores1, scores2):
        if np.all(scores1 <= scores2) and np.any(scores1 < scores2):
            return DominationResult.FIRST_DOMINATES
        elif np.all(scores2 <= scores1) and np.any(scores2 < scores1):
            return DominationResult.SECOND_DOMINATES
        return DominationResult.NO_DOMINATION

    def tournament_selection(self, population, time_slot=0):
        winner = random.randint(0, len(population) - 1)
        for _ in range(self.config.tournament_size - 1):
            candidate = random.randint(0, len(population) - 1)
            winner_scores = self.path_evaluator.evaluate_path(population[winner], time_slot)
            candidate_scores = self.path_evaluator.evaluate_path(population[candidate], time_slot)
            if self.dominates(candidate_scores.as_array(),
                              winner_scores.as_array()) == DominationResult.FIRST_DOMINATES:
                winner = candidate
        return winner

    def remove_loops(self, path):
        loop_free = copy.deepcopy(path)
        curr_idx = 0
        while curr_idx < len(loop_free) - 1:
            curr_node = loop_free[curr_idx]
            while curr_node in loop_free[curr_idx + 1:]:
                dup_idx = loop_free[curr_idx + 1:].index(curr_node) + curr_idx + 1
                del loop_free[curr_idx + 1:dup_idx + 1]
            curr_idx = loop_free.index(curr_node) + 1
        return loop_free

    def optimize(self, start_node, end_node, start_time_slot=0):
        print("\nCreating initial population with intelligent path generation...")
        population = self.path_generator.create_initial_population(
            start_node, end_node, self.config.init_population_size, start_time_slot
        )
        self.convergence_history = []
        print(f"\n  {'Gen':>6}  {'Pareto':>6}  {'Best Time (min)':>15}  {'Best Fee ($)':>12}")
        print("  " + "-" * 46)

        for gen in range(self.config.num_generations):
            population = self._breed_population(population, start_node, end_node, start_time_slot)
            scores = self.path_evaluator.evaluate_population(population, start_time_slot)
            population = self._build_pareto_population(population, scores)

            fronts = self._fast_non_dominated_sort(scores)
            pareto_idx = fronts[0] if fronts else []
            best_time = float(scores[pareto_idx, 0].min()) if pareto_idx else float('inf')
            best_fee = float(scores[pareto_idx, 1].min()) if pareto_idx else float('inf')
            self.convergence_history.append({
                'generation': gen,
                'pareto_size': len(pareto_idx),
                'best_time': best_time,
                'best_fee': best_fee,
            })

            if gen % 10 == 0 or gen == self.config.num_generations - 1:
                print(f'  {gen:>3}/{self.config.num_generations:<4}'
                      f'  {len(pareto_idx):>6}'
                      f'  {best_time:>15.2f}'
                      f'  {best_fee:>12.2f}')

        # final eval
        final_scores = self.path_evaluator.evaluate_population(population, start_time_slot)
        fronts = self._fast_non_dominated_sort(final_scores)
        pareto_indices = fronts[0] if fronts else list(range(len(population)))

        final_solutions = [population[i] for i in pareto_indices]
        final_scores = final_scores[pareto_indices]
        return final_solutions, final_scores

    def _breed_population(self, population, start_node, end_node, time_slot=0):
        offspring = []
        while len(offspring) < self.config.init_population_size:
            p1_idx = self.tournament_selection(population, time_slot)
            p2_idx = self.tournament_selection(population, time_slot)
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            child = self._crossover(parent1, parent2)

            if self.config.mutation_type == MutationType.ENABLED:
                if random.random() < self.config.mutation_probability:
                    child = self._mutate(child)

            child = self.remove_loops(child)
            if self.path_generator.is_valid_path(child, start_node, end_node):
                offspring.append(child)

        return population + offspring

    def _crossover(self, parent1, parent2):
        common_nodes = set(parent1[1:-1]).intersection(set(parent2[1:-1]))
        if not common_nodes:
            return parent1.copy()
        crossover_node = random.choice(list(common_nodes))
        idx1 = parent1.index(crossover_node)
        idx2 = parent2.index(crossover_node)
        return parent1[:idx1 + 1] + parent2[idx2 + 1:]

    def _mutate(self, path):
        if len(path) <= 2:
            return path
        idx1 = random.randint(1, len(path) - 2)
        idx2 = random.randint(idx1, len(path) - 1)
        node1 = path[idx1]
        node2 = path[idx2]
        try:
            new_seg = nx.shortest_path(self.path_generator.graph, node1, node2, weight='length')
            return path[:idx1] + new_seg + path[idx2 + 1:]
        except nx.NetworkXNoPath:
            return path

    def _build_pareto_population(self, population, scores):
        # deduplicate
        seen = set()
        unique_indices = []
        for i, path in enumerate(population):
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                unique_indices.append(i)
        population = [population[i] for i in unique_indices]
        scores = scores[unique_indices]

        fronts = self._fast_non_dominated_sort(scores)
        next_pop = []
        front_idx = 0

        while len(next_pop) < self.config.init_population_size and front_idx < len(fronts):
            front = fronts[front_idx]
            if len(next_pop) + len(front) <= self.config.init_population_size:
                next_pop.extend([population[i] for i in front])
            else:
                remaining = self.config.init_population_size - len(next_pop)
                selected = random.sample(front, remaining)
                next_pop.extend([population[i] for i in selected])
            front_idx += 1

        return next_pop

    def _fast_non_dominated_sort(self, scores):
        pop_size = scores.shape[0]
        domination_count = [0] * pop_size
        dominated_solutions = [[] for _ in range(pop_size)]
        fronts = [[]]

        for i in range(pop_size):
            for j in range(pop_size):
                if i != j:
                    result = self.dominates(scores[i], scores[j])
                    if result == DominationResult.FIRST_DOMINATES:
                        dominated_solutions[i].append(j)
                    elif result == DominationResult.SECOND_DOMINATES:
                        domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        # remove empty last front
        if not fronts[-1]:
            fronts.pop()

        return fronts
