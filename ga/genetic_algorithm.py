import random
import numpy as np
from copy import deepcopy
from config import GA_POP, GA_GENS, GA_CX_PROB, GA_MUT_PROB, GA_LAMBDA1, GA_LAMBDA2, RANDOM_SEED

def route_cost(route, tm):
    total = 0.0
    for i in range(len(route)-1):
        total += tm[route[i]][route[i+1]]
    return total

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class GeneticAlgorithmRouter:

    def __init__(self, time_matrix,
                 pop_size=GA_POP, generations=GA_GENS,
                 cx_prob=GA_CX_PROB, mut_prob=GA_MUT_PROB,
                 lambda1=GA_LAMBDA1, lambda2=GA_LAMBDA2):
        self.tm          = np.array(time_matrix, dtype=np.float32)
        self.pop_size    = pop_size
        self.generations = generations
        self.cx_prob     = cx_prob
        self.mut_prob    = mut_prob
        self.lambda1     = lambda1
        self.lambda2     = lambda2

    def _fitness(self, route):
        """Fitness = total travel time (lower is better)."""
        total = 0.0

        for k in range(len(route) - 1):
            total += self.tm[route[k]][route[k + 1]]

        return total

    @staticmethod
    def _ox_crossover(p1, p2):
        """Order crossover (OX1) — guaranteed to produce valid permutation."""
        size = len(p1)
        if size <= 2:
            return p1[:]
        a, b = sorted(random.sample(range(size), 2))
        child    = [None] * size
        child[a:b] = p1[a:b]
        fill_vals = [x for x in p2 if x not in child[a:b]]
        ptr = 0
        for k in range(size):
            if child[k] is None:
                child[k] = fill_vals[ptr]
                ptr += 1
        return child

    @staticmethod
    def _mutate_swap(route):
        r = route[:]

        # Need at least 3 nodes to swap middle points
        if len(r) <= 3:
            return r

        i, j = sorted(random.sample(range(1, len(r) - 1), 2))
        r[i:j] = reversed(r[i:j])

        return r

    def run(self, required_stops, top_k=10):
        """
        required_stops : list like [start, mid1, mid2, ..., end]
                         start and end are kept fixed; middles are evolved.
        Returns (top_k_routes, convergence_history)
        where each item in top_k_routes is (score, route_list).
        """
        start  = required_stops[0]
        end    = required_stops[-1]
        middle = required_stops[1:-1]

        if len(middle) == 0:
            # Trivial: direct route
            route = [start, end]
            score = self._fitness(route)
            return [(score, route)] * min(top_k, 1), [score]

        pop     = [[start] + random.sample(middle, len(middle)) + [end]
                   for _ in range(self.pop_size)]
        history = []

        best_overall = float("inf")
        no_improve_count = 0

        for gen in range(self.generations):
            scored    = sorted([(self._fitness(r), r) for r in pop],
                               key=lambda x: x[0])
            
            current_best = scored[0][0]
            history.append(current_best)

            # Early stopping check
            if current_best < best_overall:
                best_overall = current_best
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= 20:
                # Pad history so plots still look consistent or just let it be shorter
                # history.extend([best_overall] * (self.generations - gen - 1)) # Optional
                break

            elite = [r for _, r in scored[:10]]
            
            offspring = []
            while len(offspring) < self.pop_size - len(elite):
                # Tournament selection (size 3)
                p1 = min(random.sample(scored, 3), key=lambda x: x[0])[1]
                p2 = min(random.sample(scored, 3), key=lambda x: x[0])[1]
                
                child  = (self._ox_crossover(p1, p2)
                          if random.random() < self.cx_prob else p1[:])
                if random.random() < self.mut_prob:
                    child = self._mutate_swap(child)
                offspring.append(child)
                
            pop = elite + offspring

        final  = sorted([(self._fitness(r), r) for r in pop],
                        key=lambda x: x[0])
        seen, top = set(), []
        for score, route in final:
            key = tuple(route)
            if key not in seen:
                seen.add(key)
                top.append((score, route))
            if len(top) == top_k:
                break

        print(f"[GA] Best fitness: {top[0][0]:.2f} min over {self.generations} gens")
        # ── Local optimization (2-opt refinement) ──
        for idx in range(min(3, len(top))):
            best = top[idx][1]
            best_cost = route_cost(best, self.tm)
            improved = True
            
            while improved:
                improved = False
                for i in range(1, len(best)-1):
                    for j in range(i+2, len(best)):
                        new = best[:]
                        new[i:j] = reversed(new[i:j])
                        new_cost = route_cost(new, self.tm)
                        if new_cost < best_cost:
                            best = new
                            best_cost = new_cost
                            improved = True
                            
            top[idx] = (best_cost, best)
            
        # Re-sort after local optimization
        top = sorted(top, key=lambda x: x[0])
        return top, history