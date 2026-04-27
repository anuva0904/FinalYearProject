import heapq
import time as _time
import numpy as np
import os

from config import STOPS, N, RESULTS_DIR
from lstm.predict import build_time_matrix
from ga.genetic_algorithm import GeneticAlgorithmRouter


def dijkstra(time_matrix, start, end):
    """Classic Dijkstra on the N×N time matrix. Returns (path, cost)."""
    INF  = float("inf")
    dist = [INF] * N
    prev = [-1]  * N
    dist[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in range(N):
            if v == u:
                continue
            nd = dist[u] + float(time_matrix[u][v])
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    path, cur = [], end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    return path[::-1], dist[end]


from itertools import permutations

def brute_force_route(tm, start, waypoints, end):
    best_cost = float("inf")
    best_route = None
    for perm in permutations(waypoints):
        route = [start] + list(perm) + [end]
        cost = sum(tm[route[i]][route[i+1]] for i in range(len(route)-1))
        if cost < best_cost:
            best_cost = cost
            best_route = route
    return best_cost, best_route

class SmartRouteOptimizer:
    """
    Orchestrates LSTM → GA (with Brute Force fallback for small N) to produce an optimised route.
    Call optimize(start, end, waypoints) to get a result dict.
    """

    def __init__(self, hour=8, is_fog=0, is_rain=0):
        print("[Optimizer] Building LSTM time matrix...")
        self.tm   = build_time_matrix(hour, is_fog, is_rain)
        self.hour = hour
        self.all_stop_ids = list(range(N))

        self.ga = GeneticAlgorithmRouter(self.tm)

    def optimize(self, start, end, waypoints=None):
        """
        Returns a dict with:
          route, route_names, travel_time_min, dijkstra_time,
          improvement_pct, runtime_ms, ga_history
        """
        waypoints     = waypoints or []
        required      = [start] + [w for w in waypoints
                                    if w != start and w != end] + [end]
        # Deduplicate while preserving order
        seen_r, deduped = set(), []
        for s in required:
            if s not in seen_r:
                seen_r.add(s); deduped.append(s)
        required = deduped

        t0 = _time.perf_counter()

        # ── Smart Routing Logic ────────────────────
        if len(required) <= 2:
            ga_history = [0]
            best_route = required
            best_time = sum(self.tm[required[i]][required[i+1]] for i in range(len(required)-1)) if len(required) > 1 else 0.0
        elif len(required) <= 6:  # Up to 4 waypoints = 24 permutations (instant)
            ga_history = [0]
            middle_stops = required[1:-1]
            best_time, best_route = brute_force_route(self.tm, required[0], middle_stops, required[-1])
        else:
            top_routes, ga_history = self.ga.run(required, top_k=1)
            best_time = top_routes[0][0]
            best_route = top_routes[0][1]

        runtime_ms = (_time.perf_counter() - t0) * 1000.0

        # ── Dijkstra baseline ────────────────────────────────────
        _, dijk_time = dijkstra(self.tm, start, end)

        improv = ((dijk_time - best_time) / dijk_time * 100
                  if dijk_time > 0 else 0.0)

        return {
            "route":            best_route,
            "route_names":      [STOPS[i]["name"] for i in best_route],
            "travel_time_min":  round(float(best_time), 2),
            "dijkstra_time":    round(float(dijk_time), 2),
            "improvement_pct":  round(improv, 1),
            "runtime_ms":       round(runtime_ms, 2),
            "ga_history":       ga_history,
        }
