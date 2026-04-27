"""
main.py  — Runs the full SmartRoute pipeline:
  1. Data generation
  2. LSTM training
  3. DQN training
  4. Evaluation (4-method comparison)
  5. Results chart
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time as _time

from config import N, STOPS, RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Phase 1: Data ────────────────────────────────────────────────
print("\n" + "="*55)
print("PHASE 1  — DATA GENERATION")
print("="*55)
from utils.preprocess import (save_bus_stops, generate_traffic_data,
                               generate_weather_data, build_feature_sequences)
save_bus_stops()
generate_traffic_data()
generate_weather_data()
build_feature_sequences()

# ── Phase 2: LSTM ────────────────────────────────────────────────
print("\n" + "="*55)
print("PHASE 2  — LSTM MODEL")
print("="*55)
from lstm.train_lstm import train as train_lstm


model_path = os.path.join("lstm", "best_model.keras")

if os.path.exists(model_path):

    print("[LSTM] Existing model found → loading")

else:

    print("[LSTM] No model found → training")

    train_lstm()



# ── Phase 4: Evaluation ──────────────────────────────────────────
print("\n" + "="*55)
print("PHASE 4  — EVALUATION  (10 random routes)")
print("="*55)
from engine.optimizer import SmartRouteOptimizer, brute_force_route
from ga.genetic_algorithm import GeneticAlgorithmRouter

optimizer = SmartRouteOptimizer(hour=8)
ga_only   = GeneticAlgorithmRouter(optimizer.tm, generations=100)



rng     = np.random.default_rng(99)
N_TRIALS = 10
results  = {k: [] for k in ["greedy","ga","hybrid","optimal"]}
runtimes = {k: [] for k in ["greedy","ga","hybrid"]}

for trial in range(N_TRIALS):
    start = int(rng.integers(0, N))
    end   = int(rng.integers(0, N))
    while end == start:
        end = int(rng.integers(0, N))

    # choose 3 intermediate stops

    candidates = list(range(max(0,start-8), min(N,start+8)))

    candidates = [i for i in candidates if i not in [start,end]]

    waypoints = list(rng.choice(candidates, size=4, replace=False))

    # Greedy Nearest Neighbor Baseline
    t0 = _time.perf_counter()
    
    unvisited = set(waypoints)
    current = start
    greedy_dt = 0.0
    while unvisited:
        next_node = min(unvisited, key=lambda x: optimizer.tm[current][x])
        greedy_dt += optimizer.tm[current][next_node]
        current = next_node
        unvisited.remove(next_node)
    greedy_dt += optimizer.tm[current][end]

    runtimes["greedy"].append((_time.perf_counter() - t0) * 1000)
    results["greedy"].append(greedy_dt)

    # GA only
    t0 = _time.perf_counter()
    routes, _ = ga_only.run([start] + waypoints + [end], top_k=1)
    runtimes["ga"].append((_time.perf_counter() - t0) * 1000)
    results["ga"].append(routes[0][0])



    # Hybrid
    t0  = _time.perf_counter()
    out = optimizer.optimize(start, end, waypoints=waypoints)
    runtimes["hybrid"].append((_time.perf_counter() - t0) * 1000)
    results["hybrid"].append(out["travel_time_min"])

    # Brute Force Optimal
    optimal_cost, _ = brute_force_route(optimizer.tm, start, waypoints, end)
    results["optimal"].append(optimal_cost)

    print(
    f"Trial {trial+1:2d}: "
    f"{start} → {waypoints} → {end} | "
    f"Greedy={greedy_dt:.1f}  "
    f"Optimal={optimal_cost:.1f}  "
    f"GA={routes[0][0]:.1f}  "
    f"Hybrid={out['travel_time_min']:.1f}")

# ── Summary table ─────────────────────────────────────────────────
print("\n" + "-"*65)
print(f"{'Method':<12} {'Mean (min)':>11} {'Std':>7} {'Runtime (ms)':>14} {'% Optimal':>12}")
print("-" * 65)
optimal_mean = np.mean(results["optimal"])
for m in ["greedy","ga","hybrid"]:
    mean_val = np.mean(results[m])
    pct_optimal = (mean_val / optimal_mean - 1) * 100
    print(f"{m:<12} {mean_val:>10.2f} {np.std(results[m]):>6.2f} {np.mean(runtimes[m]):>13.2f} {pct_optimal:>11.1f}%")

# ── Bar chart ─────────────────────────────────────────────────────
labels = ["Greedy TSP", "GA Only", "Hybrid (Ours)"]
means  = [np.mean(results[m]) for m in ["greedy","ga","hybrid"]]
stds   = [np.std(results[m])  for m in ["greedy","ga","hybrid"]]
colors = ["#888780", "#1D9E75", "#534AB7"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, means, yerr=stds, capsize=6,
              color=colors, edgecolor="white", width=0.55)
ax.set_ylabel("Average travel time (minutes)")
ax.set_title("SmartRoute — routing method comparison")
for bar, v in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.15,
            f"{v:.1f} min", ha="center", fontsize=10)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "comparison_chart.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\n[main] Chart saved → {out_path}")

# ── Example route ────────────────────────────────────────────────
print("\n[main] Example route: Kashmere Gate → Noida Sec 15")
res = optimizer.optimize(0, 24, waypoints=[8, 4])
print(f"  Path      : {' → '.join(res['route_names'])}")

unvisited = set([8, 4])
current = 0
greedy_dt = 0.0
while unvisited:
    next_node = min(unvisited, key=lambda x: optimizer.tm[current][x])
    greedy_dt += optimizer.tm[current][next_node]
    current = next_node
    unvisited.remove(next_node)
greedy_dt += optimizer.tm[current][24]

print(f"  Hybrid    : {res['travel_time_min']} min")
print(f"  Greedy TSP: {greedy_dt:.2f} min")
print(f"  Improvement: {(greedy_dt - res['travel_time_min']) / greedy_dt * 100:.1f}%")
print(f"  Runtime   : {res['runtime_ms']} ms")

# ── Scalability Experiment ────────────────────────────────────────
print("\n" + "="*55)
print("PHASE 5  — SCALABILITY EXPERIMENT (Brute vs GA)")
print("="*55)
import time
sizes = [3, 5, 7]
for size in sizes:
    test_waypoints = list(range(1, size+1))
    
    # Brute Force
    t0 = time.perf_counter()
    b_cost, _ = brute_force_route(optimizer.tm, 0, test_waypoints, 24)
    b_time = (time.perf_counter() - t0) * 1000
    
    # GA Only
    t0 = time.perf_counter()
    ga_routes, _ = ga_only.run([0] + test_waypoints + [24], top_k=1)
    g_cost = ga_routes[0][0]
    g_time = (time.perf_counter() - t0) * 1000
    
    diff = g_cost - b_cost
    print(f"Waypoints: {size} | Brute Force: {b_cost:.1f} ({b_time:>7.1f}ms) | GA: {g_cost:.1f} ({g_time:>6.1f}ms) | Diff: +{diff:.1f}")

# ── Real-World Traffic Experiment ─────────────────────────────────
print("\n" + "="*55)
print("PHASE 6  — REAL-WORLD TRAFFIC IMPACT (Dynamic Routing)")
print("="*55)

opt_8am = SmartRouteOptimizer(hour=8)
opt_2pm = SmartRouteOptimizer(hour=14)

changed_routes = 0
NUM_TESTS = 20

for test in range(NUM_TESTS):
    # random start, end, and 4 waypoints
    start = int(rng.integers(0, N))
    end   = int(rng.integers(0, N))
    while end == start:
        end = int(rng.integers(0, N))
        
    wps = []
    while len(wps) < 4:
        w = int(rng.integers(0, N))
        if w not in [start, end] and w not in wps:
            wps.append(w)
            
    res_8am = opt_8am.optimize(start, end, waypoints=wps)
    res_2pm = opt_2pm.optimize(start, end, waypoints=wps)
    
    if res_8am['route'] != res_2pm['route']:
        changed_routes += 1

pct_changed = (changed_routes / NUM_TESTS) * 100

print(f"Tested {NUM_TESTS} random complex routes.")
print(f"8 AM (Rush Hour) vs 2 PM (Off-Peak) → {pct_changed:.1f}% routes changed sequence to avoid traffic!")