"""
demo_runner.py
==============
Demo script to simulate optimizer output for testing the server/dashboard.

Usage:
    python demo_runner.py <job_id> [algorithm]

    job_id: Unique identifier for this job
    algorithm: ISO, PSO, or GA (default: ISO)

It will print [PROGRESS] JSON lines with simulated progress and create
a results file at the end.
"""
import sys
import time
import json
import os
from datetime import datetime

# Parse arguments
job_id = sys.argv[1] if len(sys.argv) > 1 else "demo"
algorithm = sys.argv[2].upper() if len(sys.argv) > 2 else "ISO"

if algorithm not in ("ISO", "PSO", "GA"):
    algorithm = "ISO"

print(f"Demo runner starting for job {job_id} (algorithm: {algorithm})")
print(f"[PROGRESS] {json.dumps({'iteration': 0, 'phase': 'initialization', 'algorithm': algorithm, 'message': 'Starting demo run'})}")

# Simulate different algorithms
if algorithm == "ISO":
    # ISO: 3 iterations, each with P->NT->NF sweeps
    n_iterations = 3
    best_tac = 150000

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'='*50}")
        print(f"ISO ITERATION {iteration}")
        print(f"{'='*50}")

        # Progress: iteration start
        print(f"[PROGRESS] {json.dumps({'iteration': iteration, 'phase': 'iteration_start', 'current': iteration, 'total': n_iterations, 'algorithm': 'ISO', 'best_tac': best_tac})}")
        time.sleep(0.5)

        # Pressure sweep
        print(f"[PROGRESS] {json.dumps({'iteration': iteration, 'phase': 'pressure_sweep_start', 'algorithm': 'ISO', 'message': 'Sweeping pressure'})}")
        for p in range(1, 6):
            best_tac -= 500
            print(f"[PROGRESS] {json.dumps({'iteration': iteration, 'phase': 'pressure_sweep', 'current': p, 'total': 5, 'algorithm': 'ISO', 'best_tac': best_tac, 'pressure': 0.1 + p*0.05})}")
            time.sleep(0.3)
        print(f"  -> Optimal pressure: 0.20 bar")

        # NT sweep
        print(f"[PROGRESS] {json.dumps({'iteration': iteration, 'phase': 'nt_sweep_start', 'algorithm': 'ISO', 'message': 'Sweeping NT'})}")
        for nt_idx in range(1, 8):
            best_tac -= 800
            print(f"[PROGRESS] {json.dumps({'iteration': iteration, 'phase': 'nt_sweep', 'current': nt_idx, 'total': 7, 'algorithm': 'ISO', 'best_tac': best_tac, 'nt': 30 + nt_idx*3})}")
            time.sleep(0.2)
        print(f"  -> Optimal NT: 45")

        # Feed sweep
        print(f"[PROGRESS] {json.dumps({'iteration': iteration, 'phase': 'feed_sweep_start', 'algorithm': 'ISO', 'message': 'Sweeping feed'})}")
        for f_idx in range(1, 6):
            best_tac -= 300
            print(f"[PROGRESS] {json.dumps({'iteration': iteration, 'phase': 'feed_sweep', 'current': f_idx, 'total': 5, 'algorithm': 'ISO', 'best_tac': best_tac, 'feed': 15 + f_idx*2})}")
            time.sleep(0.2)
        print(f"  -> Optimal Feed: 22")

        print(f"Iteration {iteration} complete: TAC = ${best_tac:,}")
        time.sleep(0.3)

    optimal = {"nt": 45, "feed": 22, "pressure": 0.20, "tac": best_tac}

elif algorithm == "PSO":
    # PSO: 10 iterations of swarm
    n_iterations = 10
    best_tac = 160000

    for iteration in range(1, n_iterations + 1):
        # Simulated improvement
        improvement = 5000 / iteration
        best_tac -= improvement

        print(f"[PROGRESS] {json.dumps({'iteration': iteration, 'phase': 'pso_iteration', 'current': iteration, 'total': n_iterations, 'algorithm': 'PSO', 'best_tac': round(best_tac, 2), 'message': f'PSO iteration {iteration}'})}")
        print(f"PSO Iteration {iteration}: Best TAC = ${best_tac:,.0f}")
        time.sleep(0.5)

    optimal = {"nt": 44, "feed": 21, "pressure": 0.22, "tac": round(best_tac, 2)}

elif algorithm == "GA":
    # GA: 15 generations
    n_iterations = 15
    best_tac = 155000

    for generation in range(1, n_iterations + 1):
        # Simulated improvement
        improvement = 4000 / generation
        best_tac -= improvement

        print(f"[PROGRESS] {json.dumps({'iteration': generation, 'phase': 'ga_generation', 'current': generation, 'total': n_iterations, 'algorithm': 'GA', 'best_tac': round(best_tac, 2), 'message': f'GA generation {generation}'})}")
        print(f"GA Generation {generation}: Best TAC = ${best_tac:,.0f}")
        time.sleep(0.4)

    optimal = {"nt": 46, "feed": 23, "pressure": 0.19, "tac": round(best_tac, 2)}

# Emit completion
print(f"[PROGRESS] {json.dumps({'iteration': n_iterations, 'phase': 'completed', 'current': n_iterations, 'total': n_iterations, 'algorithm': algorithm, 'best_tac': optimal['tac'], 'message': f'{algorithm} optimization complete!', 'optimal_nt': optimal['nt'], 'optimal_feed': optimal['feed'], 'optimal_pressure': optimal['pressure']})}")

# Save result
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)

result_filename = f"{algorithm.lower()}_result_{timestamp}.json"
outfile = os.path.join(results_dir, result_filename)

result_data = {
    "metadata": {
        "case_name": f"demo_{job_id}",
        "algorithm": algorithm,
        "methodology": f"Demo {algorithm} Optimization",
        "temperature_constraint": "T_reb <= 120C",
    },
    "optimal": optimal,
    "convergence": {
        "converged": True,
        "iterations": n_iterations,
    },
    "statistics": {
        "total_evaluations": n_iterations * 10,
        "feasible": n_iterations * 9,
        "infeasible": n_iterations,
        "time_seconds": n_iterations * 0.5,
    },
    "algorithm": algorithm,
    "timestamp": timestamp,
}

with open(outfile, "w", encoding="utf-8") as f:
    json.dump(result_data, f, indent=2)

print(f"\nResults saved to: {outfile}")
print(f"Demo runner finished ({algorithm})")
