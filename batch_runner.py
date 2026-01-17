"""
batch_runner.py
===============
Batch Experiment Runner for Multiple Seeds

Runs PSO/GA multiple times with different random seeds and aggregates statistics
for thesis comparison experiments.

Usage:
    python batch_runner.py --algorithms PSO GA --n-runs 10 --case Case1_COL2 --demo

Author: PSE Lab, NTUST
Version: 1.0
"""

import argparse
import json
import os
import csv
import time
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Type

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SingleRunResult:
    """Result from a single optimization run."""
    seed: int
    tac: float
    nt: int
    feed: int
    pressure: float
    T_reb: float = 0.0
    time_seconds: float = 0.0
    evaluations: int = 0
    feasible: int = 0
    infeasible: int = 0
    converged: bool = True


@dataclass
class BatchResult:
    """Aggregated results from multiple runs."""
    algorithm: str
    case_name: str
    n_runs: int

    # TAC statistics
    tac_mean: float = 0.0
    tac_std: float = 0.0
    tac_best: float = 0.0
    tac_worst: float = 0.0
    tac_median: float = 0.0

    # Time statistics
    time_mean: float = 0.0
    time_std: float = 0.0

    # Evaluation statistics
    evals_mean: float = 0.0
    evals_std: float = 0.0

    # Best configuration
    best_nt: int = 0
    best_feed: int = 0
    best_pressure: float = 0.0

    # Individual runs
    all_results: List[SingleRunResult] = field(default_factory=list)

    # Metadata
    timestamp: str = ""


# ════════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ════════════════════════════════════════════════════════════════════════════

class BatchRunner:
    """
    Batch experiment runner for optimization algorithm comparison.

    Runs an optimizer multiple times with different seeds and aggregates
    statistics for fair algorithm comparison.

    Parameters
    ----------
    evaluator : TACEvaluator or DemoEvaluator
        Evaluator instance
    config : dict
        Configuration with bounds
    output_dir : str
        Directory for output files
    """

    def __init__(self, evaluator, config: dict, output_dir: str = "results"):
        self.evaluator = evaluator
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_batch(
        self,
        optimizer_class: Type,
        case_name: str,
        n_runs: int = 10,
        seeds: Optional[List[int]] = None,
        **optimizer_kwargs
    ) -> BatchResult:
        """
        Run optimizer multiple times with different seeds.

        Parameters
        ----------
        optimizer_class : Type
            Optimizer class (PSOOptimizer or GAOptimizer)
        case_name : str
            Case name for logging
        n_runs : int
            Number of runs
        seeds : List[int], optional
            List of random seeds (default: 42, 43, ...)
        **optimizer_kwargs
            Additional arguments for optimizer

        Returns
        -------
        BatchResult
            Aggregated results
        """
        if seeds is None:
            seeds = list(range(42, 42 + n_runs))

        algorithm = optimizer_class.__name__.replace("Optimizer", "").upper()

        logger.info("=" * 70)
        logger.info(f"BATCH RUN: {algorithm}")
        logger.info("=" * 70)
        logger.info(f"Case: {case_name}")
        logger.info(f"Runs: {n_runs}")
        logger.info(f"Seeds: {seeds}")
        logger.info("=" * 70)

        results = []
        start_time = time.time()

        for i, seed in enumerate(seeds):
            logger.info(f"\n--- Run {i + 1}/{n_runs} (seed={seed}) ---")

            # Progress
            print(f"[PROGRESS] {json.dumps({'phase': 'batch_run', 'run': i + 1, 'total': n_runs, 'seed': seed, 'algorithm': algorithm})}")

            # Set seed in config
            run_config = optimizer_kwargs.copy()
            if 'pso_config' in run_config:
                run_config['pso_config'].seed = seed
            elif 'ga_config' in run_config:
                run_config['ga_config'].seed = seed

            # Create and run optimizer
            try:
                # Reset evaluator statistics if possible
                if hasattr(self.evaluator, 'reset_statistics'):
                    self.evaluator.reset_statistics()
                if hasattr(self.evaluator, 'clear_cache'):
                    self.evaluator.clear_cache()

                # Create optimizer with seed
                np.random.seed(seed)

                # Determine config type based on optimizer class
                if 'PSO' in optimizer_class.__name__:
                    from pso_optimizer import PSOConfig
                    config_obj = PSOConfig(seed=seed, **{k: v for k, v in run_config.items() if k in ['n_particles', 'n_iterations']})
                    optimizer = optimizer_class(self.evaluator, self.config, pso_config=config_obj)
                elif 'GA' in optimizer_class.__name__:
                    from ga_optimizer import GAConfig
                    config_obj = GAConfig(seed=seed, **{k: v for k, v in run_config.items() if k in ['pop_size', 'n_generations']})
                    optimizer = optimizer_class(self.evaluator, self.config, ga_config=config_obj)
                else:
                    optimizer = optimizer_class(self.evaluator, self.config)

                result = optimizer.run(case_name)

                # Extract results
                run_result = SingleRunResult(
                    seed=seed,
                    tac=result.optimal_tac,
                    nt=result.optimal_nt,
                    feed=result.optimal_feed,
                    pressure=result.optimal_pressure,
                    T_reb=getattr(result, 'optimal_T_reb', 0),
                    time_seconds=result.total_time_seconds,
                    evaluations=result.total_evaluations,
                    feasible=result.feasible_evaluations,
                    infeasible=result.infeasible_evaluations,
                    converged=True,
                )
                results.append(run_result)

                logger.info(f"  TAC: ${result.optimal_tac:,.0f}")

            except Exception as e:
                logger.error(f"  Run failed: {e}")
                results.append(SingleRunResult(
                    seed=seed,
                    tac=float('inf'),
                    nt=0, feed=0, pressure=0,
                    converged=False
                ))

        # Aggregate statistics
        valid_results = [r for r in results if r.tac < 1e10]

        if valid_results:
            tacs = [r.tac for r in valid_results]
            times = [r.time_seconds for r in valid_results]
            evals = [r.evaluations for r in valid_results]

            best_idx = np.argmin(tacs)
            best_run = valid_results[best_idx]

            batch_result = BatchResult(
                algorithm=algorithm,
                case_name=case_name,
                n_runs=n_runs,
                tac_mean=np.mean(tacs),
                tac_std=np.std(tacs),
                tac_best=np.min(tacs),
                tac_worst=np.max(tacs),
                tac_median=np.median(tacs),
                time_mean=np.mean(times),
                time_std=np.std(times),
                evals_mean=np.mean(evals),
                evals_std=np.std(evals),
                best_nt=best_run.nt,
                best_feed=best_run.feed,
                best_pressure=best_run.pressure,
                all_results=results,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        else:
            batch_result = BatchResult(
                algorithm=algorithm,
                case_name=case_name,
                n_runs=n_runs,
                all_results=results,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            )

        total_time = time.time() - start_time
        logger.info(f"\nBatch complete in {total_time:.1f}s")
        self._print_summary(batch_result)

        return batch_result

    def _print_summary(self, result: BatchResult):
        """Print batch summary."""
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"BATCH SUMMARY: {result.algorithm}")
        logger.info("=" * 50)
        logger.info(f"Case: {result.case_name}")
        logger.info(f"Runs: {result.n_runs}")
        logger.info("")
        logger.info(f"TAC Statistics:")
        logger.info(f"  Mean: ${result.tac_mean:,.0f}")
        logger.info(f"  Std:  ${result.tac_std:,.0f}")
        logger.info(f"  Best: ${result.tac_best:,.0f}")
        logger.info(f"  Worst: ${result.tac_worst:,.0f}")
        logger.info(f"  Median: ${result.tac_median:,.0f}")
        logger.info("")
        logger.info(f"Best Configuration:")
        logger.info(f"  NT: {result.best_nt}")
        logger.info(f"  Feed: {result.best_feed}")
        logger.info(f"  Pressure: {result.best_pressure:.4f}")
        logger.info("=" * 50)

    def export_csv(self, batch_results: List[BatchResult], filename: str):
        """
        Export comparison results to CSV.

        Parameters
        ----------
        batch_results : List[BatchResult]
            Results from multiple algorithms
        filename : str
            Output CSV filename
        """
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Algorithm', 'Case', 'N_Runs',
                'TAC_Mean', 'TAC_Std', 'TAC_Best', 'TAC_Worst', 'TAC_Median',
                'Time_Mean', 'Time_Std', 'Evals_Mean', 'Evals_Std',
                'Best_NT', 'Best_Feed', 'Best_P'
            ])

            # Data rows
            for br in batch_results:
                writer.writerow([
                    br.algorithm, br.case_name, br.n_runs,
                    f"{br.tac_mean:.2f}", f"{br.tac_std:.2f}",
                    f"{br.tac_best:.2f}", f"{br.tac_worst:.2f}", f"{br.tac_median:.2f}",
                    f"{br.time_mean:.2f}", f"{br.time_std:.2f}",
                    f"{br.evals_mean:.1f}", f"{br.evals_std:.1f}",
                    br.best_nt, br.best_feed, f"{br.best_pressure:.4f}"
                ])

        logger.info(f"CSV exported to: {filepath}")
        return filepath

    def export_json(self, batch_results: List[BatchResult], filename: str):
        """Export comparison results to JSON."""
        filepath = os.path.join(self.output_dir, filename)

        output = {
            "comparison": [],
            "timestamp": datetime.now().isoformat(),
        }

        for br in batch_results:
            output["comparison"].append({
                "algorithm": br.algorithm,
                "case_name": br.case_name,
                "n_runs": br.n_runs,
                "tac_stats": {
                    "mean": br.tac_mean,
                    "std": br.tac_std,
                    "best": br.tac_best,
                    "worst": br.tac_worst,
                    "median": br.tac_median,
                },
                "time_stats": {
                    "mean": br.time_mean,
                    "std": br.time_std,
                },
                "evals_stats": {
                    "mean": br.evals_mean,
                    "std": br.evals_std,
                },
                "best_config": {
                    "nt": br.best_nt,
                    "feed": br.best_feed,
                    "pressure": br.best_pressure,
                },
                "individual_runs": [asdict(r) for r in br.all_results],
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        logger.info(f"JSON exported to: {filepath}")
        return filepath


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch Experiment Runner for Algorithm Comparison"
    )
    parser.add_argument("--algorithms", nargs="+", default=["PSO", "GA"],
                       help="Algorithms to compare (PSO, GA)")
    parser.add_argument("--n-runs", type=int, default=10,
                       help="Number of runs per algorithm")
    parser.add_argument("--case", default="Case1_COL2",
                       help="Case name")
    parser.add_argument("--demo", action="store_true",
                       help="Use demo evaluator")
    parser.add_argument("--output", default="results",
                       help="Output directory")
    parser.add_argument("--n-particles", type=int, default=20,
                       help="PSO particles")
    parser.add_argument("--n-iterations", type=int, default=50,
                       help="PSO iterations")
    parser.add_argument("--pop-size", type=int, default=50,
                       help="GA population size")
    parser.add_argument("--n-generations", type=int, default=100,
                       help="GA generations")

    args = parser.parse_args()

    # Load configuration
    try:
        from config import get_case_config
        config = get_case_config(args.case)
        if not config:
            config = {
                'bounds': {
                    'nt_bounds': (20, 70),
                    'feed_bounds': (10, 60),
                    'pressure_bounds': (0.1, 0.5),
                },
                'min_section_stages': 3,
                'T_reb_max': 120.0,
            }
    except ImportError:
        config = {
            'bounds': {
                'nt_bounds': (20, 70),
                'feed_bounds': (10, 60),
                'pressure_bounds': (0.1, 0.5),
            },
            'min_section_stages': 3,
            'T_reb_max': 120.0,
        }

    # Create evaluator
    if args.demo:
        from demo_evaluator import DemoEvaluator
        evaluator = DemoEvaluator(delay=0.02)
        logger.info("Using DEMO evaluator")
    else:
        from demo_evaluator import DemoEvaluator
        evaluator = DemoEvaluator(delay=0.02)
        logger.info("Using DEMO evaluator (fallback)")

    # Create batch runner
    runner = BatchRunner(evaluator, config, output_dir=args.output)

    # Run experiments
    batch_results = []

    for algorithm in args.algorithms:
        algorithm = algorithm.upper()

        if algorithm == "PSO":
            from pso_optimizer import PSOOptimizer
            result = runner.run_batch(
                PSOOptimizer,
                args.case,
                n_runs=args.n_runs,
                n_particles=args.n_particles,
                n_iterations=args.n_iterations,
            )
            batch_results.append(result)

        elif algorithm == "GA":
            try:
                from ga_optimizer import GAOptimizer, PYMOO_AVAILABLE
                if PYMOO_AVAILABLE:
                    result = runner.run_batch(
                        GAOptimizer,
                        args.case,
                        n_runs=args.n_runs,
                        pop_size=args.pop_size,
                        n_generations=args.n_generations,
                    )
                else:
                    from ga_optimizer import SimpleGAOptimizer
                    result = runner.run_batch(
                        SimpleGAOptimizer,
                        args.case,
                        n_runs=args.n_runs,
                        pop_size=args.pop_size,
                        n_generations=args.n_generations,
                    )
                batch_results.append(result)
            except ImportError as e:
                logger.error(f"Cannot run GA: {e}")

    # Export results
    if batch_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runner.export_csv(batch_results, f"comparison_{timestamp}.csv")
        runner.export_json(batch_results, f"comparison_{timestamp}.json")

    logger.info("\nBatch experiments complete!")


if __name__ == "__main__":
    main()
