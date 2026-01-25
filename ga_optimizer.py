"""
ga_optimizer.py
===============
Genetic Algorithm (GA) for Distillation Column Design using pymoo

Implements GA to find optimal (NT, Feed, Pressure) that minimizes TAC
while respecting the temperature constraint T_reb <= 120C.

Usage:
    python ga_optimizer.py Case1_COL2 --demo
    python ga_optimizer.py Case1_COL2 --pop-size 50 --n-generations 100

Requirements:
    pip install pymoo

Author: PSE Lab, NTUST
Version: 1.0
"""

import argparse
import json
import logging
import os
import sys
import time
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Initialize COM for Windows subprocess compatibility
try:
    import pythoncom
    pythoncom.CoInitialize()
except ImportError:
    pass  # Not on Windows or pythoncom not available

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Temperature constraint
T_REBOILER_MAX = 120.0


# ════════════════════════════════════════════════════════════════════════════
# PROGRESS LOGGING
# ════════════════════════════════════════════════════════════════════════════

def _emit_progress(iteration: int, phase: str, current: int = 0, total: int = 0,
                   best_tac: float = None, message: str = None, **kwargs):
    """Emit structured progress JSON for dashboard live tracking."""
    progress = {
        "iteration": iteration,
        "phase": phase,
        "current": current,
        "total": total,
        "algorithm": "GA",
    }
    if best_tac is not None and best_tac < 1e10:
        progress["best_tac"] = round(best_tac, 2)
    if message:
        progress["message"] = message
    progress.update(kwargs)

    print(f"[PROGRESS] {json.dumps(progress)}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class GAConfig:
    """GA algorithm configuration."""
    pop_size: int = 50
    n_generations: int = 100
    crossover_prob: float = 0.9
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    seed: Optional[int] = None


@dataclass
class GAResult:
    """GA optimization result."""
    # Optimal values
    optimal_nt: int = 0
    optimal_feed: int = 0
    optimal_pressure: float = 0.0
    optimal_tac: float = float('inf')

    # Additional optimal info
    optimal_T_reb: float = 0.0
    optimal_tpc: float = 0.0
    optimal_toc: float = 0.0

    # Convergence history
    convergence_history: List[float] = field(default_factory=list)
    best_per_generation: List[Dict] = field(default_factory=list)

    # Statistics
    total_time_seconds: float = 0.0
    total_evaluations: int = 0
    feasible_evaluations: int = 0
    infeasible_evaluations: int = 0

    # Configuration
    pop_size: int = 0
    n_generations: int = 0

    # Metadata
    case_name: str = ""
    algorithm: str = "GA"
    timestamp: str = ""


# ════════════════════════════════════════════════════════════════════════════
# PYMOO PROBLEM DEFINITION
# ════════════════════════════════════════════════════════════════════════════

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.core.callback import Callback
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logger.warning("pymoo not installed. Install with: pip install pymoo")


if PYMOO_AVAILABLE:
    class DistillationProblem(Problem):
        """
        pymoo Problem definition for distillation column optimization.

        Variables: [NT, Feed, Pressure]
        Objective: Minimize TAC
        Constraint: T_reb <= 120C
        """

        def __init__(self, evaluator, config: dict, optimizer_ref=None, purity_spec=None):
            self.evaluator = evaluator
            self.config = config
            self.optimizer_ref = optimizer_ref
            self.purity_spec = purity_spec
            self.T_reb_max = config.get('T_reb_max', T_REBOILER_MAX)
            self.min_section_stages = config.get('min_section_stages', 3)

            # Extract bounds
            nt_bounds = config['bounds']['nt_bounds']
            feed_bounds = config['bounds']['feed_bounds']
            p_bounds = config['bounds']['pressure_bounds']

            super().__init__(
                n_var=3,
                n_obj=1,
                n_ieq_constr=1,  # Temperature constraint: g(x) <= 0
                xl=np.array([nt_bounds[0], feed_bounds[0], p_bounds[0]]),
                xu=np.array([nt_bounds[1], feed_bounds[1], p_bounds[1]]),
            )

        def _evaluate(self, X, out, *args, **kwargs):
            """Evaluate population."""
            f = []  # Objectives
            g = []  # Constraints

            for x in X:
                nt = int(round(x[0]))
                feed = int(round(x[1]))
                pressure = x[2]

                # Enforce valid feed
                feed = max(self.min_section_stages + 1, min(feed, nt - self.min_section_stages))

                # Evaluate (with diagnostic on failure)
                result = self.evaluator.evaluate(nt, feed, pressure, run_diagnostic_on_fail=True, rr_sweep_on_fail=True, purity_spec=self.purity_spec)

                tac = result.get('TAC', 1e12)
                T_reb = result.get('T_reb')
                converged = result.get('converged', False)

                # If T_reb is None, cannot verify constraint -> treat as infeasible
                if T_reb is None:
                    T_reb = self.T_reb_max + 50  # Force constraint violation
                    logger.debug(f"  T_reb extraction failed for NT={nt}, P={pressure:.4f} - treating as infeasible")

                # Track statistics
                if self.optimizer_ref:
                    self.optimizer_ref.eval_count += 1
                    if converged and T_reb <= self.T_reb_max:
                        self.optimizer_ref.feasible_count += 1
                    else:
                        self.optimizer_ref.infeasible_count += 1

                # Objective: TAC (add penalty for non-convergence)
                if not converged:
                    tac = 1e12

                f.append(tac)

                # Constraint: g(x) = T_reb - T_max <= 0
                g.append(T_reb - self.T_reb_max)

            out["F"] = np.array(f).reshape(-1, 1)
            out["G"] = np.array(g).reshape(-1, 1)


    class ProgressCallback(Callback):
        """Callback to emit progress during GA evolution."""

        def __init__(self, n_generations: int, optimizer_ref=None):
            super().__init__()
            self.n_generations = n_generations
            self.optimizer_ref = optimizer_ref
            self.best_history = []

        def notify(self, algorithm):
            gen = algorithm.n_gen
            pop = algorithm.pop

            # Get best feasible individual
            F = pop.get("F")
            G = pop.get("G")

            # Find best feasible
            feasible_mask = (G <= 0).flatten() if G is not None else np.ones(len(F), dtype=bool)
            if feasible_mask.any():
                feasible_F = F[feasible_mask]
                best_idx = np.argmin(feasible_F)
                best_tac = feasible_F[best_idx][0]
            else:
                best_tac = F.min()

            self.best_history.append(best_tac)

            # Store in optimizer
            if self.optimizer_ref:
                self.optimizer_ref.convergence_history.append(
                    best_tac if best_tac < 1e10 else None
                )

            # Emit progress
            _emit_progress(
                iteration=gen,
                phase="ga_generation",
                current=gen,
                total=self.n_generations,
                best_tac=best_tac if best_tac < 1e10 else None,
                message=f"Generation {gen}/{self.n_generations}"
            )

            logger.info(f"Generation {gen}: Best TAC = ${best_tac:,.0f}")


# ════════════════════════════════════════════════════════════════════════════
# GA OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════

class GAOptimizer:
    """
    Genetic Algorithm Optimizer for Distillation Column Design.

    Uses pymoo for GA implementation with constraint handling.

    Parameters
    ----------
    evaluator : TACEvaluator or DemoEvaluator
        Evaluator that computes TAC for a given configuration
    config : dict
        Configuration with bounds
    ga_config : GAConfig, optional
        GA algorithm parameters
    """

    def __init__(self, evaluator, config: dict, ga_config: Optional[GAConfig] = None, purity_spec=None):
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo is required for GA optimizer. Install with: pip install pymoo")

        self.evaluator = evaluator
        self.config = config
        self.ga_config = ga_config or GAConfig()
        self.purity_spec = purity_spec

        # Temperature constraint
        self.T_reb_max = config.get('T_reb_max', T_REBOILER_MAX)

        # Statistics
        self.eval_count = 0
        self.feasible_count = 0
        self.infeasible_count = 0
        self.convergence_history = []

        # Set random seed
        if self.ga_config.seed is not None:
            np.random.seed(self.ga_config.seed)

    def run(self, case_name: str = "Case") -> GAResult:
        """
        Run GA optimization using pymoo.

        Parameters
        ----------
        case_name : str
            Name for this optimization case

        Returns
        -------
        GAResult
            Optimization results
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("=" * 70)
        logger.info("GENETIC ALGORITHM (GA) OPTIMIZATION")
        logger.info("=" * 70)
        logger.info(f"Case: {case_name}")
        logger.info(f"Population: {self.ga_config.pop_size}")
        logger.info(f"Generations: {self.ga_config.n_generations}")
        logger.info(f"Temperature constraint: T_reb <= {self.T_reb_max}C")
        logger.info("=" * 70)

        _emit_progress(0, "initialization", message="Initializing GA population")

        # Create problem
        problem = DistillationProblem(self.evaluator, self.config, optimizer_ref=self, purity_spec=self.purity_spec)

        # Create algorithm
        algorithm = GA(
            pop_size=self.ga_config.pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.ga_config.crossover_prob, eta=self.ga_config.crossover_eta),
            mutation=PM(eta=self.ga_config.mutation_eta),
            eliminate_duplicates=True,
        )

        # Create callback
        callback = ProgressCallback(self.ga_config.n_generations, optimizer_ref=self)

        # Run optimization
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.ga_config.n_generations),
            callback=callback,
            seed=self.ga_config.seed,
            verbose=False,
        )

        total_time = time.time() - start_time

        # Extract optimal solution
        if res.X is not None:
            optimal_nt = int(round(res.X[0]))
            optimal_feed = int(round(res.X[1]))
            optimal_pressure = res.X[2]

            # Ensure valid feed
            min_section = self.config.get('min_section_stages', 3)
            optimal_feed = max(min_section + 1, min(optimal_feed, optimal_nt - min_section))

            optimal_tac = res.F[0] if res.F is not None else float('inf')

            # Get final evaluation for complete info (with diagnostic on failure)
            final_result = self.evaluator.evaluate(optimal_nt, optimal_feed, optimal_pressure, run_diagnostic_on_fail=True, rr_sweep_on_fail=True, purity_spec=self.purity_spec)
            optimal_T_reb = final_result.get('T_reb', 0)
            optimal_tpc = final_result.get('TPC', 0)
            optimal_toc = final_result.get('TOC', 0)
        else:
            optimal_nt = 0
            optimal_feed = 0
            optimal_pressure = 0.0
            optimal_tac = float('inf')
            optimal_T_reb = 0
            optimal_tpc = 0
            optimal_toc = 0

        # Build result
        result = GAResult(
            optimal_nt=optimal_nt,
            optimal_feed=optimal_feed,
            optimal_pressure=round(optimal_pressure, 4),
            optimal_tac=optimal_tac,
            optimal_T_reb=optimal_T_reb,
            optimal_tpc=optimal_tpc,
            optimal_toc=optimal_toc,
            convergence_history=[c for c in self.convergence_history if c is not None],
            best_per_generation=callback.best_history,
            total_time_seconds=total_time,
            total_evaluations=self.eval_count,
            feasible_evaluations=self.feasible_count,
            infeasible_evaluations=self.infeasible_count,
            pop_size=self.ga_config.pop_size,
            n_generations=self.ga_config.n_generations,
            case_name=case_name,
            timestamp=timestamp,
        )

        self.result = result
        self._print_summary(result)

        # Emit completion
        _emit_progress(
            iteration=self.ga_config.n_generations,
            phase="completed",
            current=self.ga_config.n_generations,
            total=self.ga_config.n_generations,
            best_tac=result.optimal_tac,
            message=f"GA complete! TAC=${result.optimal_tac:,.0f}/year",
            optimal_nt=result.optimal_nt,
            optimal_feed=result.optimal_feed,
            optimal_pressure=result.optimal_pressure
        )

        return result

    def _print_summary(self, result: GAResult):
        """Print optimization summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("GA OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Case: {result.case_name}")
        logger.info(f"Time: {result.total_time_seconds:.1f}s")
        logger.info("")
        logger.info("-" * 40)
        logger.info("OPTIMAL CONFIGURATION")
        logger.info("-" * 40)
        logger.info(f"  Number of Stages (NT): {result.optimal_nt}")
        logger.info(f"  Feed Stage (NF): {result.optimal_feed}")
        logger.info(f"  Operating Pressure: {result.optimal_pressure:.4f} bar")
        logger.info(f"  TAC: ${result.optimal_tac:,.0f}/year")
        logger.info(f"  T_reb: {result.optimal_T_reb:.1f}C")
        logger.info("")
        logger.info("-" * 40)
        logger.info("EVALUATION STATISTICS")
        logger.info("-" * 40)
        logger.info(f"  Total evaluations: {result.total_evaluations}")
        logger.info(f"  Feasible: {result.feasible_evaluations}")
        logger.info(f"  Infeasible: {result.infeasible_evaluations}")
        logger.info("=" * 70)

    def save_results(self, output_dir: str = "results") -> str:
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        if not hasattr(self, 'result'):
            raise ValueError("No results to save. Run optimization first.")

        result = self.result

        output = {
            'metadata': {
                'methodology': 'Genetic Algorithm (GA) using pymoo',
                'algorithm': 'GA',
                'temperature_constraint': f'T_reb <= {self.T_reb_max}C',
                'pop_size': result.pop_size,
                'n_generations': result.n_generations,
            },
            'optimal': {
                'nt': result.optimal_nt,
                'feed': result.optimal_feed,
                'pressure': result.optimal_pressure,
                'tac': result.optimal_tac,
                'T_reb': result.optimal_T_reb,
                'tpc': result.optimal_tpc,
                'toc': result.optimal_toc,
            },
            'convergence': {
                'history': result.convergence_history,
            },
            'statistics': {
                'total_evaluations': result.total_evaluations,
                'feasible': result.feasible_evaluations,
                'infeasible': result.infeasible_evaluations,
                'time_seconds': result.total_time_seconds,
            },
            'case_name': result.case_name,
            'algorithm': 'GA',
            'timestamp': result.timestamp,
        }

        filename = os.path.join(output_dir, f"ga_result_{result.timestamp}.json")

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Results saved to: {filename}")
        print(f"Results saved to: {filename}")

        return filename


# ════════════════════════════════════════════════════════════════════════════
# FALLBACK GA (No pymoo)
# ════════════════════════════════════════════════════════════════════════════

class SimpleGAOptimizer:
    """
    Simple GA implementation without pymoo dependency.

    Used as fallback when pymoo is not available.
    """

    def __init__(self, evaluator, config: dict, ga_config: Optional[GAConfig] = None, purity_spec=None):
        self.evaluator = evaluator
        self.config = config
        self.ga_config = ga_config or GAConfig()
        self.purity_spec = purity_spec
        self.T_reb_max = config.get('T_reb_max', T_REBOILER_MAX)

        self.eval_count = 0
        self.feasible_count = 0
        self.infeasible_count = 0
        self.convergence_history = []

        if self.ga_config.seed is not None:
            np.random.seed(self.ga_config.seed)

    def run(self, case_name: str = "Case") -> GAResult:
        """Run simple GA optimization."""
        logger.warning("Using simple GA (pymoo not available)")

        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        _emit_progress(0, "initialization", message="Initializing simple GA")

        # Bounds
        nt_bounds = self.config['bounds']['nt_bounds']
        feed_bounds = self.config['bounds']['feed_bounds']
        p_bounds = self.config['bounds']['pressure_bounds']

        pop_size = self.ga_config.pop_size
        n_gen = self.ga_config.n_generations

        # Initialize population
        population = []
        for _ in range(pop_size):
            nt = np.random.randint(nt_bounds[0], nt_bounds[1] + 1)
            feed = np.random.randint(feed_bounds[0], min(feed_bounds[1], nt - 3) + 1)
            pressure = np.random.uniform(p_bounds[0], p_bounds[1])
            population.append([nt, feed, pressure])

        population = np.array(population)

        best_individual = None
        best_fitness = float('inf')
        best_result = None

        for gen in range(n_gen):
            # Evaluate population
            fitness = []
            for ind in population:
                nt, feed, pressure = int(ind[0]), int(ind[1]), ind[2]
                result = self.evaluator.evaluate(nt, feed, pressure, run_diagnostic_on_fail=True, rr_sweep_on_fail=True, purity_spec=self.purity_spec)
                self.eval_count += 1

                tac = result.get('TAC', 1e12)
                T_reb = result.get('T_reb')
                converged = result.get('converged', False)

                # If T_reb is None, cannot verify constraint -> treat as infeasible
                if T_reb is None:
                    T_reb = self.T_reb_max + 50  # Force constraint violation

                if converged and T_reb <= self.T_reb_max:
                    self.feasible_count += 1
                    fit = tac
                else:
                    self.infeasible_count += 1
                    fit = tac + 1e6 * max(0, T_reb - self.T_reb_max) ** 2

                fitness.append(fit)

                if fit < best_fitness:
                    best_fitness = fit
                    best_individual = ind.copy()
                    best_result = result

            fitness = np.array(fitness)
            self.convergence_history.append(best_fitness if best_fitness < 1e10 else None)

            _emit_progress(
                iteration=gen + 1,
                phase="ga_generation",
                current=gen + 1,
                total=n_gen,
                best_tac=best_fitness if best_fitness < 1e10 else None
            )

            # Selection (tournament)
            new_pop = []
            for _ in range(pop_size):
                i1, i2 = np.random.choice(pop_size, 2, replace=False)
                winner = population[i1] if fitness[i1] < fitness[i2] else population[i2]
                new_pop.append(winner.copy())

            # Crossover
            for i in range(0, pop_size - 1, 2):
                if np.random.random() < self.ga_config.crossover_prob:
                    alpha = np.random.random()
                    new_pop[i] = alpha * new_pop[i] + (1 - alpha) * new_pop[i + 1]
                    new_pop[i + 1] = (1 - alpha) * new_pop[i] + alpha * new_pop[i + 1]

            # Mutation
            for i in range(pop_size):
                if np.random.random() < 0.1:
                    j = np.random.randint(3)
                    if j == 0:
                        new_pop[i][j] += np.random.randint(-3, 4)
                    elif j == 1:
                        new_pop[i][j] += np.random.randint(-3, 4)
                    else:
                        new_pop[i][j] += np.random.uniform(-0.05, 0.05)

            population = np.array(new_pop)

            # Enforce bounds
            population[:, 0] = np.clip(population[:, 0], nt_bounds[0], nt_bounds[1])
            population[:, 1] = np.clip(population[:, 1], feed_bounds[0], feed_bounds[1])
            population[:, 2] = np.clip(population[:, 2], p_bounds[0], p_bounds[1])

            logger.info(f"Generation {gen + 1}: Best TAC = ${best_fitness:,.0f}")

        total_time = time.time() - start_time

        result = GAResult(
            optimal_nt=int(best_individual[0]) if best_individual is not None else 0,
            optimal_feed=int(best_individual[1]) if best_individual is not None else 0,
            optimal_pressure=round(best_individual[2], 4) if best_individual is not None else 0,
            optimal_tac=best_fitness,
            optimal_T_reb=(best_result.get('T_reb') or 0) if best_result else 0,
            convergence_history=[c for c in self.convergence_history if c is not None],
            total_time_seconds=total_time,
            total_evaluations=self.eval_count,
            feasible_evaluations=self.feasible_count,
            infeasible_evaluations=self.infeasible_count,
            pop_size=pop_size,
            n_generations=n_gen,
            case_name=case_name,
            timestamp=timestamp,
        )

        self.result = result

        _emit_progress(
            iteration=n_gen,
            phase="completed",
            best_tac=result.optimal_tac,
            message=f"GA complete! TAC=${result.optimal_tac:,.0f}/year"
        )

        return result

    def save_results(self, output_dir: str = "results") -> str:
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        result = self.result

        output = {
            'metadata': {'algorithm': 'GA (simple)'},
            'optimal': {
                'nt': result.optimal_nt,
                'feed': result.optimal_feed,
                'pressure': result.optimal_pressure,
                'tac': result.optimal_tac,
            },
            'statistics': {
                'total_evaluations': result.total_evaluations,
                'time_seconds': result.total_time_seconds,
            },
            'case_name': result.case_name,
            'algorithm': 'GA',
            'timestamp': result.timestamp,
        }

        filename = os.path.join(output_dir, f"ga_result_{result.timestamp}.json")
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {filename}")
        return filename


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GA Optimizer for Distillation Columns"
    )
    parser.add_argument("case", help="Case name (e.g., Case1_COL2)")
    parser.add_argument("--demo", action="store_true", help="Use demo evaluator")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--n-generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="results", help="Base output directory")
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to run-specific config JSON file (for multi-run safety)"
    )

    args = parser.parse_args()

    # Create run-specific output directory
    from config import create_run_output_dir
    run_output_dir = create_run_output_dir(f"{args.case}_GA", args.output)
    logger.info(f"Output directory: {run_output_dir}")

    # Load configuration - use config file if provided (multi-run safe)
    config = None
    if args.config_file:
        try:
            from config import load_run_config
            config = load_run_config(args.config_file)
            if config:
                logger.info(f"Using run-specific config from: {args.config_file}")
            else:
                logger.warning(f"Failed to load config file: {args.config_file}")
        except ImportError:
            logger.warning("Could not import load_run_config, falling back to get_case_config")

    # Fall back to get_case_config if no config file or loading failed
    if config is None:
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
                    'T_reb_max': T_REBOILER_MAX,
                }
        except ImportError:
            config = {
                'bounds': {
                    'nt_bounds': (20, 70),
                    'feed_bounds': (10, 60),
                    'pressure_bounds': (0.1, 0.5),
                },
                'min_section_stages': 3,
                'T_reb_max': T_REBOILER_MAX,
            }

    # Create evaluator
    if args.demo:
        from demo_evaluator import DemoEvaluator
        evaluator = DemoEvaluator(delay=0.02, seed=args.seed)
        logger.info("Using DEMO evaluator (no Aspen)")
    else:
        # Connect to Aspen Plus
        logger.info("Connecting to Aspen Plus...")
        from aspen_interface import AspenEnergyOptimizer
        from tac_calculator import TACCalculator
        from tac_evaluator import TACEvaluator

        aspen = AspenEnergyOptimizer(config['file_path'])

        if not aspen.connect_and_open(visible=True):
            logger.error("Failed to connect to Aspen Plus!")
            sys.exit(1)

        logger.info("Connected to Aspen Plus successfully!")
        aspen.get_column_info(config['column']['block_name'])

        tac_calc = TACCalculator(
            material='SS',
            cepci=800,
            cepci_base=500,
            operating_hours=8000,
            payback_period=3,
        )

        evaluator = TACEvaluator(
            aspen_interface=aspen,
            tac_calculator=tac_calc,
            block_name=config['column']['block_name'],
            feed_stream=config['column']['feed_stream'],
        )

    # Create GA config
    ga_config = GAConfig(
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        seed=args.seed,
    )

    # Load purity spec for this case (design target)
    try:
        from config import get_purity_spec
        purity_spec = get_purity_spec(args.case)
        if purity_spec:
            logger.info(f"Purity spec loaded: stream={purity_spec.get('stream')}, "
                       f"component={purity_spec.get('component')}, target={purity_spec.get('target')}")
        else:
            logger.warning(f"No purity spec found for {args.case}")
    except ImportError:
        purity_spec = None
        logger.warning("Could not import get_purity_spec from config")

    # Track aspen for cleanup
    aspen_instance = None
    if not args.demo:
        aspen_instance = aspen

    try:
        # Run optimization
        if PYMOO_AVAILABLE:
            optimizer = GAOptimizer(evaluator, config, ga_config, purity_spec=purity_spec)
        else:
            logger.warning("pymoo not available, using simple GA")
            optimizer = SimpleGAOptimizer(evaluator, config, ga_config, purity_spec=purity_spec)

        result = optimizer.run(args.case)

        # Save results
        filename = optimizer.save_results(run_output_dir)

        # Generate convergence plots
        try:
            from visualization_metaheuristic import MetaheuristicVisualizer
            visualizer = MetaheuristicVisualizer(run_output_dir)

            # Convergence plot
            conv_plot = visualizer.plot_convergence(
                result.convergence_history,
                algorithm="GA",
                case_name=args.case,
                optimal_tac=result.optimal_tac,
                n_iterations=result.n_generations,
            )
            if conv_plot:
                logger.info(f"Convergence plot saved: {conv_plot}")
                print(f"Convergence plot saved: {conv_plot}")

            # Summary plot
            summary_data = {
                'optimal': {
                    'nt': result.optimal_nt,
                    'feed': result.optimal_feed,
                    'pressure': result.optimal_pressure,
                    'tac': result.optimal_tac,
                    'T_reb': result.optimal_T_reb,
                },
                'convergence': {'history': result.convergence_history},
                'statistics': {
                    'total_evaluations': result.total_evaluations,
                    'feasible': result.feasible_evaluations,
                    'infeasible': result.infeasible_evaluations,
                    'time_seconds': result.total_time_seconds,
                },
            }
            summary_plot = visualizer.plot_solution_summary(
                summary_data, algorithm="GA", case_name=args.case
            )
            if summary_plot:
                logger.info(f"Summary plot saved: {summary_plot}")
                print(f"Summary plot saved: {summary_plot}")

        except ImportError as e:
            logger.warning(f"Could not generate plots (matplotlib not available): {e}")
        except Exception as e:
            logger.warning(f"Error generating plots: {e}")

        # Generate Reflux Ratio vs Purity curve at optimal point
        if not args.demo and purity_spec and aspen_instance:
            try:
                logger.info("")
                logger.info("=" * 70)
                logger.info("POST-OPTIMIZATION: RR vs PURITY SWEEP")
                logger.info("=" * 70)
                logger.info(f"  Optimal config: NT={result.optimal_nt}, NF={result.optimal_feed}, "
                           f"P={result.optimal_pressure:.4f} bar")

                rr_sweep_data = aspen_instance.sweep_rr_purity(
                    block_name=config['column']['block_name'],
                    nt=result.optimal_nt,
                    feed=result.optimal_feed,
                    pressure=result.optimal_pressure,
                    feed_stream=config['column']['feed_stream'],
                    rr_range=(0.5, 5.0),
                    num_points=20,
                    purity_spec=purity_spec
                )

                if rr_sweep_data:
                    # Save RR sweep data to JSON
                    rr_sweep_file = os.path.join(run_output_dir, f"{args.case}_GA_rr_vs_purity.json")
                    with open(rr_sweep_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'case_name': args.case,
                            'algorithm': 'GA',
                            'optimal': {
                                'nt': result.optimal_nt,
                                'feed': result.optimal_feed,
                                'pressure': result.optimal_pressure,
                                'tac': result.optimal_tac,
                            },
                            'purity_spec': purity_spec,
                            'rr_sweep': rr_sweep_data,
                        }, f, indent=2)
                    logger.info(f"RR sweep data saved: {rr_sweep_file}")

                    # Generate plot
                    from visualization_metaheuristic import MetaheuristicVisualizer
                    visualizer = MetaheuristicVisualizer(run_output_dir)
                    rr_plot = visualizer.plot_rr_vs_purity(
                        rr_sweep_data,
                        algorithm="GA",
                        case_name=args.case,
                        purity_target=purity_spec.get('target'),
                        optimal_config={
                            'nt': result.optimal_nt,
                            'feed': result.optimal_feed,
                            'pressure': result.optimal_pressure,
                        }
                    )
                    if rr_plot:
                        logger.info(f"RR vs Purity plot saved: {rr_plot}")
                        print(f"RR vs Purity plot saved: {rr_plot}")

            except Exception as e:
                logger.warning(f"RR sweep failed: {e}")
                import traceback
                traceback.print_exc()

        return result

    finally:
        # Cleanup: Close Aspen connection
        if aspen_instance:
            try:
                aspen_instance.close()
                logger.info("Aspen Plus connection closed.")
            except Exception as e:
                logger.warning(f"Error closing Aspen: {e}")


if __name__ == "__main__":
    main()
