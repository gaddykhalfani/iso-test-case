"""
pso_optimizer.py
================
Particle Swarm Optimization (PSO) for Distillation Column Design

Implements PSO to find optimal (NT, Feed, Pressure) that minimizes TAC
while respecting the temperature constraint T_reb <= 120C.

Usage:
    python pso_optimizer.py Case1_COL2 --demo
    python pso_optimizer.py Case1_COL2 --n-particles 20 --n-iterations 50

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
from dataclasses import dataclass, field, asdict
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
        "algorithm": "PSO",
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
class PSOConfig:
    """PSO algorithm configuration."""
    n_particles: int = 20
    n_iterations: int = 50
    w: float = 0.7          # Inertia weight
    w_min: float = 0.4      # Minimum inertia (for adaptive)
    w_max: float = 0.9      # Maximum inertia (for adaptive)
    c1: float = 1.5         # Cognitive coefficient
    c2: float = 1.5         # Social coefficient
    v_max_ratio: float = 0.2  # Max velocity as ratio of range
    seed: Optional[int] = None


@dataclass
class PSOResult:
    """PSO optimization result."""
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
    best_positions_history: List[Dict] = field(default_factory=list)

    # Statistics
    total_time_seconds: float = 0.0
    total_evaluations: int = 0
    feasible_evaluations: int = 0
    infeasible_evaluations: int = 0

    # Configuration used
    n_particles: int = 0
    n_iterations: int = 0

    # Metadata
    case_name: str = ""
    algorithm: str = "PSO"
    timestamp: str = ""


# ════════════════════════════════════════════════════════════════════════════
# PSO OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════

class PSOOptimizer:
    """
    Particle Swarm Optimization for Distillation Column Design.

    Optimizes (NT, Feed, Pressure) to minimize TAC with constraint T_reb <= 120C.

    Parameters
    ----------
    evaluator : TACEvaluator or DemoEvaluator
        Evaluator that computes TAC for a given configuration
    config : dict
        Configuration with bounds and PSO parameters
    pso_config : PSOConfig, optional
        PSO algorithm parameters
    """

    def __init__(self, evaluator, config: dict, pso_config: Optional[PSOConfig] = None):
        self.evaluator = evaluator
        self.config = config
        self.pso_config = pso_config or PSOConfig()

        # Extract bounds
        self.nt_bounds = config['bounds']['nt_bounds']
        self.feed_bounds = config['bounds']['feed_bounds']
        self.pressure_bounds = config['bounds']['pressure_bounds']

        # Temperature constraint
        self.T_reb_max = config.get('T_reb_max', T_REBOILER_MAX)

        # Minimum section stages
        self.min_section_stages = config.get('min_section_stages', 3)

        # Statistics
        self.eval_count = 0
        self.feasible_count = 0
        self.infeasible_count = 0

        # Set random seed
        if self.pso_config.seed is not None:
            np.random.seed(self.pso_config.seed)

    def run(self, case_name: str = "Case") -> PSOResult:
        """
        Run PSO optimization.

        Parameters
        ----------
        case_name : str
            Name for this optimization case

        Returns
        -------
        PSOResult
            Optimization results
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("=" * 70)
        logger.info("PARTICLE SWARM OPTIMIZATION (PSO)")
        logger.info("=" * 70)
        logger.info(f"Case: {case_name}")
        logger.info(f"Particles: {self.pso_config.n_particles}")
        logger.info(f"Iterations: {self.pso_config.n_iterations}")
        logger.info(f"Temperature constraint: T_reb <= {self.T_reb_max}C")
        logger.info("=" * 70)

        _emit_progress(0, "initialization", message="Initializing PSO swarm")

        # ════════════════════════════════════════════════════════════════════
        # INITIALIZATION
        # ════════════════════════════════════════════════════════════════════

        n_particles = self.pso_config.n_particles
        n_iterations = self.pso_config.n_iterations

        # Bounds as arrays
        lb = np.array([self.nt_bounds[0], self.feed_bounds[0], self.pressure_bounds[0]])
        ub = np.array([self.nt_bounds[1], self.feed_bounds[1], self.pressure_bounds[1]])
        ranges = ub - lb

        # Initialize particles (position and velocity)
        positions = np.random.uniform(lb, ub, (n_particles, 3))
        velocities = np.random.uniform(-ranges * 0.1, ranges * 0.1, (n_particles, 3))

        # Velocity limits
        v_max = ranges * self.pso_config.v_max_ratio

        # Personal best positions and fitness
        pbest_positions = positions.copy()
        pbest_fitness = np.full(n_particles, float('inf'))

        # Global best
        gbest_position = positions[0].copy()
        gbest_fitness = float('inf')
        gbest_result = None

        # Convergence history
        convergence_history = []
        best_positions_history = []

        # ════════════════════════════════════════════════════════════════════
        # MAIN PSO LOOP
        # ════════════════════════════════════════════════════════════════════

        for iteration in range(n_iterations):
            # Adaptive inertia weight (decreases over time)
            w = self.pso_config.w_max - (
                (self.pso_config.w_max - self.pso_config.w_min) * iteration / n_iterations
            )

            _emit_progress(
                iteration=iteration + 1,
                phase="pso_iteration",
                current=iteration + 1,
                total=n_iterations,
                best_tac=gbest_fitness if gbest_fitness < 1e10 else None,
                message=f"Iteration {iteration + 1}/{n_iterations}"
            )

            # Evaluate all particles
            for i in range(n_particles):
                pos = positions[i]

                # Decode position
                nt = int(round(pos[0]))
                feed = int(round(pos[1]))
                pressure = pos[2]

                # Enforce bounds
                nt = max(self.nt_bounds[0], min(nt, self.nt_bounds[1]))
                feed = max(self.min_section_stages + 1, min(feed, nt - self.min_section_stages))
                pressure = max(self.pressure_bounds[0], min(pressure, self.pressure_bounds[1]))

                # Evaluate
                fitness, result = self._evaluate_fitness(nt, feed, pressure)

                # Update personal best
                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = positions[i].copy()

                # Update global best
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = positions[i].copy()
                    gbest_result = result

                    logger.info(f"  New best: NT={nt}, Feed={feed}, P={pressure:.4f}, TAC=${fitness:,.0f}")

            # Update velocities and positions
            r1 = np.random.random((n_particles, 3))
            r2 = np.random.random((n_particles, 3))

            cognitive = self.pso_config.c1 * r1 * (pbest_positions - positions)
            social = self.pso_config.c2 * r2 * (gbest_position - positions)

            velocities = w * velocities + cognitive + social

            # Clamp velocities
            velocities = np.clip(velocities, -v_max, v_max)

            # Update positions
            positions = positions + velocities

            # Clamp positions to bounds
            positions = np.clip(positions, lb, ub)

            # Store convergence history
            convergence_history.append(gbest_fitness if gbest_fitness < 1e10 else None)
            best_positions_history.append({
                'iteration': iteration + 1,
                'nt': int(round(gbest_position[0])),
                'feed': int(round(gbest_position[1])),
                'pressure': round(gbest_position[2], 4),
                'tac': gbest_fitness if gbest_fitness < 1e10 else None,
            })

            logger.info(f"Iteration {iteration + 1}: Best TAC = ${gbest_fitness:,.0f}")

        # ════════════════════════════════════════════════════════════════════
        # BUILD RESULT
        # ════════════════════════════════════════════════════════════════════

        total_time = time.time() - start_time

        # Extract optimal values
        optimal_nt = int(round(gbest_position[0]))
        optimal_feed = int(round(gbest_position[1]))
        optimal_pressure = gbest_position[2]

        # Ensure valid feed
        optimal_feed = max(self.min_section_stages + 1,
                          min(optimal_feed, optimal_nt - self.min_section_stages))

        result = PSOResult(
            optimal_nt=optimal_nt,
            optimal_feed=optimal_feed,
            optimal_pressure=round(optimal_pressure, 4),
            optimal_tac=gbest_fitness,
            optimal_T_reb=(gbest_result.get('T_reb') or 0) if gbest_result else 0,
            optimal_tpc=(gbest_result.get('TPC') or 0) if gbest_result else 0,
            optimal_toc=(gbest_result.get('TOC') or 0) if gbest_result else 0,
            convergence_history=[c for c in convergence_history if c is not None],
            best_positions_history=best_positions_history,
            total_time_seconds=total_time,
            total_evaluations=self.eval_count,
            feasible_evaluations=self.feasible_count,
            infeasible_evaluations=self.infeasible_count,
            n_particles=n_particles,
            n_iterations=n_iterations,
            case_name=case_name,
            timestamp=timestamp,
        )

        self._print_summary(result)

        # Emit completion
        _emit_progress(
            iteration=n_iterations,
            phase="completed",
            current=n_iterations,
            total=n_iterations,
            best_tac=result.optimal_tac,
            message=f"PSO complete! TAC=${result.optimal_tac:,.0f}/year",
            optimal_nt=result.optimal_nt,
            optimal_feed=result.optimal_feed,
            optimal_pressure=result.optimal_pressure
        )

        return result

    def _evaluate_fitness(self, nt: int, feed: int, pressure: float) -> Tuple[float, Dict]:
        """
        Evaluate fitness with constraint handling.

        Uses penalty method for temperature constraint.
        """
        self.eval_count += 1

        result = self.evaluator.evaluate(nt, feed, pressure)

        tac = result.get('TAC', float('inf'))
        T_reb = result.get('T_reb') or 0  # Handle None from failed simulations
        converged = result.get('converged', False)

        # Penalty for constraint violations
        penalty = 0

        if not converged:
            penalty = 1e12
            self.infeasible_count += 1
        elif T_reb > self.T_reb_max:
            # Quadratic penalty for temperature violation
            penalty = 1e6 * (T_reb - self.T_reb_max) ** 2
            self.infeasible_count += 1
        else:
            self.feasible_count += 1

        fitness = tac + penalty

        return fitness, result

    def _print_summary(self, result: PSOResult):
        """Print optimization summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("PSO OPTIMIZATION COMPLETE")
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
                'methodology': 'Particle Swarm Optimization (PSO)',
                'algorithm': 'PSO',
                'temperature_constraint': f'T_reb <= {self.T_reb_max}C',
                'n_particles': result.n_particles,
                'n_iterations': result.n_iterations,
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
                'best_positions': result.best_positions_history,
            },
            'statistics': {
                'total_evaluations': result.total_evaluations,
                'feasible': result.feasible_evaluations,
                'infeasible': result.infeasible_evaluations,
                'time_seconds': result.total_time_seconds,
            },
            'case_name': result.case_name,
            'algorithm': 'PSO',
            'timestamp': result.timestamp,
        }

        filename = os.path.join(output_dir, f"pso_result_{result.timestamp}.json")

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Results saved to: {filename}")
        print(f"Results saved to: {filename}")

        return filename


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PSO Optimizer for Distillation Columns"
    )
    parser.add_argument("case", help="Case name (e.g., Case1_COL2)")
    parser.add_argument("--demo", action="store_true", help="Use demo evaluator")
    parser.add_argument("--n-particles", type=int, default=20, help="Number of particles")
    parser.add_argument("--n-iterations", type=int, default=50, help="Number of iterations")
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
    run_output_dir = create_run_output_dir(f"{args.case}_PSO", args.output)
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
                # Use default config
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
        evaluator = DemoEvaluator(delay=0.05, seed=args.seed)
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

    # Create PSO config
    pso_config = PSOConfig(
        n_particles=args.n_particles,
        n_iterations=args.n_iterations,
        seed=args.seed,
    )

    # Track aspen for cleanup
    aspen_instance = None
    if not args.demo:
        aspen_instance = aspen

    try:
        # Run optimization
        optimizer = PSOOptimizer(evaluator, config, pso_config)
        result = optimizer.run(args.case)
        optimizer.result = result

        # Save results
        filename = optimizer.save_results(run_output_dir)

        # Generate convergence plots
        try:
            from visualization_metaheuristic import MetaheuristicVisualizer
            visualizer = MetaheuristicVisualizer(run_output_dir)

            # Convergence plot
            conv_plot = visualizer.plot_convergence(
                result.convergence_history,
                algorithm="PSO",
                case_name=args.case,
                optimal_tac=result.optimal_tac,
                n_iterations=result.n_iterations,
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
                summary_data, algorithm="PSO", case_name=args.case
            )
            if summary_plot:
                logger.info(f"Summary plot saved: {summary_plot}")
                print(f"Summary plot saved: {summary_plot}")

        except ImportError as e:
            logger.warning(f"Could not generate plots (matplotlib not available): {e}")
        except Exception as e:
            logger.warning(f"Error generating plots: {e}")

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
