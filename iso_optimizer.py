# iso_optimizer.py
"""
TRUE Iterative Sequential Optimization (ISO) for Distillation Columns
======================================================================

Implements the professor's requirements:
1. Optimize variables ONE AT A TIME: P -> NT -> NF
2. Outer iteration loop with convergence check
3. Temperature constraint: T_reb <= 120C (polymerization risk)
4. Infeasibility labeling for constrained points
5. Reflux diagnostic for unconverged cases

METHODOLOGY (Physical Justification):
====================================

OUTER LOOP: Design-level iterations until convergence
    |
    ├── STEP 1: Optimize PRESSURE (Strategic Decision)
    |   * Fix NT, NF from previous iteration
    |   * Sweep P with temperature constraint
    |   * Find P* that minimizes feasible TAC
    |
    ├── STEP 2: Optimize NT (Capital vs Energy Trade-off)
    |   * Fix P = P*, NF from previous iteration
    |   * Sweep NT to generate U-curve
    |   * Find NT* at minimum of U-curve
    |
    └── STEP 3: Optimize NF (Feed Location)
        * Fix P = P*, NT = NT*
        * Sweep NF to generate U-curve
        * Find NF* at minimum of U-curve

CONVERGENCE: When (P*, NT*, NF*) unchanged between iterations

Author: PSE Lab, NTUST
Version: 5.0 - True ISO with Temperature Constraint
"""

import logging
import time
import math
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# PROGRESS LOGGING FOR DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def _emit_progress(iteration: int, phase: str, current: int = 0, total: int = 0,
                   best_tac: float = None, message: str = None, **kwargs):
    """
    Emit structured progress JSON for dashboard live tracking.

    Format: [PROGRESS] {"iteration": 1, "phase": "pressure_sweep", ...}
    """
    progress = {
        "iteration": iteration,
        "phase": phase,
        "current": current,
        "total": total,
        "algorithm": "ISO",
    }
    if best_tac is not None and best_tac < 1e10:
        progress["best_tac"] = round(best_tac, 2)
    if message:
        progress["message"] = message
    progress.update(kwargs)

    print(f"[PROGRESS] {json.dumps(progress)}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# CRITICAL: Styrene polymerization temperature limit
T_REBOILER_MAX = 120.0  # C - Professor's hard constraint

# Convergence criteria
TAC_TOLERANCE = 100  # $/year - TAC change threshold
MAX_OUTER_ITERATIONS = 10  # Safety limit

# U-curve validation criteria
MIN_CONSECUTIVE_FEASIBLE = 3  # Minimum consecutive feasible NT points required
U_CURVE_NEIGHBOR_CHECK = True  # Require neighbors to have higher TAC (proper minimum)


# ════════════════════════════════════════════════════════════════════════════
# FEASIBILITY STATUS
# ════════════════════════════════════════════════════════════════════════════

class FeasibilityStatus(Enum):
    """Feasibility status for evaluated points.

    FEASIBLE (hard feasible): Naturally converged with Aspen Design Specs
    SOFT_FEASIBLE: RR-recovered point - thermodynamically possible but solver couldn't
                   find it naturally. These are valid but should be deprioritized
                   in favor of naturally converged points.
    """
    FEASIBLE = "feasible"              # Hard feasible - naturally converged
    SOFT_FEASIBLE = "soft_feasible"    # RR-recovered - valid but deprioritized
    INFEASIBLE_TEMPERATURE = "infeasible_temperature"
    INFEASIBLE_CONVERGENCE = "infeasible_convergence"
    INFEASIBLE_SEPARATION = "infeasible_separation"


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES FOR RESULTS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationPoint:
    """Single evaluation point with feasibility status."""
    nt: int
    feed: int
    pressure: float
    tac: float
    tpc: float = 0
    toc: float = 0
    q_reb: float = 0
    q_cond: float = 0
    diameter: float = 0
    T_reb: float = None
    T_cond: float = None
    converged: bool = False
    feasibility: FeasibilityStatus = FeasibilityStatus.FEASIBLE
    infeasibility_reason: str = ""
    reflux_ratio: float = None


@dataclass
class SweepResult:
    """Result from a 1D parameter sweep."""
    parameter_name: str  # 'pressure', 'nt', or 'feed'
    fixed_values: Dict  # e.g., {'nt': 45, 'feed': 22}
    points: List[EvaluationPoint] = field(default_factory=list)
    optimal_value: float = 0
    optimal_tac: float = float('inf')


@dataclass
class ISOIterationResult:
    """Result from one complete ISO iteration."""
    iteration: int
    pressure_sweep: SweepResult
    nt_sweep: SweepResult
    feed_sweep: SweepResult
    optimal_pressure: float
    optimal_nt: int
    optimal_feed: int
    optimal_tac: float


@dataclass
class ISOResult:
    """Complete ISO optimization result."""
    # Final optimal values
    optimal_nt: int
    optimal_feed: int
    optimal_pressure: float
    optimal_tac: float
    
    # All iterations
    iterations: List[ISOIterationResult] = field(default_factory=list)
    
    # Convergence info
    converged: bool = False
    convergence_iteration: int = 0
    
    # Statistics
    total_time_seconds: float = 0
    total_evaluations: int = 0
    feasible_evaluations: int = 0
    infeasible_evaluations: int = 0

    # RR sweep data from failed evaluations (infeasible designs)
    failed_rr_sweeps: List[Dict] = field(default_factory=list)

    # Metadata
    case_name: str = ""
    timestamp: str = ""


# ════════════════════════════════════════════════════════════════════════════
# ISO OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════

class ISOOptimizer:
    """
    TRUE Iterative Sequential Optimization for Distillation Columns.
    
    Implements professor's requirements:
    1. Variables optimized ONE AT A TIME
    2. Outer iteration loop with convergence check
    3. Temperature constraint: T_reb <= 120C
    4. Infeasibility labeling
    """
    
    def __init__(self, evaluator, config, purity_spec=None):
        """
        Initialize ISO optimizer.

        Parameters
        ----------
        evaluator : TACEvaluator
            The TAC evaluator instance connected to Aspen
        config : dict
            Configuration dictionary with bounds and settings
        purity_spec : dict, optional
            Purity specification from config.PURITY_SPECS for diagnostic features
        """
        self.evaluator = evaluator
        self.config = config
        self.purity_spec = purity_spec

        # Extract bounds
        self.nt_bounds = config['bounds']['nt_bounds']
        self.feed_bounds = config['bounds']['feed_bounds']
        self.pressure_bounds = config['bounds']['pressure_bounds']

        # Sweep settings
        self.pressure_points = config.get('pressure_points', 9)
        self.nt_step = config.get('nt_step', 1)
        self.feed_step = config.get('feed_step', 1)
        self.min_section_stages = config.get('min_section_stages', 3)

        # Temperature constraint: applies to ANY column with styrene in its feed
        # (Styrene polymerizes above ~120°C on hot stages — even if SM goes to
        #  distillate, intermediate stages can have SM at elevated temperatures)
        # Default True (conservative); auto-detected from feed stream after baseline
        self.has_styrene = True  # Conservative default, overridden by auto-detect
        self.T_reb_max = config.get('T_reb_max', T_REBOILER_MAX)
        
        # Convergence settings
        self.tac_tolerance = config.get('tac_tolerance', TAC_TOLERANCE)
        self.max_iterations = config.get('max_iterations', MAX_OUTER_ITERATIONS)
        
        # Results storage
        self.iterations = []
        self.all_evaluations = []
        self.cache = {}
        
        # Statistics
        self.eval_count = 0
        self.feasible_count = 0       # Hard feasible (naturally converged)
        self.soft_feasible_count = 0  # Soft feasible (RR-recovered)
        self.infeasible_count = 0
        self.start_time = None

        # RR sweep data from failed evaluations (for infeasible design visualization)
        self.failed_rr_sweeps = []

        # Baseline TAC recording (for improvement metrics)
        self.baseline_result = None
        self.baseline_tac = None

        # Global best tracking across all iterations
        # (Prevents losing a good solution when ISO oscillates)
        self.global_best_tac = float('inf')
        self.global_best_config = None  # (nt, feed, pressure)
        self.global_best_iteration = 0
    
    def run(self, case_name: str = "Case") -> ISOResult:
        """
        Run TRUE Iterative Sequential Optimization.
        
        Sequence: P -> NT -> NF, repeated until convergence.
        
        Parameters
        ----------
        case_name : str
            Name for this optimization case
            
        Returns
        -------
        ISOResult : Complete results with all iterations
        """
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._print_header(case_name)
        
        # ════════════════════════════════════════════════════════════════════
        # INITIALIZATION
        # ════════════════════════════════════════════════════════════════════
        
        # Start from initial guesses (midpoints)
        current_nt = self.config.get('initial', {}).get('nt', 
                     (self.nt_bounds[0] + self.nt_bounds[1]) // 2)
        current_feed = self.config.get('initial', {}).get('feed',
                       current_nt // 2)
        current_pressure = self.config.get('initial', {}).get('pressure',
                           (self.pressure_bounds[0] + self.pressure_bounds[1]) / 2)
        current_tac = float('inf')
        
        # Ensure valid initial feed
        current_feed = max(current_feed, self.min_section_stages + 1)
        current_feed = min(current_feed, current_nt - self.min_section_stages)
        
        logger.info("")
        logger.info(f"INITIAL: NT={current_nt}, NF={current_feed}, P={current_pressure:.4f} bar")

        # ════════════════════════════════════════════════════════════════════
        # BASELINE TAC EVALUATION
        # ════════════════════════════════════════════════════════════════════

        logger.info("")
        logger.info("=" * 50)
        logger.info("EVALUATING BASELINE TAC (Initial Configuration)")
        logger.info("=" * 50)

        _emit_progress(
            iteration=0,
            phase="baseline_evaluation",
            message="Evaluating baseline TAC at initial configuration"
        )

        baseline_result = self._evaluate_with_feasibility(
            current_nt, current_feed, current_pressure
        )
        self.baseline_result = baseline_result

        # Auto-detect styrene from FEED stream — always run, independent of baseline convergence
        try:
            feed_stream = self.config['column']['feed_stream']
            self.has_styrene = self.evaluator.aspen.auto_detect_styrene_in_feed(feed_stream)
            logger.info("  [OK] Styrene auto-detect from feed '{}': has_styrene={}".format(
                feed_stream, self.has_styrene))
        except Exception as e:
            logger.warning("  Could not auto-detect styrene from feed: {}".format(e))
            logger.warning("  Using conservative default: has_styrene=True")

        if baseline_result.converged and baseline_result.feasibility in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.SOFT_FEASIBLE):
            self.baseline_tac = baseline_result.tac
            logger.info(f"  BASELINE TAC: ${self.baseline_tac:,.0f}/year")
            logger.info(f"  Configuration: NT={current_nt}, NF={current_feed}, P={current_pressure:.4f} bar")
            if baseline_result.feasibility == FeasibilityStatus.SOFT_FEASIBLE:
                logger.warning("  NOTE: Baseline is SOFT FEASIBLE (RR-recovered)")
        else:
            self.baseline_tac = None
            logger.warning("  Baseline did not converge or is infeasible")
            logger.warning(f"  Reason: {baseline_result.feasibility.value}")
            logger.info("  Proceeding without baseline comparison")

        logger.info("=" * 50)

        converged = False
        prev_optimal_pressure = None  # Track previous iteration's best P for carry-forward

        # ════════════════════════════════════════════════════════════════════
        # OUTER ITERATION LOOP
        # ════════════════════════════════════════════════════════════════════
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info("")
            logger.info("=" * 70)
            logger.info("=" * 70)
            logger.info(f"ISO ITERATION {iteration}")
            logger.info(f"START Condition: P={current_pressure:.4f}, NT={current_nt}, NF={current_feed}")
            logger.info("=" * 70)

            # Emit progress for dashboard
            _emit_progress(
                iteration=iteration,
                phase="iteration_start",
                current=iteration,
                total=self.max_iterations,
                best_tac=current_tac if current_tac < float('inf') else None,
                message=f"Starting iteration {iteration}"
            )
            
            prev_nt = current_nt
            prev_feed = current_feed
            prev_pressure = current_pressure
            prev_tac = current_tac
            
            # ────────────────────────────────────────────────────────────────
            # STEP 1: OPTIMIZE PRESSURE
            # ────────────────────────────────────────────────────────────────
            
            logger.info("")
            logger.info("-" * 50)
            logger.info(f"STEP 1: Optimize PRESSURE (NT={current_nt}, NF={current_feed} fixed)")
            logger.info("-" * 50)

            _emit_progress(
                iteration=iteration,
                phase="pressure_sweep_start",
                message=f"Sweeping pressure (NT={current_nt}, NF={current_feed} fixed)"
            )

            pressure_sweep = self._sweep_pressure(current_nt, current_feed, iteration, prev_optimal_pressure=prev_optimal_pressure)

            if pressure_sweep.optimal_tac < float('inf'):
                current_pressure = pressure_sweep.optimal_value
                logger.info(f"  -> P* = {current_pressure:.4f} bar, TAC=${pressure_sweep.optimal_tac:,.0f}")
                # Log feasible/infeasible counts from this sweep (both hard and soft feasible)
                hard_feasible_pts = sum(1 for p in pressure_sweep.points if p.feasibility == FeasibilityStatus.FEASIBLE)
                soft_feasible_pts = sum(1 for p in pressure_sweep.points if p.feasibility == FeasibilityStatus.SOFT_FEASIBLE)
                infeasible_pts = len(pressure_sweep.points) - hard_feasible_pts - soft_feasible_pts
                logger.info(f"     ({hard_feasible_pts} hard feasible, {soft_feasible_pts} soft feasible, {infeasible_pts} infeasible out of {len(pressure_sweep.points)} points)")
            else:
                logger.warning("  No feasible pressure found! Keeping previous pressure.")
                logger.warning(f"  All {len(pressure_sweep.points)} points were infeasible")
            
            # ────────────────────────────────────────────────────────────────
            # STEP 2: OPTIMIZE NT
            # ────────────────────────────────────────────────────────────────
            
            logger.info("")
            logger.info("-" * 50)
            logger.info(f"STEP 2: Optimize NT (P={current_pressure:.4f}, NF={current_feed} fixed)")
            logger.info("-" * 50)

            _emit_progress(
                iteration=iteration,
                phase="nt_sweep_start",
                message=f"Sweeping NT (P={current_pressure:.4f}, NF={current_feed} fixed)"
            )

            nt_sweep = self._sweep_nt(current_pressure, current_feed, iteration)

            if nt_sweep.optimal_tac < float('inf'):
                current_nt = int(nt_sweep.optimal_value)
                # Update feed if it's now invalid for new NT
                current_feed = self._adjust_feed_for_nt(current_feed, current_nt)
                logger.info(f"  -> NT* = {current_nt}")

                # ────────────────────────────────────────────────────────────────
                # U-CURVE VALIDATION CHECK
                # ────────────────────────────────────────────────────────────────
                is_valid, consec_count, reason = self._validate_u_curve_quality(
                    current_pressure, current_feed
                )

                if is_valid:
                    logger.info(f"  [OK] U-curve validation passed ({consec_count} consecutive feasible points)")
                else:
                    logger.warning(f"  [!] U-curve validation FAILED: {reason}")
                    logger.warning(f"      This may indicate an edge-case optimum at P={current_pressure:.4f} bar")

                    # Try to find alternative pressure with valid U-curve
                    logger.info("  Searching for alternative pressure with valid U-curve...")

                    # Get feasible pressures sorted by TAC (prefer hard feasible, then soft feasible)
                    # First get hard feasible pressures
                    hard_feasible_pressures = [
                        (p.pressure, p.tac, 'hard')
                        for p in pressure_sweep.points
                        if p.feasibility == FeasibilityStatus.FEASIBLE and p.pressure != current_pressure
                    ]
                    # Then soft feasible pressures
                    soft_feasible_pressures = [
                        (p.pressure, p.tac, 'soft')
                        for p in pressure_sweep.points
                        if p.feasibility == FeasibilityStatus.SOFT_FEASIBLE and p.pressure != current_pressure
                    ]
                    # Combine: hard feasible first (sorted by TAC), then soft feasible (sorted by TAC)
                    hard_feasible_pressures.sort(key=lambda x: x[1])
                    soft_feasible_pressures.sort(key=lambda x: x[1])
                    feasible_pressures = [(p, t) for p, t, _ in hard_feasible_pressures] + [(p, t) for p, t, _ in soft_feasible_pressures]

                    found_valid = False
                    for alt_pressure, alt_tac in feasible_pressures:
                        # Do a quick NT sweep at this alternative pressure
                        logger.info(f"    Testing P={alt_pressure:.4f} bar...")
                        alt_nt_sweep = self._sweep_nt(alt_pressure, current_feed, iteration)

                        if alt_nt_sweep.optimal_tac < float('inf'):
                            # Check U-curve quality
                            alt_valid, alt_count, alt_reason = self._validate_u_curve_quality(
                                alt_pressure, current_feed
                            )

                            if alt_valid:
                                logger.info(f"    [OK] Found valid U-curve at P={alt_pressure:.4f} bar")
                                logger.info(f"         TAC=${alt_nt_sweep.optimal_tac:,.0f} vs edge-case TAC=${nt_sweep.optimal_tac:,.0f}")

                                # Check if the TAC difference is acceptable (within 50% of edge case)
                                tac_ratio = alt_nt_sweep.optimal_tac / nt_sweep.optimal_tac
                                if tac_ratio < 1.5:  # Within 50% is acceptable
                                    current_pressure = alt_pressure
                                    current_nt = int(alt_nt_sweep.optimal_value)
                                    current_feed = self._adjust_feed_for_nt(current_feed, current_nt)
                                    nt_sweep = alt_nt_sweep  # Update sweep for later use
                                    logger.info(f"    -> SWITCHED to P={current_pressure:.4f}, NT={current_nt}")
                                    found_valid = True
                                    break
                                else:
                                    logger.info(f"    TAC too high ({tac_ratio:.1f}x), continuing search...")
                            else:
                                logger.info(f"    U-curve invalid: {alt_reason}")

                    if not found_valid:
                        logger.warning("  No alternative pressure found with valid U-curve")
                        logger.warning("  Proceeding with edge-case pressure (results may be unstable)")
            else:
                logger.warning("  No feasible NT found!")
            
            # ────────────────────────────────────────────────────────────────
            # STEP 3: OPTIMIZE FEED
            # ────────────────────────────────────────────────────────────────
            
            logger.info("")
            logger.info("-" * 50)
            logger.info(f"STEP 3: Optimize NF (P={current_pressure:.4f}, NT={current_nt} fixed)")
            logger.info("-" * 50)

            _emit_progress(
                iteration=iteration,
                phase="feed_sweep_start",
                message=f"Sweeping feed (P={current_pressure:.4f}, NT={current_nt} fixed)"
            )

            feed_sweep = self._sweep_feed(current_pressure, current_nt, iteration)
            
            if feed_sweep.optimal_tac < float('inf'):
                current_feed = int(feed_sweep.optimal_value)
                current_tac = feed_sweep.optimal_tac
                logger.info(f"  -> NF* = {current_feed}")
            else:
                logger.warning("  No feasible feed found!")

            # Log iteration state summary
            logger.info("")
            logger.info(f"  [STATE] After iteration {iteration}: P={current_pressure:.4f} bar, NT={current_nt}, NF={current_feed}, TAC=${current_tac:,.0f}")

            # ────────────────────────────────────────────────────────────────
            # STORE ITERATION RESULT
            # ────────────────────────────────────────────────────────────────
            
            iter_result = ISOIterationResult(
                iteration=iteration,
                pressure_sweep=pressure_sweep,
                nt_sweep=nt_sweep,
                feed_sweep=feed_sweep,
                optimal_pressure=current_pressure,
                optimal_nt=current_nt,
                optimal_feed=current_feed,
                optimal_tac=current_tac,
            )
            self.iterations.append(iter_result)

            # Update carry-forward pressure for next iteration
            prev_optimal_pressure = current_pressure

            # Update global best tracking
            if current_tac < self.global_best_tac:
                self.global_best_tac = current_tac
                self.global_best_config = (current_nt, current_feed, current_pressure)
                self.global_best_iteration = iteration
                logger.info(f"  [Global Best] Updated: TAC=${current_tac:,.0f} at P={current_pressure:.4f}, NT={current_nt}, NF={current_feed} (iter {iteration})")

            # ────────────────────────────────────────────────────────────────
            # CONVERGENCE CHECK
            # ────────────────────────────────────────────────────────────────
            
            logger.info("")
            logger.info("-" * 50)
            logger.info("CONVERGENCE CHECK")
            logger.info("-" * 50)
            
            tac_change = abs(current_tac - prev_tac)
            nt_changed = (current_nt != prev_nt)
            feed_changed = (current_feed != prev_feed)
            pressure_changed = abs(current_pressure - prev_pressure) > 0.001
            
            logger.info(f"  TAC change: ${tac_change:,.0f} (tolerance: ${self.tac_tolerance:,.0f})")
            logger.info(f"  NT changed: {nt_changed} ({prev_nt} -> {current_nt})")
            logger.info(f"  NF changed: {feed_changed} ({prev_feed} -> {current_feed})")
            logger.info(f"  P changed: {pressure_changed} ({prev_pressure:.4f} -> {current_pressure:.4f})")
            
            if tac_change < self.tac_tolerance and not nt_changed and not feed_changed and not pressure_changed:
                converged = True
                logger.info("")
                logger.info("★ CONVERGED! Design unchanged between iterations.")
                break
            else:
                logger.info("")
                logger.info("-> Not converged, continuing to next iteration...")
        
        # ════════════════════════════════════════════════════════════════════
        # BUILD FINAL RESULT (with global best check)
        # ════════════════════════════════════════════════════════════════════

        total_time = time.time() - self.start_time

        # Check if an earlier iteration found a better solution than the
        # converged point. This can happen when the ISO outer loop oscillates
        # (e.g., narrow feasibility bands missed by the coarse pressure grid).
        used_global_best = False
        if self.global_best_tac < current_tac and self.global_best_config is not None:
            gb_nt, gb_feed, gb_pressure = self.global_best_config
            logger.info("")
            logger.info("=" * 60)
            logger.info("GLOBAL BEST RECOVERY")
            logger.info("=" * 60)
            logger.info(f"  Converged solution: P={current_pressure:.4f}, NT={current_nt}, NF={current_feed}, TAC=${current_tac:,.0f}")
            logger.info(f"  Global best (iter {self.global_best_iteration}): P={gb_pressure:.4f}, NT={gb_nt}, NF={gb_feed}, TAC=${self.global_best_tac:,.0f}")
            logger.info(f"  Savings: ${current_tac - self.global_best_tac:,.0f}/year")
            logger.info(f"  -> Using global best as final optimum")
            logger.info("=" * 60)
            current_nt = gb_nt
            current_feed = gb_feed
            current_pressure = gb_pressure
            current_tac = self.global_best_tac
            used_global_best = True

        result = ISOResult(
            optimal_nt=current_nt,
            optimal_feed=current_feed,
            optimal_pressure=current_pressure,
            optimal_tac=current_tac,
            iterations=self.iterations,
            converged=converged,
            convergence_iteration=len(self.iterations),
            total_time_seconds=total_time,
            total_evaluations=self.eval_count,
            feasible_evaluations=self.feasible_count,
            infeasible_evaluations=self.infeasible_count,
            failed_rr_sweeps=self.failed_rr_sweeps,
            case_name=case_name,
            timestamp=timestamp,
        )

        # Store for visualization
        self.result = result
        self.used_global_best = used_global_best

        self._print_summary(result)

        # Emit final progress
        _emit_progress(
            iteration=len(self.iterations),
            phase="completed",
            current=len(self.iterations),
            total=len(self.iterations),
            best_tac=result.optimal_tac,
            message=f"Optimization complete! TAC=${result.optimal_tac:,.0f}/year",
            converged=result.converged,
            optimal_nt=result.optimal_nt,
            optimal_feed=result.optimal_feed,
            optimal_pressure=round(result.optimal_pressure, 4)
        )

        return result
    
    # ════════════════════════════════════════════════════════════════════════
    # SWEEP METHODS (ONE VARIABLE AT A TIME)
    # ════════════════════════════════════════════════════════════════════════
    
    def _sweep_pressure(self, fixed_nt: int, fixed_feed: int, iteration: int = 1,
                        prev_optimal_pressure: float = None) -> SweepResult:
        """
        Sweep pressure at fixed NT and NF.

        Includes temperature constraint check: T_reb <= 120C
        """
        result = SweepResult(
            parameter_name='pressure',
            fixed_values={'nt': fixed_nt, 'feed': fixed_feed}
        )

        p_min, p_max = self.pressure_bounds
        pressures = [
            p_min + i * (p_max - p_min) / (self.pressure_points - 1)
            for i in range(self.pressure_points)
        ]

        # Carry-forward: include previous iteration's optimal pressure
        # Prevents losing a known-good pressure when the coarse grid misses
        # narrow feasibility bands (e.g., COL4 P~0.044 between grid points)
        if prev_optimal_pressure is not None:
            already_in_grid = any(abs(p - prev_optimal_pressure) < 0.001 for p in pressures)
            if not already_in_grid and p_min <= prev_optimal_pressure <= p_max:
                pressures.append(prev_optimal_pressure)
                pressures.sort()
                logger.info(f"  [Carry-forward] Added previous optimal P={prev_optimal_pressure:.4f} to sweep grid")

        logger.info("")
        logger.info(f"{'Pressure':^10} {'T_reb':^10} {'TAC':^15} {'Status':^25}")
        logger.info("-" * 60)

        # Track best hard feasible and soft feasible separately
        best_hard_tac = float('inf')
        best_hard_pressure = p_min
        best_soft_tac = float('inf')
        best_soft_pressure = p_min
        total_points = len(pressures)

        last_rr = None  # RR warm-starting for sweep continuity
        for idx, p in enumerate(pressures):
            point = self._evaluate_with_feasibility(fixed_nt, fixed_feed, p, rr_hint=last_rr)
            result.points.append(point)
            if point.feasibility in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.SOFT_FEASIBLE) and point.reflux_ratio is not None:
                last_rr = point.reflux_ratio

            # Emit progress (use overall best for display)
            current_best = min(best_hard_tac, best_soft_tac)
            _emit_progress(
                iteration=iteration,
                phase="pressure_sweep",
                current=idx + 1,
                total=total_points,
                best_tac=current_best if current_best < float('inf') else None,
                pressure=round(p, 4),
                T_reb=round(point.T_reb, 1) if point.T_reb is not None and point.T_reb > 0 else None
            )

            # Format status - prioritize hard feasible over soft feasible
            if point.feasibility == FeasibilityStatus.FEASIBLE:
                status = "OK"
                if point.tac < best_hard_tac:
                    best_hard_tac = point.tac
                    best_hard_pressure = p
                    status = "*** BEST ***"
            elif point.feasibility == FeasibilityStatus.SOFT_FEASIBLE:
                status = "~OK (RR-recovered)"
                if point.tac < best_soft_tac:
                    best_soft_tac = point.tac
                    best_soft_pressure = p
                    # Only show as best if no hard feasible exists
                    if best_hard_tac >= float('inf'):
                        status = "~~ SOFT BEST ~~"
            elif point.feasibility == FeasibilityStatus.INFEASIBLE_TEMPERATURE:
                status = "[X] T_reb > 120C"
            else:
                status = f"[X] {point.infeasibility_reason[:20]}"

            tac_str = f"${point.tac:,.0f}" if point.tac < 1e10 else "N/A"
            T_str = f"{point.T_reb:.1f}C" if point.T_reb is not None and point.T_reb > 0 else "N/A"

            logger.info(f"{p:^10.4f} {T_str:^10} {tac_str:^15} {status:^25}")

        # Select optimal: prefer hard feasible, fallback to soft feasible
        if best_hard_tac < float('inf'):
            result.optimal_value = best_hard_pressure
            result.optimal_tac = best_hard_tac
            logger.info(f"  -> Coarse sweep HARD FEASIBLE: P={best_hard_pressure:.4f}, TAC=${best_hard_tac:,.0f}")
        elif best_soft_tac < float('inf'):
            result.optimal_value = best_soft_pressure
            result.optimal_tac = best_soft_tac
            logger.warning(f"  -> Coarse sweep SOFT FEASIBLE: P={best_soft_pressure:.4f}, TAC=${best_soft_tac:,.0f}")
        else:
            result.optimal_value = p_min
            result.optimal_tac = float('inf')
            logger.warning("  -> No feasible pressure found!")
            return result

        # ════════════════════════════════════════════════════════════════════
        # PRESSURE REFINEMENT (Bisection near feasibility boundary)
        # ════════════════════════════════════════════════════════════════════
        # If TAC is declining at the best pressure (i.e., the next higher
        # pressure is infeasible), refine the boundary to find the true
        # optimum. This prevents the "stuck at boundary" issue (Comment #6).
        result = self._refine_pressure(
            result, fixed_nt, fixed_feed, pressures, iteration
        )

        return result
    
    def _refine_pressure(self, result: SweepResult, fixed_nt: int, fixed_feed: int,
                          coarse_pressures: list, iteration: int) -> SweepResult:
        """
        Refine pressure near the feasibility boundary using bisection.

        After the coarse sweep, if TAC is still declining at the best feasible
        pressure (i.e., the next grid point is infeasible), we bisect the gap
        to find the true optimum closer to the feasibility boundary.

        This addresses Comment #6: "Is the pressure trapped at the boundary?"
        """
        # Identify feasibility of each coarse point
        feasible_points = []  # (pressure, tac, point)
        infeasible_after = []  # pressures that are infeasible right after a feasible

        for p_point in result.points:
            if p_point.feasibility in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.SOFT_FEASIBLE):
                feasible_points.append((p_point.pressure, p_point.tac, p_point))

        if len(feasible_points) < 2:
            return result  # Not enough points to refine

        # Sort by pressure
        feasible_points.sort(key=lambda x: x[0])

        # Find the best feasible pressure
        best_p = result.optimal_value
        best_tac = result.optimal_tac

        # Find the next coarse pressure above best_p
        sorted_coarse = sorted(coarse_pressures)
        best_idx = None
        for i, p in enumerate(sorted_coarse):
            if abs(p - best_p) < 1e-6:
                best_idx = i
                break

        if best_idx is None or best_idx >= len(sorted_coarse) - 1:
            return result  # Best is at or beyond the last coarse point

        next_p = sorted_coarse[best_idx + 1]

        # Check if the next point is infeasible
        next_point = None
        for p_point in result.points:
            if abs(p_point.pressure - next_p) < 1e-6:
                next_point = p_point
                break

        if next_point is None:
            return result

        next_is_infeasible = next_point.feasibility not in (
            FeasibilityStatus.FEASIBLE, FeasibilityStatus.SOFT_FEASIBLE
        )

        # Also check if TAC was declining (best is the LAST feasible, not an interior minimum)
        # An interior minimum doesn't need refinement — the U-curve found it naturally.
        prev_p_point = None
        if best_idx > 0:
            prev_p = sorted_coarse[best_idx - 1]
            for p_point in result.points:
                if abs(p_point.pressure - prev_p) < 1e-6:
                    prev_p_point = p_point
                    break

        tac_declining = True  # Default: assume declining
        if prev_p_point and prev_p_point.feasibility in (
            FeasibilityStatus.FEASIBLE, FeasibilityStatus.SOFT_FEASIBLE
        ):
            # If previous point has HIGHER TAC, the trend is declining → boundary optimum
            tac_declining = prev_p_point.tac > best_tac

        if not next_is_infeasible or not tac_declining:
            # Best is interior (proper U-curve minimum) or next point is still feasible
            # No refinement needed — the coarse sweep found a true minimum
            logger.info(f"  -> Pressure refinement: NOT NEEDED (interior minimum at P={best_p:.4f})")
            return result

        # ════════════════════════════════════════════════════════════════════
        # BISECTION: TAC still declining, next point infeasible
        # Explore between best_p and next_p to find better feasible point
        # ════════════════════════════════════════════════════════════════════
        logger.info("")
        logger.info(f"  -> Pressure refinement: TAC declining at P={best_p:.4f}, "
                    f"next grid point P={next_p:.4f} is infeasible")
        logger.info(f"     Bisecting [{best_p:.4f}, {next_p:.4f}] to find true boundary...")

        refine_tol = 0.005  # 5 mbar tolerance
        max_refine_steps = 6  # At most 6 bisection steps

        low = best_p
        high = next_p
        best_refine_p = best_p
        best_refine_tac = best_tac

        last_rr = None
        # Get RR from the best coarse point for warm-starting
        for p_point in result.points:
            if abs(p_point.pressure - best_p) < 1e-6 and p_point.reflux_ratio is not None:
                last_rr = p_point.reflux_ratio
                break

        logger.info(f"{'Step':^6} {'Pressure':^10} {'T_reb':^10} {'TAC':^15} {'Status':^20}")
        logger.info("-" * 65)

        for step in range(max_refine_steps):
            mid = (low + high) / 2.0

            if high - low < refine_tol:
                logger.info(f"  Refinement converged (gap < {refine_tol} bar)")
                break

            point = self._evaluate_with_feasibility(fixed_nt, fixed_feed, mid, rr_hint=last_rr)
            result.points.append(point)

            if point.feasibility in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.SOFT_FEASIBLE):
                if point.reflux_ratio is not None:
                    last_rr = point.reflux_ratio

                tac_str = f"${point.tac:,.0f}"
                T_str = f"{point.T_reb:.1f}C" if point.T_reb is not None and point.T_reb > 0 else "N/A"

                if point.tac < best_refine_tac:
                    best_refine_tac = point.tac
                    best_refine_p = mid
                    status = "*** BETTER ***"
                else:
                    status = "OK (TAC rising)"

                logger.info(f"{step+1:^6} {mid:^10.4f} {T_str:^10} {tac_str:^15} {status:^20}")

                # Feasible: try higher pressure (push toward boundary)
                low = mid
            else:
                # Infeasible: try lower pressure
                T_str = f"{point.T_reb:.1f}C" if point.T_reb is not None and point.T_reb > 0 else "N/A"
                reason = point.infeasibility_reason[:15] if point.infeasibility_reason else "infeasible"
                logger.info(f"{step+1:^6} {mid:^10.4f} {T_str:^10} {'N/A':^15} {'[X] ' + reason:^20}")
                high = mid

        # Update result with refined optimum
        if best_refine_tac < best_tac:
            logger.info(f"  -> Refinement improved: P={best_refine_p:.4f} (TAC=${best_refine_tac:,.0f}) "
                       f"vs coarse P={best_p:.4f} (TAC=${best_tac:,.0f})")
            logger.info(f"     Savings: ${best_tac - best_refine_tac:,.0f}/year")
            result.optimal_value = best_refine_p
            result.optimal_tac = best_refine_tac
        else:
            logger.info(f"  -> Refinement: no improvement over coarse sweep (P={best_p:.4f})")

        return result

    def _sweep_nt(self, fixed_pressure: float, fixed_feed: int, iteration: int = 1) -> SweepResult:
        """
        Sweep NT at fixed pressure and feed.

        Generates classic U-shaped TAC curve.
        """
        result = SweepResult(
            parameter_name='nt',
            fixed_values={'pressure': fixed_pressure, 'feed': fixed_feed}
        )

        nt_min, nt_max = self.nt_bounds
        nt_values = list(range(nt_min, nt_max + 1, self.nt_step))

        logger.info("")
        logger.info(f"{'NT':^8} {'TAC':^15} {'Status':^20}")
        logger.info("-" * 45)

        # Track best hard feasible and soft feasible separately
        best_hard_tac = float('inf')
        best_hard_nt = nt_min
        best_soft_tac = float('inf')
        best_soft_nt = nt_min
        total_points = len(nt_values)

        last_rr = None  # RR warm-starting for sweep continuity
        rising_count = 0  # Consecutive rising-TAC feasible points after minimum
        EARLY_STOP_THRESHOLD = 5  # Stop after this many consecutive rises past minimum
        for idx, nt in enumerate(nt_values):
            # Adjust feed if needed
            adjusted_feed = self._adjust_feed_for_nt(fixed_feed, nt)

            if adjusted_feed < self.min_section_stages + 1:
                continue  # Skip invalid configurations

            point = self._evaluate_with_feasibility(nt, adjusted_feed, fixed_pressure, rr_hint=last_rr)
            result.points.append(point)
            if point.feasibility in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.SOFT_FEASIBLE) and point.reflux_ratio is not None:
                last_rr = point.reflux_ratio

            # Format status - prioritize hard feasible over soft feasible
            if point.feasibility == FeasibilityStatus.FEASIBLE:
                if point.tac < best_hard_tac:
                    best_hard_tac = point.tac
                    best_hard_nt = nt
                    status = "*** BEST ***"
                    rising_count = 0  # New minimum found, reset counter
                else:
                    status = "OK"
                    rising_count += 1
            elif point.feasibility == FeasibilityStatus.SOFT_FEASIBLE:
                if point.tac < best_soft_tac:
                    best_soft_tac = point.tac
                    best_soft_nt = nt
                    # Only show as best if no hard feasible exists
                    if best_hard_tac >= float('inf'):
                        status = "~~ SOFT BEST ~~"
                        rising_count = 0  # New minimum found, reset counter
                    else:
                        status = "~OK (RR-recovered)"
                        rising_count += 1
                else:
                    status = "~OK (RR-recovered)"
                    rising_count += 1
            else:
                status = f"[X] {point.infeasibility_reason[:15]}"
                # Don't count infeasible points toward early termination

            tac_str = f"${point.tac:,.0f}" if point.tac < 1e10 else "N/A"
            logger.info(f"{nt:^8} {tac_str:^15} {status:^20}")

            # Emit progress (use overall best for display)
            current_best = min(best_hard_tac, best_soft_tac)
            _emit_progress(
                iteration=iteration,
                phase="nt_sweep",
                current=idx + 1,
                total=total_points,
                best_tac=current_best if current_best < float('inf') else None,
                nt=nt
            )

            # Early termination: stop if past U-curve minimum
            if rising_count >= EARLY_STOP_THRESHOLD and current_best < float('inf'):
                logger.info(f"  -> Early termination: {EARLY_STOP_THRESHOLD} consecutive rises after minimum")
                break

        # Select optimal: prefer hard feasible, fallback to soft feasible
        if best_hard_tac < float('inf'):
            result.optimal_value = best_hard_nt
            result.optimal_tac = best_hard_tac
            logger.info(f"  -> Selected HARD FEASIBLE: NT={best_hard_nt}, TAC=${best_hard_tac:,.0f}")
        elif best_soft_tac < float('inf'):
            result.optimal_value = best_soft_nt
            result.optimal_tac = best_soft_tac
            logger.warning(f"  -> Selected SOFT FEASIBLE (no hard feasible available): NT={best_soft_nt}, TAC=${best_soft_tac:,.0f}")
            logger.warning(f"     WARNING: This is an RR-recovered point - results may be less reliable")
        else:
            result.optimal_value = nt_min
            result.optimal_tac = float('inf')
            logger.warning("  -> No feasible NT found!")

        return result

    def _sweep_feed(self, fixed_pressure: float, fixed_nt: int, iteration: int = 1) -> SweepResult:
        """
        Sweep feed stage at fixed pressure and NT.
        
        Generates U-shaped curve for feed optimization.
        """
        result = SweepResult(
            parameter_name='feed',
            fixed_values={'pressure': fixed_pressure, 'nt': fixed_nt}
        )
        
        # Calculate valid feed range for this NT
        f_min = max(self.feed_bounds[0], self.min_section_stages + 1)
        f_max = min(self.feed_bounds[1], fixed_nt - self.min_section_stages)
        
        if f_min > f_max:
            logger.warning(f"  No valid feed range for NT={fixed_nt}")
            return result
        
        feed_values = list(range(f_min, f_max + 1, self.feed_step))

        logger.info("")
        logger.info(f"{'Feed':^8} {'TAC':^15} {'Status':^20}")
        logger.info("-" * 45)

        # Track best hard feasible and soft feasible separately
        best_hard_tac = float('inf')
        best_hard_feed = f_min
        best_soft_tac = float('inf')
        best_soft_feed = f_min
        total_points = len(feed_values)

        last_rr = None  # RR warm-starting for sweep continuity
        rising_count = 0  # Consecutive rising-TAC feasible points after minimum
        EARLY_STOP_THRESHOLD = 5  # Stop after this many consecutive rises past minimum
        for idx, feed in enumerate(feed_values):
            point = self._evaluate_with_feasibility(fixed_nt, feed, fixed_pressure, rr_hint=last_rr)
            result.points.append(point)
            if point.feasibility in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.SOFT_FEASIBLE) and point.reflux_ratio is not None:
                last_rr = point.reflux_ratio

            # Format status - prioritize hard feasible over soft feasible
            if point.feasibility == FeasibilityStatus.FEASIBLE:
                if point.tac < best_hard_tac:
                    best_hard_tac = point.tac
                    best_hard_feed = feed
                    status = "*** BEST ***"
                    rising_count = 0  # New minimum found, reset counter
                else:
                    status = "OK"
                    rising_count += 1
            elif point.feasibility == FeasibilityStatus.SOFT_FEASIBLE:
                if point.tac < best_soft_tac:
                    best_soft_tac = point.tac
                    best_soft_feed = feed
                    # Only show as best if no hard feasible exists
                    if best_hard_tac >= float('inf'):
                        status = "~~ SOFT BEST ~~"
                        rising_count = 0  # New minimum found, reset counter
                    else:
                        status = "~OK (RR-recovered)"
                        rising_count += 1
                else:
                    status = "~OK (RR-recovered)"
                    rising_count += 1
            else:
                status = f"[X] {point.infeasibility_reason[:15]}"
                # Don't count infeasible points toward early termination

            tac_str = f"${point.tac:,.0f}" if point.tac < 1e10 else "N/A"
            logger.info(f"{feed:^8} {tac_str:^15} {status:^20}")

            # Emit progress (use overall best for display)
            current_best = min(best_hard_tac, best_soft_tac)
            _emit_progress(
                iteration=iteration,
                phase="feed_sweep",
                current=idx + 1,
                total=total_points,
                best_tac=current_best if current_best < float('inf') else None,
                feed=feed
            )

            # Early termination: stop if past U-curve minimum
            if rising_count >= EARLY_STOP_THRESHOLD and current_best < float('inf'):
                logger.info(f"  -> Early termination: {EARLY_STOP_THRESHOLD} consecutive rises after minimum")
                break

        # Select optimal: prefer hard feasible, fallback to soft feasible
        if best_hard_tac < float('inf'):
            result.optimal_value = best_hard_feed
            result.optimal_tac = best_hard_tac
            logger.info(f"  -> Selected HARD FEASIBLE: NF={best_hard_feed}, TAC=${best_hard_tac:,.0f}")
        elif best_soft_tac < float('inf'):
            result.optimal_value = best_soft_feed
            result.optimal_tac = best_soft_tac
            logger.warning(f"  -> Selected SOFT FEASIBLE (no hard feasible available): NF={best_soft_feed}, TAC=${best_soft_tac:,.0f}")
            logger.warning(f"     WARNING: This is an RR-recovered point - results may be less reliable")
        else:
            result.optimal_value = f_min
            result.optimal_tac = float('inf')
            logger.warning("  -> No feasible feed found!")

        return result

    # ════════════════════════════════════════════════════════════════════════
    # U-CURVE VALIDATION
    # ════════════════════════════════════════════════════════════════════════

    def _validate_u_curve_quality(self, pressure: float, fixed_feed: int) -> Tuple[bool, int, str]:
        """
        Validate that a given pressure produces a proper U-curve for NT optimization.

        A valid U-curve requires:
        1. At least MIN_CONSECUTIVE_FEASIBLE consecutive feasible NT points
        2. The optimal NT has neighbors with higher TAC (proper minimum, not edge)

        Parameters
        ----------
        pressure : float
            The pressure to validate
        fixed_feed : int
            The feed stage to use for validation

        Returns
        -------
        tuple : (is_valid, consecutive_feasible_count, reason)
            - is_valid: True if pressure produces valid U-curve
            - consecutive_feasible_count: Number of consecutive feasible points found
            - reason: Explanation if invalid
        """
        nt_min, nt_max = self.nt_bounds
        nt_values = list(range(nt_min, nt_max + 1, self.nt_step))

        hard_feasible_points = []
        soft_feasible_points = []
        best_hard_nt = None
        best_hard_tac = float('inf')
        best_soft_nt = None
        best_soft_tac = float('inf')

        # Quick sweep to check feasibility
        for nt in nt_values:
            adjusted_feed = self._adjust_feed_for_nt(fixed_feed, nt)
            if adjusted_feed < self.min_section_stages + 1:
                continue

            # Use cache if available, otherwise do quick eval
            key = (nt, adjusted_feed, round(pressure, 4))
            if key in self.cache:
                point = self.cache[key]
            else:
                # Skip full evaluation - just record as unknown
                continue

            if point.feasibility == FeasibilityStatus.FEASIBLE:
                hard_feasible_points.append((nt, point.tac))
                if point.tac < best_hard_tac:
                    best_hard_tac = point.tac
                    best_hard_nt = nt
            elif point.feasibility == FeasibilityStatus.SOFT_FEASIBLE:
                soft_feasible_points.append((nt, point.tac))
                if point.tac < best_soft_tac:
                    best_soft_tac = point.tac
                    best_soft_nt = nt

        # For U-curve validation, primarily use HARD FEASIBLE points
        # This ensures the U-curve is based on naturally converged points
        feasible_points = hard_feasible_points
        best_nt = best_hard_nt
        best_tac = best_hard_tac

        # Check 1: Minimum consecutive feasible points (prefer hard feasible)
        if len(hard_feasible_points) < MIN_CONSECUTIVE_FEASIBLE:
            # Check if including soft feasible would help
            all_feasible = hard_feasible_points + soft_feasible_points
            if len(all_feasible) >= MIN_CONSECUTIVE_FEASIBLE:
                logger.warning(f"  U-curve validation: Only {len(hard_feasible_points)} hard feasible points, "
                              f"but {len(all_feasible)} total (including {len(soft_feasible_points)} soft feasible)")
                logger.warning(f"  WARNING: U-curve contains RR-recovered points - may not be reliable")
                # Use all feasible points for validation, but this is a warning sign
                feasible_points = all_feasible
                if best_hard_nt is None:
                    best_nt = best_soft_nt
                    best_tac = best_soft_tac
            else:
                return (False, len(hard_feasible_points),
                        f"Only {len(hard_feasible_points)} hard feasible points (need {MIN_CONSECUTIVE_FEASIBLE})")

        # Check consecutive feasible points
        consecutive_count = 1
        max_consecutive = 1
        feasible_nts = sorted([p[0] for p in feasible_points])

        for i in range(1, len(feasible_nts)):
            if feasible_nts[i] - feasible_nts[i-1] == self.nt_step:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1

        if max_consecutive < MIN_CONSECUTIVE_FEASIBLE:
            return (False, max_consecutive,
                    f"Only {max_consecutive} consecutive feasible points (need {MIN_CONSECUTIVE_FEASIBLE})")

        # Check 2: Best NT should have BOTH neighbors with higher TAC (proper U minimum)
        if U_CURVE_NEIGHBOR_CHECK and best_nt is not None:
            tac_by_nt = {nt: tac for nt, tac in feasible_points}
            lower_nt = best_nt - self.nt_step
            upper_nt = best_nt + self.nt_step

            lower_exists = lower_nt in tac_by_nt
            upper_exists = upper_nt in tac_by_nt

            # STRICT: Require BOTH neighbors to exist for a proper U-curve minimum
            # A boundary minimum (only one neighbor) is not reliable
            if not lower_exists or not upper_exists:
                missing = []
                if not lower_exists:
                    missing.append(f"NT={lower_nt} (lower)")
                if not upper_exists:
                    missing.append(f"NT={upper_nt} (upper)")
                return (False, max_consecutive,
                        f"Optimal NT={best_nt} missing neighbors: {', '.join(missing)} - boundary minimum")

            # Both neighbors must have higher TAC (true U-curve minimum)
            lower_tac = tac_by_nt[lower_nt]
            upper_tac = tac_by_nt[upper_nt]
            lower_higher = lower_tac > best_tac
            upper_higher = upper_tac > best_tac

            if not lower_higher and not upper_higher:
                return (False, max_consecutive,
                        f"NT={best_nt} is not at minimum (both neighbors have lower TAC)")

            if not lower_higher:
                return (False, max_consecutive,
                        f"NT={best_nt} not at minimum: NT={lower_nt} has TAC=${lower_tac:,.0f} <= ${best_tac:,.0f}")

            if not upper_higher:
                return (False, max_consecutive,
                        f"NT={best_nt} not at minimum: NT={upper_nt} has TAC=${upper_tac:,.0f} <= ${best_tac:,.0f}")

            # Check 3: TAC jump to neighbors shouldn't be abnormally large (>100% is suspicious)
            # This catches edge cases where the "minimum" is actually a boundary artifact
            lower_ratio = lower_tac / best_tac if best_tac > 0 else float('inf')
            upper_ratio = upper_tac / best_tac if best_tac > 0 else float('inf')

            MAX_TAC_RATIO = 2.0  # Neighbor TAC shouldn't be more than 2x the minimum
            if lower_ratio > MAX_TAC_RATIO or upper_ratio > MAX_TAC_RATIO:
                return (False, max_consecutive,
                        f"Abnormal TAC jump at NT={best_nt}: ratios={lower_ratio:.1f}x/{upper_ratio:.1f}x (max {MAX_TAC_RATIO}x)")

        return (True, max_consecutive, "Valid U-curve")

    def _select_pressure_with_validation(self, pressure_sweep: 'SweepResult',
                                          fixed_nt: int, fixed_feed: int) -> Tuple[float, float]:
        """
        Select the best pressure that produces a valid U-curve.

        Falls back to lower pressures if the lowest-TAC pressure has edge-case behavior.

        Parameters
        ----------
        pressure_sweep : SweepResult
            Results from pressure sweep
        fixed_nt : int
            Current NT value
        fixed_feed : int
            Current feed value

        Returns
        -------
        tuple : (selected_pressure, selected_tac)
        """
        # Sort pressures by TAC (best first) - prefer HARD FEASIBLE, fallback to SOFT FEASIBLE
        hard_feasible_pressures = [
            (p.pressure, p.tac)
            for p in pressure_sweep.points
            if p.feasibility == FeasibilityStatus.FEASIBLE
        ]
        soft_feasible_pressures = [
            (p.pressure, p.tac)
            for p in pressure_sweep.points
            if p.feasibility == FeasibilityStatus.SOFT_FEASIBLE
        ]

        if hard_feasible_pressures:
            hard_feasible_pressures.sort(key=lambda x: x[1])  # Sort by TAC
            best_pressure = hard_feasible_pressures[0][0]
            best_tac = hard_feasible_pressures[0][1]
            logger.info(f"  Selected HARD FEASIBLE: P={best_pressure:.4f} bar with TAC=${best_tac:,.0f}")
        elif soft_feasible_pressures:
            soft_feasible_pressures.sort(key=lambda x: x[1])  # Sort by TAC
            best_pressure = soft_feasible_pressures[0][0]
            best_tac = soft_feasible_pressures[0][1]
            logger.warning(f"  Selected SOFT FEASIBLE (no hard feasible available): P={best_pressure:.4f} bar with TAC=${best_tac:,.0f}")
        else:
            return (self.pressure_bounds[0], float('inf'))

        return (best_pressure, best_tac)

    # ════════════════════════════════════════════════════════════════════════
    # EVALUATION WITH FEASIBILITY CHECK
    # ════════════════════════════════════════════════════════════════════════

    def _evaluate_with_feasibility(self, nt: int, feed: int, pressure: float, rr_hint: float = None) -> EvaluationPoint:
        """
        Evaluate a configuration and check feasibility constraints.

        CRITICAL: Applies temperature constraint T_reb <= 120C
        """
        # Check cache
        key = (nt, feed, round(pressure, 4))
        if key in self.cache:
            return self.cache[key]
        
        # Evaluate using existing evaluator (with diagnostic on failure)
        self.eval_count += 1
        result = self.evaluator.evaluate(nt, feed, pressure, run_diagnostic_on_fail=True, rr_sweep_on_fail=True, purity_spec=self.purity_spec, rr_hint=rr_hint)

        # Handle None result from evaluator (simulation/recovery failure)
        if result is None:
            logger.warning(f"  Evaluator returned None for NT={nt}, NF={feed}, P={pressure:.4f}")
            point = EvaluationPoint(
                nt=nt,
                feed=feed,
                pressure=pressure,
                tac=float('inf'),
                tpc=0,
                toc=0,
                q_reb=0,
                q_cond=0,
                diameter=0,
                T_reb=None,
                T_cond=None,
                converged=False,
            )
            point.feasibility = FeasibilityStatus.INFEASIBLE_CONVERGENCE
            point.infeasibility_reason = "Evaluator returned None (simulation failure)"
            self.infeasible_count += 1
            self.cache[key] = point
            self.all_evaluations.append(point)
            return point

        # Collect RR sweep data from failed evaluations (for infeasible design visualization)
        if result.get('rr_sweep'):
            self.failed_rr_sweeps.append({
                'nt': nt,
                'feed': feed,
                'pressure': pressure,
                'rr_sweep': result['rr_sweep']
            })
            logger.debug(f"  Collected RR sweep from failed design: NT={nt}, NF={feed}, P={pressure:.4f}")

        # Extract values (with None safety)
        tac = result.get('TAC', float('inf'))
        converged = result.get('converged', False)
        T_reb = result.get('T_reb')
        T_cond = result.get('T_cond')
        
        # Track whether temperatures are missing (for feasibility check)
        T_reb_missing = (T_reb is None)
        T_cond_missing = (T_cond is None)

        # Create evaluation point
        point = EvaluationPoint(
            nt=nt,
            feed=feed,
            pressure=pressure,
            tac=tac,
            tpc=result.get('TPC', 0),
            toc=result.get('TOC', 0),
            q_reb=result.get('Q_reb', 0),
            q_cond=result.get('Q_cond', 0),
            diameter=result.get('diameter', 0),
            T_reb=T_reb,
            T_cond=T_cond,
            converged=converged,
        )

        # Store reflux ratio from Aspen output (for warm-starting)
        point.reflux_ratio = result.get('reflux_ratio')

        # ════════════════════════════════════════════════════════════════════
        # FEASIBILITY CHECKS (Professor's requirements)
        # ════════════════════════════════════════════════════════════════════

        if not converged:
            point.feasibility = FeasibilityStatus.INFEASIBLE_CONVERGENCE
            point.infeasibility_reason = "Simulation did not converge"
            self.infeasible_count += 1

        elif self.has_styrene and T_reb_missing:
            # CRITICAL: If T_reb cannot be extracted AND styrene is present,
            # treat as infeasible (cannot verify temperature constraint)
            point.feasibility = FeasibilityStatus.INFEASIBLE_TEMPERATURE
            point.infeasibility_reason = "T_reb extraction failed (cannot verify constraint)"
            point.tac = float('inf')
            self.infeasible_count += 1
            logger.warning(f"  [X] INFEASIBLE: T_reb could not be extracted for NT={nt}, P={pressure:.4f}")

        elif self.has_styrene and T_reb is not None and T_reb > self.T_reb_max:
            # Temperature constraint: ONLY for columns with styrene monomer
            # Styrene polymerizes above ~120°C in the reboiler
            point.feasibility = FeasibilityStatus.INFEASIBLE_TEMPERATURE
            point.infeasibility_reason = f"T_reb={T_reb:.1f}C > {self.T_reb_max}C (polymerization risk)"
            point.tac = float('inf')  # Exclude from optimization
            self.infeasible_count += 1

            logger.info(f"  [X] INFEASIBLE: {point.infeasibility_reason}")

        elif tac >= 1e10:
            point.feasibility = FeasibilityStatus.INFEASIBLE_CONVERGENCE
            point.infeasibility_reason = "Invalid TAC result"
            self.infeasible_count += 1

        else:
            # Check if this was an RR-recovered point (soft feasible)
            is_rr_recovered = result.get('recovered_from_rr_sweep', False)

            if is_rr_recovered:
                point.feasibility = FeasibilityStatus.SOFT_FEASIBLE
                self.soft_feasible_count += 1
                recovery_rr = result.get('recovery_rr', 0)
                T_reb_str = f"{T_reb:.1f}" if T_reb is not None else "N/A"
                logger.info(f"  [~] SOFT FEASIBLE (RR-recovered): TAC=${tac:,.0f}, T_reb={T_reb_str}C, RR={recovery_rr:.2f} (P={pressure:.4f})")
            else:
                point.feasibility = FeasibilityStatus.FEASIBLE
                self.feasible_count += 1
                T_reb_str = f"{T_reb:.1f}" if T_reb is not None else "N/A"
                logger.info(f"  [OK] FEASIBLE: TAC=${tac:,.0f}, T_reb={T_reb_str}C (P={pressure:.4f})")
        
        # Cache and store
        self.cache[key] = point
        self.all_evaluations.append(point)
        
        return point
    
    def _adjust_feed_for_nt(self, feed: int, nt: int) -> int:
        """Adjust feed stage to be valid for given NT."""
        f_min = self.min_section_stages + 1
        f_max = nt - self.min_section_stages
        return max(f_min, min(feed, f_max))
    
    # ════════════════════════════════════════════════════════════════════════
    # REFLUX DIAGNOSTIC (For unconverged cases)
    # ════════════════════════════════════════════════════════════════════════
    
    def diagnose_infeasibility(self, nt: int, pressure: float) -> Dict:
        """
        Diagnostic for unconverged cases.
        
        Temporarily disables purity specs and sweeps reflux ratio
        to determine if separation is physically feasible.
        
        NOTE: This is a diagnostic tool, not part of main optimization.
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("INFEASIBILITY DIAGNOSTIC: Reflux Ratio Sweep")
        logger.info("=" * 60)
        logger.info(f"NT={nt}, P={pressure:.4f} bar")
        logger.info("")
        logger.warning("This diagnostic requires manual Aspen manipulation.")
        logger.warning("Steps:")
        logger.warning("  1. Deactivate product purity specs and VARY blocks")
        logger.warning("  2. Set reflux ratio as a free variable")
        logger.warning("  3. Sweep reflux ratio and record purity")
        logger.warning("  4. Plot purity vs reflux ratio")
        logger.info("")
        
        # Return placeholder - actual implementation would require
        # specific Aspen manipulation
        return {
            'status': 'diagnostic_required',
            'message': 'Manual Aspen manipulation needed for reflux sweep',
            'suggested_action': 'Run reflux diagnostic separately'
        }
    
    # ════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ════════════════════════════════════════════════════════════════════════
    
    def _print_header(self, case_name: str):
        """Print optimization header."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("ITERATIVE SEQUENTIAL OPTIMIZATION (ISO)")
        logger.info("=" * 70)
        logger.info(f"Case: {case_name}")
        logger.info("")
        logger.info("Methodology: TRUE ISO (Professor's requirements)")
        logger.info("  * Variables optimized ONE AT A TIME: P -> NT -> NF")
        logger.info("  * Outer iteration loop with convergence check")
        if self.has_styrene:
            logger.info(f"  * Temperature constraint: T_reb <= {self.T_reb_max}C (styrene in reboiler)")
        else:
            logger.info(f"  * Temperature constraint: INACTIVE (no styrene monomer in reboiler)")
        logger.info("")
        logger.info("Penalty conditions:")
        logger.info("  * T_reb violation (if styrene present): TAC = infinity")
        logger.info("  * Non-convergence (RadFrac solver failure): TAC = infinity")
        logger.info("  * Invalid results (Q_reb/Q_cond unreadable): TAC = infinity")
        logger.info("")
        logger.info("Physical basis:")
        logger.info("  * Pressure: Strategic (thermodynamic regime)")
        logger.info("  * NT: Capital vs energy trade-off")
        logger.info("  * NF: Feed location optimization")
        logger.info("=" * 70)
    
    def _print_summary(self, result: ISOResult):
        """Print optimization summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("ISO OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info("")
        logger.info(f"Case: {result.case_name}")
        logger.info(f"Converged: {'YES [OK]' if result.converged else 'NO'}")
        logger.info(f"Iterations: {result.convergence_iteration}")
        logger.info(f"Time: {result.total_time_seconds:.1f}s ({result.total_time_seconds/60:.1f} min)")
        logger.info("")
        logger.info("-" * 40)
        logger.info("OPTIMAL CONFIGURATION")
        if getattr(self, 'used_global_best', False):
            logger.info(f"  (from global best at iteration {self.global_best_iteration})")
        logger.info("-" * 40)
        logger.info(f"  Number of Stages (NT): {result.optimal_nt}")
        logger.info(f"  Feed Stage (NF): {result.optimal_feed}")
        logger.info(f"  Operating Pressure: {result.optimal_pressure:.4f} bar")
        logger.info(f"  TAC: ${result.optimal_tac:,.0f}/year")
        logger.info("")
        logger.info("-" * 40)
        logger.info("EVALUATION STATISTICS")
        logger.info("-" * 40)
        logger.info(f"  Total evaluations: {result.total_evaluations}")
        logger.info(f"  Feasible: {result.feasible_evaluations}")
        if self.has_styrene:
            logger.info(f"  Infeasible (T_reb > {self.T_reb_max}C): {result.infeasible_evaluations}")
        else:
            logger.info(f"  Infeasible (convergence): {result.infeasible_evaluations}")
            logger.info(f"  T_reb constraint: INACTIVE (no styrene)")
        logger.info("")
        logger.info("=" * 70)
    
    def get_ucurve_data(self) -> Dict:
        """
        Get data formatted for U-curve plotting.
        
        Returns dictionary with curves from each step of final iteration.
        """
        if not self.iterations:
            return {}
        
        final = self.iterations[-1]
        
        # Pressure curve
        pressure_curve = [
            (p.pressure, p.tac, p.feasibility.value, p.T_reb)
            for p in final.pressure_sweep.points
        ]
        
        # NT curve (at optimal pressure and feed)
        nt_curve = [
            (p.nt, p.tac, p.feasibility.value)
            for p in final.nt_sweep.points
        ]
        
        # Feed curve (at optimal pressure and NT)
        feed_curve = [
            (p.feed, p.tac, p.feasibility.value)
            for p in final.feed_sweep.points
        ]
        
        return {
            'pressure_curve': pressure_curve,
            'nt_curve': nt_curve,
            'feed_curve': feed_curve,
            'optimal': {
                'pressure': final.optimal_pressure,
                'nt': final.optimal_nt,
                'feed': final.optimal_feed,
                'tac': final.optimal_tac,
            },
            'iterations': len(self.iterations),
        }

    def _build_baseline_section(self) -> Dict:
        """Build baseline section for result JSON."""
        if self.baseline_result is None:
            return {
                'recorded': False,
                'reason': 'No baseline evaluation performed',
            }

        initial = self.config.get('initial', {})
        return {
            'recorded': True,
            'nt': initial.get('nt'),
            'feed': initial.get('feed'),
            'pressure': initial.get('pressure'),
            'tac': self.baseline_tac,
            'converged': self.baseline_result.converged,
            'feasibility': self.baseline_result.feasibility.value,
            'T_reb': self.baseline_result.T_reb if hasattr(self.baseline_result, 'T_reb') else None,
        }

    def _build_improvement_section(self) -> Dict:
        """Build improvement metrics section for result JSON."""
        if self.baseline_tac is None or self.baseline_tac <= 0:
            return {
                'baseline_available': False,
                'reason': 'Baseline TAC not available (did not converge or infeasible)',
            }

        optimized_tac = self.result.optimal_tac
        absolute_savings = self.baseline_tac - optimized_tac
        relative_improvement = (absolute_savings / self.baseline_tac) * 100

        return {
            'baseline_available': True,
            'baseline_tac': round(self.baseline_tac, 2),
            'optimized_tac': round(optimized_tac, 2),
            'absolute_savings_per_year': round(absolute_savings, 2),
            'relative_improvement_percent': round(relative_improvement, 2),
            'summary': f"${absolute_savings:,.0f}/year savings ({relative_improvement:.1f}% reduction)",
        }

    def save_results(self, output_dir: str = "results") -> str:
        """Save all results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        # Build baseline section
        baseline_section = self._build_baseline_section()
        improvement_section = self._build_improvement_section()

        # Build output dictionary
        output = {
            'metadata': {
                'methodology': 'TRUE Iterative Sequential Optimization (ISO)',
                'description': 'P -> NT -> NF, one at a time, with outer loop',
                'has_styrene': self.has_styrene,
                'temperature_constraint': f'T_reb <= {self.T_reb_max}C' if self.has_styrene else 'INACTIVE (no styrene)',
                'convergence_tolerance': self.tac_tolerance,
                'cost_index': 'CEPCI (unified)',
            },
            'baseline': baseline_section,
            'optimal': {
                'nt': self.result.optimal_nt,
                'feed': self.result.optimal_feed,
                'pressure': self.result.optimal_pressure,
                'tac': self.result.optimal_tac,
                'from_global_best': getattr(self, 'used_global_best', False),
                'global_best_iteration': self.global_best_iteration if getattr(self, 'used_global_best', False) else None,
            },
            'improvement': improvement_section,
            'convergence': {
                'converged': self.result.converged,
                'iterations': self.result.convergence_iteration,
            },
            'statistics': {
                'total_evaluations': self.eval_count,
                'feasible': self.feasible_count,
                'infeasible': self.infeasible_count,
                'time_seconds': time.time() - self.start_time,
            },
        }
        
        # Add iteration details
        output['iterations'] = []
        for iter_result in self.iterations:
            iter_dict = {
                'iteration': iter_result.iteration,
                'optimal_pressure': iter_result.optimal_pressure,
                'optimal_nt': iter_result.optimal_nt,
                'optimal_feed': iter_result.optimal_feed,
                'optimal_tac': iter_result.optimal_tac,
            }
            output['iterations'].append(iter_dict)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"iso_result_{timestamp}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to: {filename}")
        
        return filename
    
    # ════════════════════════════════════════════════════════════════════════
    # COMPATIBILITY WITH EXISTING CODE
    # ════════════════════════════════════════════════════════════════════════
    
    @property
    def pressure_sweep_results(self):
        """Compatibility: Return pressure sweep from final iteration."""
        if self.iterations:
            return self.iterations[-1].pressure_sweep.points
        return []
    
    @property
    def nt_feed_sweep_results(self):
        """Compatibility: Return NT/Feed data for visualization."""
        # Combine NT and Feed sweeps for existing visualizer
        points = []
        for iter_result in self.iterations:
            points.extend(iter_result.nt_sweep.points)
            points.extend(iter_result.feed_sweep.points)
        return points


# ════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ITERATIVE SEQUENTIAL OPTIMIZATION (ISO)")
    print("=" * 70)
    print("""
    
This module implements TRUE ISO per professor's requirements:

OUTER LOOP (Iterations until convergence):
├── STEP 1: Optimize PRESSURE
|   * Fix NT, NF from previous iteration
|   * Sweep P with T_reb <= 120C constraint
|   * Find P* minimizing TAC
|
├── STEP 2: Optimize NT  
|   * Fix P = P*, NF from previous iteration
|   * Sweep NT to generate U-curve
|   * Find NT* at minimum
|
└── STEP 3: Optimize NF
    * Fix P = P*, NT = NT*
    * Sweep NF to generate U-curve
    * Find NF* at minimum

CONVERGENCE: When design (P*, NT*, NF*) unchanged

KEY FEATURES:
[OK] Variables optimized ONE AT A TIME
[OK] Outer iteration loop
[OK] Temperature constraint: T_reb <= 120C
[OK] Infeasibility labeling
[OK] U-curves for each variable

""")
    print("=" * 70)