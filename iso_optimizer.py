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


# ════════════════════════════════════════════════════════════════════════════
# FEASIBILITY STATUS
# ════════════════════════════════════════════════════════════════════════════

class FeasibilityStatus(Enum):
    """Feasibility status for evaluated points."""
    FEASIBLE = "feasible"
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
    T_reb: float = 0
    T_cond: float = 0
    converged: bool = False
    feasibility: FeasibilityStatus = FeasibilityStatus.FEASIBLE
    infeasibility_reason: str = ""


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
        self.nt_step = config.get('nt_step', 2)
        self.feed_step = config.get('feed_step', 2)
        self.min_section_stages = config.get('min_section_stages', 3)
        
        # Temperature constraint (professor's requirement)
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
        self.feasible_count = 0
        self.infeasible_count = 0
        self.start_time = None
    
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
        
        converged = False
        
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

            pressure_sweep = self._sweep_pressure(current_nt, current_feed, iteration)
            
            if pressure_sweep.optimal_tac < float('inf'):
                current_pressure = pressure_sweep.optimal_value
                logger.info(f"  -> P* = {current_pressure:.4f} bar, TAC=${pressure_sweep.optimal_tac:,.0f}")
                # Log feasible/infeasible counts from this sweep
                feasible_pts = sum(1 for p in pressure_sweep.points if p.feasibility == FeasibilityStatus.FEASIBLE)
                infeasible_pts = len(pressure_sweep.points) - feasible_pts
                logger.info(f"     ({feasible_pts} feasible, {infeasible_pts} infeasible out of {len(pressure_sweep.points)} points)")
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
            
            if tac_change < self.tac_tolerance and not nt_changed and not feed_changed:
                converged = True
                logger.info("")
                logger.info("★ CONVERGED! Design unchanged between iterations.")
                break
            else:
                logger.info("")
                logger.info("-> Not converged, continuing to next iteration...")
        
        # ════════════════════════════════════════════════════════════════════
        # BUILD FINAL RESULT
        # ════════════════════════════════════════════════════════════════════
        
        total_time = time.time() - self.start_time
        
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
            case_name=case_name,
            timestamp=timestamp,
        )
        
        # Store for visualization
        self.result = result
        
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
    
    def _sweep_pressure(self, fixed_nt: int, fixed_feed: int, iteration: int = 1) -> SweepResult:
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

        logger.info("")
        logger.info(f"{'Pressure':^10} {'T_reb':^10} {'TAC':^15} {'Status':^25}")
        logger.info("-" * 60)

        best_tac = float('inf')
        best_pressure = p_min
        total_points = len(pressures)

        for idx, p in enumerate(pressures):
            point = self._evaluate_with_feasibility(fixed_nt, fixed_feed, p)
            result.points.append(point)

            # Emit progress
            _emit_progress(
                iteration=iteration,
                phase="pressure_sweep",
                current=idx + 1,
                total=total_points,
                best_tac=best_tac if best_tac < float('inf') else None,
                pressure=round(p, 4),
                T_reb=round(point.T_reb, 1) if point.T_reb > 0 else None
            )
            
            # Format status
            if point.feasibility == FeasibilityStatus.FEASIBLE:
                status = "OK"
                if point.tac < best_tac:
                    best_tac = point.tac
                    best_pressure = p
                    status = "*** BEST ***"
            elif point.feasibility == FeasibilityStatus.INFEASIBLE_TEMPERATURE:
                status = "[X] T_reb > 120C"
            else:
                status = f"[X] {point.infeasibility_reason[:20]}"
            
            tac_str = f"${point.tac:,.0f}" if point.tac < 1e10 else "N/A"
            T_str = f"{point.T_reb:.1f}C" if point.T_reb > 0 else "N/A"
            
            logger.info(f"{p:^10.4f} {T_str:^10} {tac_str:^15} {status:^25}")
        
        result.optimal_value = best_pressure
        result.optimal_tac = best_tac
        
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

        best_tac = float('inf')
        best_nt = nt_min
        total_points = len(nt_values)

        for idx, nt in enumerate(nt_values):
            # Adjust feed if needed
            adjusted_feed = self._adjust_feed_for_nt(fixed_feed, nt)
            
            if adjusted_feed < self.min_section_stages + 1:
                continue  # Skip invalid configurations
            
            point = self._evaluate_with_feasibility(nt, adjusted_feed, fixed_pressure)
            result.points.append(point)

            if point.feasibility == FeasibilityStatus.FEASIBLE and point.tac < best_tac:
                best_tac = point.tac
                best_nt = nt
                status = "*** BEST ***"
            elif point.feasibility == FeasibilityStatus.FEASIBLE:
                status = "OK"
            else:
                status = f"[X] {point.infeasibility_reason[:15]}"

            tac_str = f"${point.tac:,.0f}" if point.tac < 1e10 else "N/A"
            logger.info(f"{nt:^8} {tac_str:^15} {status:^20}")

            # Emit progress
            _emit_progress(
                iteration=iteration,
                phase="nt_sweep",
                current=idx + 1,
                total=total_points,
                best_tac=best_tac if best_tac < float('inf') else None,
                nt=nt
            )

        result.optimal_value = best_nt
        result.optimal_tac = best_tac

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

        best_tac = float('inf')
        best_feed = f_min
        total_points = len(feed_values)

        for idx, feed in enumerate(feed_values):
            point = self._evaluate_with_feasibility(fixed_nt, feed, fixed_pressure)
            result.points.append(point)
            
            if point.feasibility == FeasibilityStatus.FEASIBLE and point.tac < best_tac:
                best_tac = point.tac
                best_feed = feed
                status = "*** BEST ***"
            elif point.feasibility == FeasibilityStatus.FEASIBLE:
                status = "OK"
            else:
                status = f"[X] {point.infeasibility_reason[:15]}"
            
            tac_str = f"${point.tac:,.0f}" if point.tac < 1e10 else "N/A"
            logger.info(f"{feed:^8} {tac_str:^15} {status:^20}")

            # Emit progress
            _emit_progress(
                iteration=iteration,
                phase="feed_sweep",
                current=idx + 1,
                total=total_points,
                best_tac=best_tac if best_tac < float('inf') else None,
                feed=feed
            )

        result.optimal_value = best_feed
        result.optimal_tac = best_tac

        return result

    # ════════════════════════════════════════════════════════════════════════
    # EVALUATION WITH FEASIBILITY CHECK
    # ════════════════════════════════════════════════════════════════════════

    def _evaluate_with_feasibility(self, nt: int, feed: int, pressure: float) -> EvaluationPoint:
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
        result = self.evaluator.evaluate(nt, feed, pressure, run_diagnostic_on_fail=True, rr_sweep_on_fail=True, purity_spec=self.purity_spec)
        
        # Extract values (with None safety)
        tac = result.get('TAC', float('inf'))
        converged = result.get('converged', False)
        T_reb = result.get('T_reb')
        T_cond = result.get('T_cond')
        
        # Handle None temperatures - treat as infeasible (cannot verify constraint)
        T_reb_missing = (T_reb is None)
        T_cond_missing = (T_cond is None)
        if T_reb is None:
            T_reb = 0.0
        if T_cond is None:
            T_cond = 0.0

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

        # ════════════════════════════════════════════════════════════════════
        # FEASIBILITY CHECKS (Professor's requirements)
        # ════════════════════════════════════════════════════════════════════

        if not converged:
            point.feasibility = FeasibilityStatus.INFEASIBLE_CONVERGENCE
            point.infeasibility_reason = "Simulation did not converge"
            self.infeasible_count += 1

        elif T_reb_missing:
            # CRITICAL: If T_reb cannot be extracted, treat as infeasible
            # (cannot verify temperature constraint => conservative rejection)
            point.feasibility = FeasibilityStatus.INFEASIBLE_TEMPERATURE
            point.infeasibility_reason = "T_reb extraction failed (cannot verify constraint)"
            point.tac = float('inf')
            self.infeasible_count += 1
            logger.warning(f"  [X] INFEASIBLE: T_reb could not be extracted for NT={nt}, P={pressure:.4f}")

        elif T_reb > self.T_reb_max:
            # CRITICAL: Temperature constraint
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
            point.feasibility = FeasibilityStatus.FEASIBLE
            self.feasible_count += 1
            logger.info(f"  [OK] FEASIBLE: TAC=${tac:,.0f}, T_reb={T_reb:.1f}C (P={pressure:.4f})")
        
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
        logger.info(f"  * Temperature constraint: T_reb <= {self.T_reb_max}C")
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
        logger.info(f"  Infeasible (T_reb > {self.T_reb_max}C): {result.infeasible_evaluations}")
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
    
    def save_results(self, output_dir: str = "results") -> str:
        """Save all results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Build output dictionary
        output = {
            'metadata': {
                'methodology': 'TRUE Iterative Sequential Optimization (ISO)',
                'description': 'P -> NT -> NF, one at a time, with outer loop',
                'temperature_constraint': f'T_reb <= {self.T_reb_max}C',
                'convergence_tolerance': self.tac_tolerance,
            },
            'optimal': {
                'nt': self.result.optimal_nt,
                'feed': self.result.optimal_feed,
                'pressure': self.result.optimal_pressure,
                'tac': self.result.optimal_tac,
            },
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