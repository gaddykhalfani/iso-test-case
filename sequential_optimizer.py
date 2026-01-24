# sequential_optimizer.py
"""
Sequential Parametric Optimizer for Distillation Columns
=========================================================

Implements Option A: Academically Defensible Sequential Approach

METHODOLOGY (Physical Justification):
=====================================

Phase 1: PRESSURE OPTIMIZATION (Strategic Decision)
----------------------------------------------------
Pressure fundamentally determines the thermodynamic regime:
- Relative volatility (α) - affects separation difficulty
- Operating temperatures - determines utility selection
- Column construction - vacuum vs. atmospheric design

We optimize pressure FIRST because it sets the "landscape" on which
the NT/NF optimization occurs. This follows the hierarchy:
    Strategic (Pressure) → Tactical (NT, NF)

Phase 2: NT/NF PARAMETRIC STUDY (Tactical Decisions)
----------------------------------------------------
At fixed optimal pressure, sweep NT and NF to:
- Generate U-shaped TAC curves (classic trade-off visualization)
- Show capital vs. energy trade-off clearly
- Find optimal configuration with full insight

OUTPUT:
=======
- Multiple U-curves: TAC vs NT, one curve per feed stage
- Clear visualization of the capital-energy trade-off
- Academically defensible methodology for thesis

Author: PSE Lab, NTUST
Version: 4.0 - Sequential Parametric Approach
"""

import logging
import time
import math
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES FOR RESULTS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PressureSweepPoint:
    """Single point from pressure sweep."""
    pressure: float
    tac: float
    tpc: float
    toc: float
    q_reb: float
    q_cond: float
    diameter: float
    converged: bool
    nt: int
    feed: int


@dataclass
class NTFeedSweepPoint:
    """Single point from NT/Feed sweep."""
    nt: int
    feed: int
    pressure: float
    tac: float
    tpc: float
    toc: float
    q_reb: float
    q_cond: float
    diameter: float
    converged: bool


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    optimal_nt: int
    optimal_feed: int
    optimal_pressure: float
    optimal_tac: float
    
    # Phase 1 results
    pressure_sweep_results: List[PressureSweepPoint]
    
    # Phase 2 results
    nt_feed_sweep_results: List[NTFeedSweepPoint]
    
    # Metadata
    total_time_seconds: float
    total_evaluations: int
    case_name: str
    timestamp: str


# ════════════════════════════════════════════════════════════════════════════
# SEQUENTIAL PARAMETRIC OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════

class SequentialParametricOptimizer:
    """
    Sequential Parametric Optimizer for Distillation Columns.
    
    Implements the academically defensible approach:
    1. Optimize pressure first (strategic decision)
    2. Then sweep NT/NF at optimal pressure (tactical decisions)
    
    This produces clear U-curves showing the capital-energy trade-off.
    """
    
    def __init__(self, evaluator, config, purity_spec=None):
        """
        Initialize optimizer.

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
        
        # Optimization settings
        self.pressure_sweep_points = config.get('pressure_sweep_points', 9)
        self.nt_step = config.get('nt_step', 2)
        self.feed_step = config.get('feed_step', 2)
        self.min_section_stages = config.get('min_section_stages', 3)
        
        # Golden section refinement
        self.pressure_refine = config.get('pressure_refine', True)
        self.pressure_refine_tol = config.get('pressure_refine_tol', 0.005)
        
        # Results storage
        self.pressure_sweep_results = []
        self.nt_feed_sweep_results = []
        self.cache = {}  # All evaluated points
        
        # Statistics
        self.eval_count = 0
        self.start_time = None
    
    def run(self, case_name: str = "Case") -> OptimizationResult:
        """
        Run the complete sequential parametric optimization.
        
        Parameters
        ----------
        case_name : str
            Name for this optimization case
            
        Returns
        -------
        OptimizationResult : Complete results
        """
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("SEQUENTIAL PARAMETRIC OPTIMIZATION")
        logger.info("=" * 70)
        logger.info(f"Case: {case_name}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info("")
        logger.info("Methodology: Option A (Pressure First, then NT/NF)")
        logger.info("  Phase 1: Pressure sweep at representative NT/NF")
        logger.info("  Phase 2: Full NT/NF sweep at optimal pressure")
        logger.info("=" * 70)
        
        # ════════════════════════════════════════════════════════════════════
        # PHASE 1: PRESSURE OPTIMIZATION
        # ════════════════════════════════════════════════════════════════════
        
        optimal_pressure = self._phase1_pressure_optimization()
        
        # ════════════════════════════════════════════════════════════════════
        # PHASE 2: NT/NF PARAMETRIC SWEEP
        # ════════════════════════════════════════════════════════════════════
        
        optimal_nt, optimal_feed, optimal_tac = self._phase2_nt_feed_sweep(optimal_pressure)
        
        # ════════════════════════════════════════════════════════════════════
        # BUILD RESULTS
        # ════════════════════════════════════════════════════════════════════
        
        total_time = time.time() - self.start_time
        
        result = OptimizationResult(
            optimal_nt=optimal_nt,
            optimal_feed=optimal_feed,
            optimal_pressure=optimal_pressure,
            optimal_tac=optimal_tac,
            pressure_sweep_results=self.pressure_sweep_results,
            nt_feed_sweep_results=self.nt_feed_sweep_results,
            total_time_seconds=total_time,
            total_evaluations=self.eval_count,
            case_name=case_name,
            timestamp=timestamp,
        )
        
        # Store result as attribute for visualization
        self.result = result
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: PRESSURE OPTIMIZATION
    # ════════════════════════════════════════════════════════════════════════
    
    def _phase1_pressure_optimization(self) -> float:
        """
        Phase 1: Find optimal operating pressure.
        
        Physical Justification:
        ----------------------
        Pressure determines the thermodynamic regime:
        - Low pressure (vacuum): Higher α, easier separation, but vacuum costs
        - Higher pressure: Lower α, harder separation, but simpler construction
        
        We use representative NT and NF (midpoints of ranges) because:
        - The optimal pressure trend is relatively insensitive to exact NT/NF
        - We want to find the "operating regime" first
        
        Returns
        -------
        float : Optimal pressure (bar)
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 1: PRESSURE OPTIMIZATION")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Physical basis: Pressure sets the thermodynamic regime")
        logger.info("  - Affects relative volatility (separation difficulty)")
        logger.info("  - Determines utility selection (vacuum vs. atmospheric)")
        logger.info("  - Sets the 'landscape' for NT/NF optimization")
        logger.info("")
        
        # Use representative NT and NF (midpoints)
        repr_nt = (self.nt_bounds[0] + self.nt_bounds[1]) // 2
        repr_feed = repr_nt // 2  # Approximately middle of column
        
        # Ensure valid configuration
        repr_feed = max(repr_feed, self.min_section_stages + 1)
        repr_feed = min(repr_feed, repr_nt - self.min_section_stages)
        
        logger.info(f"Representative configuration: NT={repr_nt}, Feed={repr_feed}")
        logger.info(f"Pressure range: {self.pressure_bounds[0]:.3f} to {self.pressure_bounds[1]:.3f} bar")
        logger.info("")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Coarse pressure sweep
        # ─────────────────────────────────────────────────────────────────────
        
        logger.info("-" * 50)
        logger.info("Step 1: Coarse Pressure Sweep")
        logger.info("-" * 50)
        
        p_min, p_max = self.pressure_bounds
        pressures = [
            p_min + i * (p_max - p_min) / (self.pressure_sweep_points - 1)
            for i in range(self.pressure_sweep_points)
        ]
        
        self.pressure_sweep_results = []
        
        logger.info("")
        logger.info(f"{'Pressure (bar)':^15} {'TAC ($/yr)':^15} {'Q_reb (kW)':^12} {'Status':^10}")
        logger.info("-" * 55)
        
        for p in pressures:
            result = self._evaluate(repr_nt, repr_feed, p)
            
            point = PressureSweepPoint(
                pressure=p,
                tac=result.get('TAC', float('inf')),
                tpc=result.get('TPC', 0),
                toc=result.get('TOC', 0),
                q_reb=result.get('Q_reb', 0),
                q_cond=result.get('Q_cond', 0),
                diameter=result.get('diameter', 0),
                converged=result.get('converged', False),
                nt=repr_nt,
                feed=repr_feed,
            )
            self.pressure_sweep_results.append(point)
            
            status = "OK" if point.converged and point.tac < 1e10 else "FAIL"
            logger.info(f"{p:^15.4f} {point.tac:^15,.0f} {point.q_reb:^12.1f} {status:^10}")
        
        # Find best from coarse sweep
        valid_points = [p for p in self.pressure_sweep_results if p.converged and p.tac < 1e10]
        
        if not valid_points:
            logger.error("No valid pressure points found!")
            return (p_min + p_max) / 2  # Return midpoint as fallback
        
        best_coarse = min(valid_points, key=lambda x: x.tac)
        logger.info("")
        logger.info(f"Best from coarse sweep: P={best_coarse.pressure:.4f} bar, TAC=${best_coarse.tac:,.0f}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Golden Section Refinement (Optional)
        # ─────────────────────────────────────────────────────────────────────
        
        if self.pressure_refine:
            logger.info("")
            logger.info("-" * 50)
            logger.info("Step 2: Golden Section Refinement")
            logger.info("-" * 50)
            
            optimal_pressure = self._golden_section_pressure(
                repr_nt, repr_feed, 
                best_coarse.pressure - 0.05, 
                best_coarse.pressure + 0.05
            )
            
            # Final evaluation at optimal pressure
            final_result = self._evaluate(repr_nt, repr_feed, optimal_pressure)
            final_tac = final_result.get('TAC', best_coarse.tac)
            
            logger.info("")
            logger.info(f"Refined optimal: P={optimal_pressure:.4f} bar, TAC=${final_tac:,.0f}")
            
            improvement = best_coarse.tac - final_tac
            if improvement > 0:
                logger.info(f"  (Improved by ${improvement:,.0f} from coarse sweep)")
        else:
            optimal_pressure = best_coarse.pressure
        
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"PHASE 1 RESULT: Optimal Pressure = {optimal_pressure:.4f} bar")
        logger.info("=" * 50)
        
        return optimal_pressure
    
    def _golden_section_pressure(self, nt: int, feed: int, 
                                  p_low: float, p_high: float) -> float:
        """
        Refine pressure using Golden Section Search.
        
        Parameters
        ----------
        nt, feed : int
            Fixed configuration
        p_low, p_high : float
            Pressure bracket
            
        Returns
        -------
        float : Optimal pressure
        """
        # Ensure bounds are valid
        p_low = max(p_low, self.pressure_bounds[0])
        p_high = min(p_high, self.pressure_bounds[1])
        
        phi = (1 + math.sqrt(5)) / 2
        resphi = 2 - phi
        
        a, b = p_low, p_high
        c = b - resphi * (b - a)
        d = a + resphi * (b - a)
        
        result_c = self._evaluate(nt, feed, c)
        result_d = self._evaluate(nt, feed, d)
        
        fc = result_c.get('TAC', float('inf'))
        fd = result_d.get('TAC', float('inf'))
        
        iteration = 0
        max_iter = 20
        
        while abs(b - a) > self.pressure_refine_tol and iteration < max_iter:
            iteration += 1
            
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - resphi * (b - a)
                result_c = self._evaluate(nt, feed, c)
                fc = result_c.get('TAC', float('inf'))
            else:
                a = c
                c = d
                fc = fd
                d = a + resphi * (b - a)
                result_d = self._evaluate(nt, feed, d)
                fd = result_d.get('TAC', float('inf'))
        
        return (a + b) / 2
    
    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: NT/NF PARAMETRIC SWEEP
    # ════════════════════════════════════════════════════════════════════════
    
    def _phase2_nt_feed_sweep(self, optimal_pressure: float) -> Tuple[int, int, float]:
        """
        Phase 2: Parametric sweep of NT and NF at fixed pressure.
        
        Physical Justification:
        ----------------------
        At fixed pressure (thermodynamic regime), NT and NF represent the
        capital-energy trade-off:
        - More stages (high NT): Lower energy, higher capital → right side of U
        - Fewer stages (low NT): Higher energy, lower capital → left side of U
        - Feed location affects minimum reflux and efficiency
        
        This generates the classic U-shaped TAC curve.
        
        Returns
        -------
        tuple : (optimal_nt, optimal_feed, optimal_tac)
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 2: NT/NF PARAMETRIC SWEEP")
        logger.info("=" * 70)
        logger.info("")
        logger.info(f"Fixed pressure: {optimal_pressure:.4f} bar")
        logger.info("")
        logger.info("Physical basis: NT/NF trade-off at fixed thermodynamic regime")
        logger.info("  - More stages → Lower energy (less reflux needed)")
        logger.info("  - Fewer stages → Higher energy (more reflux needed)")
        logger.info("  - This creates the classic U-shaped TAC curve")
        logger.info("")
        
        nt_min, nt_max = self.nt_bounds
        feed_min, feed_max = self.feed_bounds
        
        logger.info(f"NT range: {nt_min} to {nt_max} (step={self.nt_step})")
        logger.info(f"Feed range: {feed_min} to {feed_max} (step={self.feed_step})")
        logger.info("")
        
        self.nt_feed_sweep_results = []
        
        # Generate NT values
        nt_values = list(range(nt_min, nt_max + 1, self.nt_step))
        
        total_points = 0
        for nt in nt_values:
            f_min = max(feed_min, self.min_section_stages + 1)
            f_max = min(feed_max, nt - self.min_section_stages)
            if f_min <= f_max:
                total_points += len(range(f_min, f_max + 1, self.feed_step))
        
        logger.info(f"Total points to evaluate: ~{total_points}")
        logger.info("")
        logger.info("-" * 70)
        
        evaluated = 0
        best_tac = float('inf')
        best_nt = nt_min
        best_feed = feed_min
        
        for nt in nt_values:
            # Calculate valid feed range for this NT
            f_min = max(feed_min, self.min_section_stages + 1)
            f_max = min(feed_max, nt - self.min_section_stages)
            
            if f_min > f_max:
                continue
            
            feed_values = list(range(f_min, f_max + 1, self.feed_step))
            
            logger.info("")
            logger.info(f"NT = {nt} (Feed: {f_min} to {f_max})")
            logger.info("-" * 40)
            
            row_results = []
            
            for feed in feed_values:
                evaluated += 1
                
                result = self._evaluate(nt, feed, optimal_pressure)
                tac = result.get('TAC', float('inf'))
                converged = result.get('converged', False)
                
                point = NTFeedSweepPoint(
                    nt=nt,
                    feed=feed,
                    pressure=optimal_pressure,
                    tac=tac,
                    tpc=result.get('TPC', 0),
                    toc=result.get('TOC', 0),
                    q_reb=result.get('Q_reb', 0),
                    q_cond=result.get('Q_cond', 0),
                    diameter=result.get('diameter', 0),
                    converged=converged,
                )
                self.nt_feed_sweep_results.append(point)
                
                if converged and tac < 1e10:
                    row_results.append((feed, tac))
                    
                    if tac < best_tac:
                        best_tac = tac
                        best_nt = nt
                        best_feed = feed
                        logger.info(f"  Feed={feed}: TAC=${tac:,.0f} *** NEW BEST ***")
                    else:
                        logger.info(f"  Feed={feed}: TAC=${tac:,.0f}")
                else:
                    logger.info(f"  Feed={feed}: FAILED")
            
            # Show best for this NT
            if row_results:
                best_in_row = min(row_results, key=lambda x: x[1])
                logger.info(f"  -> Best at NT={nt}: Feed={best_in_row[0]}, TAC=${best_in_row[1]:,.0f}")
        
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"PHASE 2 RESULT:")
        logger.info(f"  Optimal NT: {best_nt}")
        logger.info(f"  Optimal Feed: {best_feed}")
        logger.info(f"  Optimal TAC: ${best_tac:,.0f}/year")
        logger.info("=" * 50)
        
        return best_nt, best_feed, best_tac
    
    # ════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ════════════════════════════════════════════════════════════════════════
    
    def _evaluate(self, nt: int, feed: int, pressure: float) -> Dict:
        """
        Evaluate a configuration (with caching).
        
        Parameters
        ----------
        nt : int
            Number of stages
        feed : int
            Feed stage
        pressure : float
            Pressure (bar)
            
        Returns
        -------
        dict : Evaluation result
        """
        # Check cache
        key = (nt, feed, round(pressure, 4))
        if key in self.cache:
            return self.cache[key]
        
        # Evaluate (with diagnostic on failure)
        self.eval_count += 1
        result = self.evaluator.evaluate(nt, feed, pressure, run_diagnostic_on_fail=True, rr_sweep_on_fail=True, purity_spec=self.purity_spec)

        # Cache result
        self.cache[key] = result
        
        return result
    
    def _print_summary(self, result: OptimizationResult):
        """Print optimization summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("OPTIMIZATION COMPLETE - SUMMARY")
        logger.info("=" * 70)
        logger.info("")
        logger.info(f"Case: {result.case_name}")
        logger.info(f"Time: {result.total_time_seconds:.1f} seconds ({result.total_time_seconds/60:.1f} minutes)")
        logger.info(f"Total Evaluations: {result.total_evaluations}")
        logger.info("")
        logger.info("-" * 40)
        logger.info("OPTIMAL CONFIGURATION")
        logger.info("-" * 40)
        logger.info(f"  Number of Stages (NT): {result.optimal_nt}")
        logger.info(f"  Feed Stage (NF): {result.optimal_feed}")
        logger.info(f"  Operating Pressure: {result.optimal_pressure:.4f} bar")
        logger.info("")
        logger.info(f"  TAC: ${result.optimal_tac:,.0f}/year")
        logger.info("")
        logger.info("=" * 70)
    
    def get_ucurve_data(self) -> Dict:
        """
        Get data formatted for U-curve plotting.
        
        Returns dictionary with:
        - pressure_curve: TAC vs Pressure data
        - nt_curves: Dict of {feed_stage: [(nt, tac), ...]}
        - feed_curves: Dict of {nt: [(feed, tac), ...]}
        """
        # Pressure curve data
        pressure_curve = [
            (p.pressure, p.tac) 
            for p in self.pressure_sweep_results 
            if p.converged and p.tac < 1e10
        ]
        
        # NT curves (one per feed stage)
        nt_curves = {}
        for point in self.nt_feed_sweep_results:
            if point.converged and point.tac < 1e10:
                feed = point.feed
                if feed not in nt_curves:
                    nt_curves[feed] = []
                nt_curves[feed].append((point.nt, point.tac))
        
        # Sort each curve by NT
        for feed in nt_curves:
            nt_curves[feed].sort(key=lambda x: x[0])
        
        # Feed curves (one per NT)
        feed_curves = {}
        for point in self.nt_feed_sweep_results:
            if point.converged and point.tac < 1e10:
                nt = point.nt
                if nt not in feed_curves:
                    feed_curves[nt] = []
                feed_curves[nt].append((point.feed, point.tac))
        
        # Sort each curve by feed
        for nt in feed_curves:
            feed_curves[nt].sort(key=lambda x: x[0])
        
        return {
            'pressure_curve': pressure_curve,
            'nt_curves': nt_curves,
            'feed_curves': feed_curves,
        }
    
    def save_results(self, output_dir: str = "results") -> str:
        """
        Save all results to JSON file.
        
        Parameters
        ----------
        output_dir : str
            Output directory
            
        Returns
        -------
        str : Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get optimal result
        valid_points = [p for p in self.nt_feed_sweep_results if p.converged and p.tac < 1e10]
        if valid_points:
            best = min(valid_points, key=lambda x: x.tac)
        else:
            return None
        
        # Build output dictionary
        output = {
            'metadata': {
                'methodology': 'Sequential Parametric Optimization (Option A)',
                'description': 'Pressure first (strategic), then NT/NF (tactical)',
                'total_evaluations': self.eval_count,
                'total_time_seconds': time.time() - self.start_time if self.start_time else 0,
            },
            'optimal': {
                'nt': best.nt,
                'feed': best.feed,
                'pressure': best.pressure,
                'tac': best.tac,
                'tpc': best.tpc,
                'toc': best.toc,
            },
            'pressure_sweep': [asdict(p) for p in self.pressure_sweep_results],
            'nt_feed_sweep': [asdict(p) for p in self.nt_feed_sweep_results],
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"sequential_opt_{timestamp}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to: {filename}")
        
        return filename


# ════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SEQUENTIAL PARAMETRIC OPTIMIZER")
    print("Option A: Pressure First, then NT/NF")
    print("=" * 70)
    print("""
    
This module implements the academically defensible sequential approach:

PHASE 1: PRESSURE OPTIMIZATION (Strategic)
------------------------------------------
• Sweep pressure at representative NT/NF
• Find optimal thermodynamic regime
• Physical basis: Pressure affects α, temperatures, utilities

PHASE 2: NT/NF PARAMETRIC SWEEP (Tactical)
------------------------------------------
• Fix pressure at optimal value
• Full sweep of NT and NF
• Generate U-shaped curves for thesis

OUTPUT:
-------
• TAC vs Pressure curve
• TAC vs NT curves (one per feed stage) - Classic U-shape
• TAC vs Feed curves (one per NT)
• Clear visualization of capital-energy trade-off

""")
    print("=" * 70)