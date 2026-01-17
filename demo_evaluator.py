"""
demo_evaluator.py
=================
Mock evaluator for testing optimization algorithms without Aspen Plus.

Generates realistic TAC values based on a U-curve model that simulates
the behavior of a distillation column (EB/SM separation).

Usage:
    from demo_evaluator import DemoEvaluator
    evaluator = DemoEvaluator(delay=0.1)
    result = evaluator.evaluate(nt=45, feed=22, pressure=0.2)
"""

import numpy as np
import time
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class DemoConfig:
    """Configuration for demo evaluator behavior."""
    # Optimal point (realistic for EB/SM column)
    optimal_nt: int = 45
    optimal_feed: int = 22
    optimal_pressure: float = 0.2
    base_tac: float = 125000.0  # $/year

    # U-curve sensitivity parameters
    nt_sensitivity: float = 800.0      # TAC penalty per (NT - optimal)^2
    feed_sensitivity: float = 500.0    # TAC penalty per (feed - optimal)^2
    pressure_sensitivity: float = 20000.0  # TAC penalty per (P - optimal)^2

    # Temperature model (T_reb increases at lower pressure)
    t_reb_base: float = 90.0          # Base reboiler temperature at P=0.5
    t_reb_pressure_coeff: float = 30.0  # Temperature increase per 0.5 bar decrease
    t_cond_offset: float = 30.0       # T_cond is typically 30C below T_reb

    # Noise and timing
    noise_std: float = 0.02           # 2% noise on TAC
    temp_noise_std: float = 2.0       # Temperature noise (C)
    delay: float = 0.1                # Simulated evaluation time (seconds)


class DemoEvaluator:
    """
    Mock evaluator that simulates realistic TAC calculations for distillation columns.

    The evaluator generates U-curve behavior where TAC increases as parameters
    move away from the optimal point. It also simulates:
    - Temperature constraint behavior (T_reb increases at lower pressure)
    - Convergence failures for extreme parameters
    - Realistic energy duties and column dimensions

    Parameters
    ----------
    config : DemoConfig, optional
        Configuration for the evaluator behavior
    delay : float, optional
        Override delay in seconds (default from config)
    noise_std : float, optional
        Override noise standard deviation (default from config)
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        config: Optional[DemoConfig] = None,
        delay: Optional[float] = None,
        noise_std: Optional[float] = None,
        seed: Optional[int] = None
    ):
        self.config = config or DemoConfig()

        # Override config values if provided
        if delay is not None:
            self.config.delay = delay
        if noise_std is not None:
            self.config.noise_std = noise_std

        # Statistics tracking
        self.eval_count = 0
        self.feasible_count = 0
        self.infeasible_count = 0
        self.cache_hits = 0
        self._cache: Dict[tuple, Dict] = {}

        # Random state
        if seed is not None:
            np.random.seed(seed)

    def evaluate(self, nt: int, feed: int, pressure: float) -> Dict:
        """
        Evaluate TAC for given column configuration.

        Parameters
        ----------
        nt : int
            Number of stages
        feed : int
            Feed stage location
        pressure : float
            Column pressure (bar)

        Returns
        -------
        dict
            Evaluation result with keys:
            - TAC: Total Annual Cost ($/year)
            - TPC: Total Plant Cost ($)
            - TOC: Total Operating Cost ($/year)
            - Q_reb: Reboiler duty (kW)
            - Q_cond: Condenser duty (kW)
            - diameter: Column diameter (m)
            - NT: Number of stages
            - feed: Feed stage
            - pressure: Column pressure (bar)
            - converged: bool
            - T_reb: Reboiler temperature (C)
            - T_cond: Condenser temperature (C)
        """
        self.eval_count += 1

        # Check cache
        cache_key = (nt, feed, round(pressure, 4))
        if cache_key in self._cache:
            self.cache_hits += 1
            return self._cache[cache_key].copy()

        # Simulate computation time
        if self.config.delay > 0:
            time.sleep(self.config.delay)

        # Check convergence (fails for extreme parameters)
        converged = self._check_convergence(nt, feed, pressure)

        if not converged:
            self.infeasible_count += 1
            result = self._create_failed_result(nt, feed, pressure)
            self._cache[cache_key] = result
            return result.copy()

        # Calculate TAC with U-curve behavior
        tac = self._calculate_tac(nt, feed, pressure)

        # Calculate temperature (T_reb increases at lower pressure)
        T_reb = self._calculate_t_reb(pressure)
        T_cond = T_reb - self.config.t_cond_offset + np.random.normal(0, 2)

        # Calculate other properties
        Q_reb = self._calculate_q_reb(nt, pressure)
        Q_cond = self._calculate_q_cond(nt, pressure)
        diameter = self._calculate_diameter(nt, pressure)

        # Cost breakdown (approximate)
        tpc = tac * 0.4 * 3  # Capital cost (3-year payback)
        toc = tac * 0.6      # Operating cost

        self.feasible_count += 1

        result = {
            'TAC': tac,
            'TPC': tpc,
            'TOC': toc,
            'Q_reb': Q_reb,
            'Q_cond': Q_cond,
            'diameter': diameter,
            'NT': nt,
            'feed': feed,
            'pressure': pressure,
            'converged': True,
            'T_reb': T_reb,
            'T_cond': T_cond,
            'reflux_ratio': 2.5 + np.random.normal(0, 0.2),
            'distillate_rate': 100 + np.random.normal(0, 5),
        }

        self._cache[cache_key] = result
        return result.copy()

    def _check_convergence(self, nt: int, feed: int, pressure: float) -> bool:
        """Check if configuration would converge in Aspen."""
        # Fails for extreme NT
        if nt < 15 or nt > 80:
            return False

        # Fails if feed too close to ends
        if feed < 5 or feed > nt - 3:
            return False

        # Fails for extreme pressure
        if pressure < 0.05 or pressure > 1.5:
            return False

        # Small random chance of failure (realistic)
        if np.random.random() < 0.02:
            return False

        return True

    def _calculate_tac(self, nt: int, feed: int, pressure: float) -> float:
        """Calculate TAC with U-curve behavior."""
        cfg = self.config

        # U-curve penalties (quadratic)
        nt_penalty = cfg.nt_sensitivity * ((nt - cfg.optimal_nt) ** 2) / cfg.optimal_nt
        feed_penalty = cfg.feed_sensitivity * ((feed - cfg.optimal_feed) ** 2) / cfg.optimal_feed
        pressure_penalty = cfg.pressure_sensitivity * ((pressure - cfg.optimal_pressure) ** 2)

        # Base TAC + penalties
        tac = cfg.base_tac + nt_penalty + feed_penalty + pressure_penalty

        # Add noise
        tac *= (1 + np.random.normal(0, cfg.noise_std))

        return max(tac, cfg.base_tac * 0.8)  # Floor at 80% of base

    def _calculate_t_reb(self, pressure: float) -> float:
        """Calculate reboiler temperature (increases at lower pressure for vacuum)."""
        cfg = self.config
        # T_reb = base + coefficient * (0.5 - pressure)
        # At P=0.5: T_reb = 90C
        # At P=0.1: T_reb = 90 + 30*0.4 = 102C
        # At P=0.05: T_reb = 90 + 30*0.45 = 103.5C
        T_reb = cfg.t_reb_base + cfg.t_reb_pressure_coeff * (0.5 - pressure)
        T_reb += np.random.normal(0, cfg.temp_noise_std)
        return T_reb

    def _calculate_q_reb(self, nt: int, pressure: float) -> float:
        """Calculate reboiler duty (kW)."""
        # Base duty + effect of NT and pressure
        base_duty = 200
        nt_effect = 2 * (nt - 45)  # More stages = more duty
        pressure_effect = -50 * (pressure - 0.2)  # Lower pressure = higher duty

        Q_reb = base_duty + nt_effect + pressure_effect
        Q_reb += np.random.normal(0, 20)
        return max(Q_reb, 100)

    def _calculate_q_cond(self, nt: int, pressure: float) -> float:
        """Calculate condenser duty (kW)."""
        # Typically slightly higher than reboiler duty
        Q_reb = self._calculate_q_reb(nt, pressure)
        Q_cond = Q_reb * 1.1 + np.random.normal(0, 30)
        return max(Q_cond, 150)

    def _calculate_diameter(self, nt: int, pressure: float) -> float:
        """Calculate column diameter (m)."""
        # Lower pressure = larger diameter (higher vapor velocity)
        base_diameter = 0.9
        pressure_effect = 0.3 * (0.5 - pressure)
        nt_effect = 0.005 * (nt - 45)

        diameter = base_diameter + pressure_effect + nt_effect
        diameter += np.random.normal(0, 0.05)
        return max(diameter, 0.5)

    def _create_failed_result(self, nt: int, feed: int, pressure: float) -> Dict:
        """Create result for failed convergence."""
        return {
            'TAC': 1e12,
            'TPC': 1e12,
            'TOC': 1e12,
            'Q_reb': 0,
            'Q_cond': 0,
            'diameter': 0,
            'NT': nt,
            'feed': feed,
            'pressure': pressure,
            'converged': False,
            'T_reb': 0,
            'T_cond': 0,
            'reflux_ratio': 0,
            'distillate_rate': 0,
        }

    def get_statistics(self) -> Dict:
        """Get evaluation statistics."""
        return {
            'total_evaluations': self.eval_count,
            'feasible': self.feasible_count,
            'infeasible': self.infeasible_count,
            'cache_hits': self.cache_hits,
            'success_rate': self.feasible_count / max(1, self.eval_count),
        }

    def reset_statistics(self):
        """Reset evaluation statistics."""
        self.eval_count = 0
        self.feasible_count = 0
        self.infeasible_count = 0
        self.cache_hits = 0

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache.clear()
        self.cache_hits = 0


# Convenience function to create evaluator with common settings
def create_demo_evaluator(
    delay: float = 0.1,
    noise: float = 0.02,
    seed: Optional[int] = None
) -> DemoEvaluator:
    """
    Create a demo evaluator with specified settings.

    Parameters
    ----------
    delay : float
        Simulated evaluation time in seconds
    noise : float
        Noise standard deviation (fraction of TAC)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    DemoEvaluator
        Configured evaluator instance
    """
    return DemoEvaluator(delay=delay, noise_std=noise, seed=seed)


if __name__ == "__main__":
    # Quick test
    print("Testing DemoEvaluator...")
    evaluator = create_demo_evaluator(delay=0.01, seed=42)

    # Test optimal point
    result = evaluator.evaluate(nt=45, feed=22, pressure=0.2)
    print(f"\nOptimal point (NT=45, Feed=22, P=0.2):")
    print(f"  TAC: ${result['TAC']:,.0f}/year")
    print(f"  T_reb: {result['T_reb']:.1f}C")
    print(f"  Converged: {result['converged']}")

    # Test non-optimal point
    result = evaluator.evaluate(nt=60, feed=30, pressure=0.1)
    print(f"\nNon-optimal point (NT=60, Feed=30, P=0.1):")
    print(f"  TAC: ${result['TAC']:,.0f}/year")
    print(f"  T_reb: {result['T_reb']:.1f}C")
    print(f"  Converged: {result['converged']}")

    # Test extreme point (should fail)
    result = evaluator.evaluate(nt=10, feed=5, pressure=0.05)
    print(f"\nExtreme point (NT=10, Feed=5, P=0.05):")
    print(f"  Converged: {result['converged']}")

    # Statistics
    print(f"\nStatistics: {evaluator.get_statistics()}")
