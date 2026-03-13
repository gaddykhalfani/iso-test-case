# visualization_iso.py
"""
Visualization Module for TRUE Iterative Sequential Optimization (ISO)
======================================================================

Features:
- U-curves with infeasibility markers
- Convergence history across iterations
- Temperature constraint visualization
- Reflux diagnostic plots (for unconverged cases)
- **NEW: Multiple U-Curves visualization**

Author: PSE Lab, NTUST
Version: 5.1 - ISO Visualization + Multiple U-Curves
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import feasibility enum
try:
    from iso_optimizer import FeasibilityStatus, T_REBOILER_MAX
except ImportError:
    # Fallback definitions
    class FeasibilityStatus:
        FEASIBLE = "feasible"
        INFEASIBLE_TEMPERATURE = "infeasible_temperature"
        INFEASIBLE_CONVERGENCE = "infeasible_convergence"
    T_REBOILER_MAX = 120.0


def safe_get(obj: Any, key: str, default=None):
    """Safely get attribute from dataclass or dict."""
    if hasattr(obj, key):
        return getattr(obj, key)
    elif isinstance(obj, dict):
        return obj.get(key, default)
    return default


class ISOVisualizer:
    """
    Visualization for ISO optimization results.
    
    Key features:
    - Marks infeasible points (T_reb > 120°C)
    - Shows convergence history
    - Generates publication-quality U-curves
    - **NEW: Multiple U-curves at different feed stages**
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Colors
        self.colors = {
            'feasible': '#1f77b4',      # Blue - hard feasible (naturally converged)
            'soft_feasible': '#ff7f0e', # Orange - soft feasible (RR-recovered)
            'infeasible_temp': '#d62728',  # Red
            'infeasible_conv': '#7f7f7f',  # Gray
            'optimal': '#2ca02c',       # Green
            'pressure': '#9467bd',      # Purple
            'nt': '#17becf',            # Cyan
            'feed': '#e377c2',          # Pink (changed from orange to avoid conflict)
        }
    
    def _get_smart_ylim(self, tac_values: List[float]) -> tuple:
        """Get smart Y-axis limits."""
        valid = [t for t in tac_values if t < 1e10]
        if not valid:
            return 0, 1000
        
        p5 = np.percentile(valid, 5)
        p95 = np.percentile(valid, 95)
        margin = (p95 - p5) * 0.15
        
        return max(0, p5 - margin), p95 + margin
    
    # ════════════════════════════════════════════════════════════════════════
    # PRESSURE SWEEP WITH TEMPERATURE CONSTRAINT
    # ════════════════════════════════════════════════════════════════════════
    
    def plot_pressure_sweep(self, optimizer, case_name: str, iteration: int = -1) -> str:
        """
        Plot pressure sweep with temperature constraint visualization.
        Marks infeasible points where T_reb > 120°C in RED.
        Only shows constraint when has_styrene=True.
        """
        if not optimizer.iterations:
            return None

        has_styrene = getattr(optimizer, 'has_styrene', True)
        iter_result = optimizer.iterations[iteration]
        points = iter_result.pressure_sweep.points
        
        if not points:
            return None
        
        # Separate feasible and infeasible points
        p_feasible, tac_feasible = [], []
        p_infeasible, tac_infeasible, T_infeasible = [], [], []
        
        for pt in points:
            p = safe_get(pt, 'pressure')
            tac = safe_get(pt, 'tac', 1e12)
            T_reb = safe_get(pt, 'T_reb', None)
            feasibility = safe_get(pt, 'feasibility', FeasibilityStatus.FEASIBLE)
            
            if hasattr(feasibility, 'value'):
                feasibility = feasibility.value
            
            if feasibility in ('feasible', 'soft_feasible') and tac < 1e10:
                p_feasible.append(p)
                tac_feasible.append(tac / 1000)
            elif 'temperature' in str(feasibility):
                p_infeasible.append(p)
                tac_infeasible.append(tac / 1000 if tac < 1e10 else None)
                T_infeasible.append(T_reb)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[2, 1])
        
        # TOP: TAC vs Pressure
        if p_feasible:
            ax1.plot(p_feasible, tac_feasible, 'o-', 
                    color=self.colors['feasible'], linewidth=2, markersize=8,
                    label='Feasible')
        
        if p_infeasible and tac_feasible:
            y_pos = max(tac_feasible) * 1.1
            ax1.scatter(p_infeasible, [y_pos] * len(p_infeasible),
                       marker='^', color=self.colors['infeasible_temp'], s=100,
                       label=f'Infeasible (T_reb > {T_REBOILER_MAX}°C)', zorder=5)
            for p, T in zip(p_infeasible, T_infeasible):
                if T is not None and T > 0:
                    ax1.annotate(f'{T:.0f}°C', (p, y_pos),
                               textcoords="offset points", xytext=(0, 10),
                               ha='center', fontsize=9, color='red')
        
        opt_p = iter_result.optimal_pressure
        if p_feasible:
            opt_idx = None
            for i, p in enumerate(p_feasible):
                if abs(p - opt_p) < 0.001:
                    opt_idx = i
                    break
            if opt_idx is not None:
                ax1.plot(p_feasible[opt_idx], tac_feasible[opt_idx], '*',
                        color=self.colors['optimal'], markersize=20,
                        markeredgecolor='black', markeredgewidth=1.5,
                        label=f'Optimal P = {opt_p:.4f} bar', zorder=10)
        
        ax1.axvline(x=opt_p, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        if p_feasible:
            y_min, y_max = self._get_smart_ylim(tac_feasible)
            ax1.set_ylim(y_min, y_max * 1.2)
        
        ax1.set_xlabel('Operating Pressure (bar)', fontsize=12)
        ax1.set_ylabel('TAC (×$1,000/year)', fontsize=12)
        ax1.set_title(f'{case_name} - ISO Step 1: Pressure Optimization\n'
                     f'(Iteration {abs(iteration) if iteration >= 0 else len(optimizer.iterations)})',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # BOTTOM: Temperature vs Pressure (only points with valid T_reb)
        valid_pairs = [
            (safe_get(pt, 'pressure'), safe_get(pt, 'T_reb'))
            for pt in points
            if safe_get(pt, 'T_reb') is not None and safe_get(pt, 'T_reb') > 0
        ]

        if valid_pairs:
            valid_p, valid_T = zip(*valid_pairs)
            ax2.plot(valid_p, valid_T, 's-', color='#d62728', linewidth=2, markersize=8,
                    label='T_reboiler')
            T_max_val = max(valid_T)
        else:
            valid_p, valid_T = [], []
            T_max_val = T_REBOILER_MAX + 20

        if has_styrene:
            # Show constraint line and infeasible region
            ax2.axhline(y=T_REBOILER_MAX, color='red', linestyle='--', linewidth=2,
                       label=f'Constraint: T_reb ≤ {T_REBOILER_MAX}°C')
            ax2.fill_between(ax2.get_xlim(), T_REBOILER_MAX, max(T_max_val * 1.1, T_REBOILER_MAX + 10),
                            color='red', alpha=0.1, label='Infeasible region')
            ax2.set_title('Temperature Constraint Check', fontsize=12, fontweight='bold')

            all_pressures = [safe_get(pt, 'pressure') for pt in points if safe_get(pt, 'pressure') is not None]
            if all_pressures:
                ax2.annotate('STYRENE POLYMERIZATION RISK\nT > 120°C',
                            xy=(min(all_pressures) + 0.1, T_REBOILER_MAX + 5),
                            fontsize=10, color='red', fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
        else:
            # No styrene — constraint inactive, just show T_reb for reference
            ax2.set_title('Reboiler Temperature (no styrene — constraint inactive)',
                         fontsize=12, fontweight='bold')
            ax2.annotate('T_reb constraint INACTIVE\n(no styrene in reboiler)',
                        xy=(0.5, 0.9), xycoords='axes fraction', ha='center',
                        fontsize=10, color='green', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.9))

        ax2.set_xlabel('Operating Pressure (bar)', fontsize=12)
        ax2.set_ylabel('Reboiler Temperature (°C)', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{case_name}_ISO_Pressure_Sweep.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {filepath}")
        return filepath
    
    # ════════════════════════════════════════════════════════════════════════
    # NT U-CURVE
    # ════════════════════════════════════════════════════════════════════════
    
    def plot_nt_ucurve(self, optimizer, case_name: str, iteration: int = -1) -> str:
        """Plot U-curve for NT optimization with soft/hard feasible distinction."""
        if not optimizer.iterations:
            return None

        iter_result = optimizer.iterations[iteration]
        points = iter_result.nt_sweep.points

        if not points:
            return None

        # Collect all points with feasibility flag: (nt, tac, is_soft)
        all_data = []

        for pt in points:
            nt = safe_get(pt, 'nt')
            tac = safe_get(pt, 'tac', 1e12)
            feasibility = safe_get(pt, 'feasibility', FeasibilityStatus.FEASIBLE)

            if hasattr(feasibility, 'value'):
                feasibility = feasibility.value

            if tac < 1e10:
                is_soft = (feasibility == 'soft_feasible')
                all_data.append((nt, tac / 1000, is_soft))

        if not all_data:
            return None

        # Sort by NT for continuous line
        all_data.sort(key=lambda x: x[0])

        all_nts = [d[0] for d in all_data]
        all_tacs = [d[1] for d in all_data]
        is_soft = [d[2] for d in all_data]
        has_soft = any(is_soft)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot continuous line
        ax.plot(all_nts, all_tacs, '-', color=self.colors['feasible'],
               linewidth=2.5, zorder=5)

        # Plot markers: filled for hard, hollow for soft
        hard_nts = [all_nts[j] for j in range(len(all_nts)) if not is_soft[j]]
        hard_tacs = [all_tacs[j] for j in range(len(all_tacs)) if not is_soft[j]]
        soft_nts = [all_nts[j] for j in range(len(all_nts)) if is_soft[j]]
        soft_tacs = [all_tacs[j] for j in range(len(all_tacs)) if is_soft[j]]

        if hard_nts:
            ax.scatter(hard_nts, hard_tacs, color=self.colors['feasible'], s=64,
                      marker='o', zorder=6, edgecolors='white', linewidths=0.5,
                      label='Hard Feasible (naturally converged)')

        if soft_nts:
            ax.scatter(soft_nts, soft_tacs, color=self.colors['soft_feasible'], s=64,
                      marker='o', facecolors='white', edgecolors=self.colors['soft_feasible'],
                      linewidths=2, zorder=6, label='Soft Feasible (RR-recovered)')

        opt_nt = iter_result.optimal_nt
        opt_tac = iter_result.optimal_tac / 1000

        ax.plot(opt_nt, opt_tac, '*', color=self.colors['optimal'], markersize=25,
               markeredgecolor='black', markeredgewidth=1.5,
               label=f'Optimal NT = {opt_nt}', zorder=10)
        ax.axvline(x=opt_nt, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        if len(all_nts) > 5:
            ax.annotate('↑ Energy Cost\n(High reflux)',
                       xy=(min(all_nts) + 2, max(all_tacs) * 0.95),
                       fontsize=10, style='italic',
                       bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
            ax.annotate('↑ Capital Cost\n(Tall column)',
                       xy=(max(all_nts) - 5, min(all_tacs) * 1.05),
                       fontsize=10, style='italic',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        y_min, y_max = self._get_smart_ylim(all_tacs)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel('Number of Stages (NT)', fontsize=12)
        ax.set_ylabel('TAC (×$1,000/year)', fontsize=12)

        fixed = iter_result.nt_sweep.fixed_values
        title = f'{case_name} - ISO Step 2: NT Optimization (U-Curve)\n'
        title += f'P = {fixed["pressure"]:.4f} bar, NF = {fixed["feed"]} fixed'
        if has_soft:
            title += '\n(● = natural convergence, ○ = RR-recovered)'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f"{case_name}_ISO_NT_UCurve.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved: {filepath}")
        return filepath
    
    # ════════════════════════════════════════════════════════════════════════
    # FEED U-CURVE
    # ════════════════════════════════════════════════════════════════════════
    
    def plot_feed_ucurve(self, optimizer, case_name: str, iteration: int = -1) -> str:
        """Plot U-curve for feed optimization with soft/hard feasible distinction."""
        if not optimizer.iterations:
            return None

        iter_result = optimizer.iterations[iteration]
        points = iter_result.feed_sweep.points

        if not points:
            return None

        # Collect all points with feasibility flag: (feed, tac, is_soft)
        all_data = []

        for pt in points:
            feed = safe_get(pt, 'feed')
            tac = safe_get(pt, 'tac', 1e12)
            feasibility = safe_get(pt, 'feasibility', FeasibilityStatus.FEASIBLE)

            if hasattr(feasibility, 'value'):
                feasibility = feasibility.value

            if tac < 1e10:
                is_soft = (feasibility == 'soft_feasible')
                all_data.append((feed, tac / 1000, is_soft))

        if not all_data:
            return None

        # Sort by feed for continuous line
        all_data.sort(key=lambda x: x[0])

        all_feeds = [d[0] for d in all_data]
        all_tacs = [d[1] for d in all_data]
        is_soft = [d[2] for d in all_data]
        has_soft = any(is_soft)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot continuous line
        ax.plot(all_feeds, all_tacs, '-', color=self.colors['feasible'],
               linewidth=2.5, zorder=5)

        # Plot markers: filled for hard, hollow for soft
        hard_feeds = [all_feeds[j] for j in range(len(all_feeds)) if not is_soft[j]]
        hard_tacs = [all_tacs[j] for j in range(len(all_tacs)) if not is_soft[j]]
        soft_feeds = [all_feeds[j] for j in range(len(all_feeds)) if is_soft[j]]
        soft_tacs = [all_tacs[j] for j in range(len(all_tacs)) if is_soft[j]]

        if hard_feeds:
            ax.scatter(hard_feeds, hard_tacs, color=self.colors['feasible'], s=64,
                      marker='s', zorder=6, edgecolors='white', linewidths=0.5,
                      label='Hard Feasible (naturally converged)')

        if soft_feeds:
            ax.scatter(soft_feeds, soft_tacs, color=self.colors['soft_feasible'], s=64,
                      marker='s', facecolors='white', edgecolors=self.colors['soft_feasible'],
                      linewidths=2, zorder=6, label='Soft Feasible (RR-recovered)')

        opt_feed = iter_result.optimal_feed
        opt_tac = iter_result.optimal_tac / 1000

        ax.plot(opt_feed, opt_tac, '*', color=self.colors['optimal'], markersize=25,
               markeredgecolor='black', markeredgewidth=1.5,
               label=f'Optimal NF = {opt_feed}', zorder=10)
        ax.axvline(x=opt_feed, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        y_min, y_max = self._get_smart_ylim(all_tacs)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel('Feed Stage (NF)', fontsize=12)
        ax.set_ylabel('TAC (×$1,000/year)', fontsize=12)

        fixed = iter_result.feed_sweep.fixed_values
        title = f'{case_name} - ISO Step 3: Feed Optimization\n'
        title += f'P = {fixed["pressure"]:.4f} bar, NT = {fixed["nt"]} fixed'
        if has_soft:
            title += '\n(■ = natural convergence, □ = RR-recovered)'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f"{case_name}_ISO_Feed_UCurve.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved: {filepath}")
        return filepath
    
    # ════════════════════════════════════════════════════════════════════════
    # CONVERGENCE HISTORY
    # ════════════════════════════════════════════════════════════════════════
    
    def plot_convergence(self, optimizer, case_name: str) -> str:
        """Plot TAC convergence across ISO iterations."""
        if len(optimizer.iterations) < 2:
            return None
        
        iterations = list(range(1, len(optimizer.iterations) + 1))
        tacs = [it.optimal_tac / 1000 for it in optimizer.iterations]
        nts = [it.optimal_nt for it in optimizer.iterations]
        feeds = [it.optimal_feed for it in optimizer.iterations]
        pressures = [it.optimal_pressure for it in optimizer.iterations]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        ax1.plot(iterations, tacs, 'D-', color=self.colors['optimal'], 
                linewidth=2.5, markersize=10)
        ax1.set_xlabel('ISO Iteration', fontsize=11)
        ax1.set_ylabel('TAC (×$1,000/year)', fontsize=11)
        ax1.set_title('TAC Convergence', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(iterations, nts, 's-', color=self.colors['nt'],
                linewidth=2.5, markersize=10)
        ax2.set_xlabel('ISO Iteration', fontsize=11)
        ax2.set_ylabel('Number of Stages (NT)', fontsize=11)
        ax2.set_title('NT Convergence', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.plot(iterations, feeds, '^-', color=self.colors['feed'],
                linewidth=2.5, markersize=10)
        ax3.set_xlabel('ISO Iteration', fontsize=11)
        ax3.set_ylabel('Feed Stage (NF)', fontsize=11)
        ax3.set_title('Feed Stage Convergence', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.plot(iterations, pressures, 'o-', color=self.colors['pressure'],
                linewidth=2.5, markersize=10)
        ax4.set_xlabel('ISO Iteration', fontsize=11)
        ax4.set_ylabel('Pressure (bar)', fontsize=11)
        ax4.set_title('Pressure Convergence', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(f'{case_name} - ISO Convergence History', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{case_name}_ISO_Convergence.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {filepath}")
        return filepath
    
    # ════════════════════════════════════════════════════════════════════════
    # NEW: MULTIPLE U-CURVES (TAC vs NT at different Feed stages)
    # ════════════════════════════════════════════════════════════════════════
    
    def plot_multiple_ucurves(self, nt_feed_results: List, case_name: str,
                              optimal: Dict = None,
                              n_curves: int = 5,
                              selection_method: str = 'spread') -> str:
        """
        Plot multiple U-curves at different feed stages with soft/hard feasible distinction.

        This is the KEY VISUALIZATION showing how the TAC-NT trade-off
        varies with feed stage location.

        Parameters
        ----------
        nt_feed_results : List
            Results with 'nt', 'feed', 'tac', 'feasibility' keys
        case_name : str
            Case identifier
        optimal : Dict
            Optimal point {nt, feed, tac, pressure}
        n_curves : int
            Number of U-curves to display (default: 5)
        selection_method : str
            'spread', 'around_optimal', or 'all'
        """
        if not nt_feed_results:
            logger.warning("No NT/Feed results to plot multiple U-curves")
            return None

        # Group by feed stage, separating hard and soft feasible
        hard_feed_groups = defaultdict(list)  # Hard feasible (naturally converged)
        soft_feed_groups = defaultdict(list)  # Soft feasible (RR-recovered)
        all_tacs = []

        for r in nt_feed_results:
            tac = safe_get(r, 'tac')
            nt = safe_get(r, 'nt')
            feed = safe_get(r, 'feed')
            feasibility = safe_get(r, 'feasibility', 'feasible')

            # Handle enum objects
            if hasattr(feasibility, 'value'):
                feasibility = feasibility.value

            if tac and tac < 1e10:
                if feasibility == 'soft_feasible':
                    soft_feed_groups[feed].append((nt, tac / 1000))
                else:
                    hard_feed_groups[feed].append((nt, tac / 1000))
                all_tacs.append(tac / 1000)

        # Combine all feeds for selection
        all_feeds_set = set(hard_feed_groups.keys()) | set(soft_feed_groups.keys())
        feed_groups = {}  # Combined for compatibility with rest of function
        for feed in all_feeds_set:
            feed_groups[feed] = hard_feed_groups.get(feed, []) + soft_feed_groups.get(feed, [])
        
        if not feed_groups:
            return None
        
        y_min, y_max = self._get_smart_ylim(all_tacs)
        
        # Select feeds to display
        all_feeds = sorted(feed_groups.keys())
        optimal_feed = optimal.get('feed') if optimal else None
        
        if selection_method == 'all':
            selected_feeds = all_feeds
        elif selection_method == 'around_optimal' and optimal_feed:
            opt_idx = all_feeds.index(optimal_feed) if optimal_feed in all_feeds else len(all_feeds) // 2
            half = n_curves // 2
            start = max(0, opt_idx - half)
            end = min(len(all_feeds), start + n_curves)
            start = max(0, end - n_curves)
            selected_feeds = all_feeds[start:end]
            if optimal_feed and optimal_feed in feed_groups and optimal_feed not in selected_feeds:
                selected_feeds.append(optimal_feed)
                selected_feeds.sort()
        else:  # 'spread'
            if len(all_feeds) <= n_curves:
                selected_feeds = all_feeds
            else:
                indices = np.linspace(0, len(all_feeds) - 1, n_curves, dtype=int)
                selected_feeds = [all_feeds[i] for i in indices]
                if optimal_feed and optimal_feed in feed_groups and optimal_feed not in selected_feeds:
                    selected_feeds.append(optimal_feed)
                    selected_feeds.sort()
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(selected_feeds)))
        feed_optima = {}
        
        has_soft_feasible = False  # Track if any soft feasible points exist

        for i, feed in enumerate(selected_feeds):
            is_optimal_feed = (optimal_feed and feed == optimal_feed)

            # Get hard and soft feasible data for this feed
            hard_data = sorted(hard_feed_groups.get(feed, []))
            soft_data = sorted(soft_feed_groups.get(feed, []))

            # ════════════════════════════════════════════════════════════════
            # HARD FEASIBLE LINE + SOFT FEASIBLE MARKERS
            # Line drawn through hard feasible only (reliable TAC values)
            # Soft feasible shown as hollow markers (diagnostic reference)
            # ════════════════════════════════════════════════════════════════

            if soft_data:
                has_soft_feasible = True

            # Filter valid TAC values
            hard_data_filtered = [(d[0], d[1]) for d in hard_data if d[1] <= y_max * 1.2]
            soft_data_filtered = [(d[0], d[1]) for d in soft_data if d[1] <= y_max * 1.2]

            if len(hard_data_filtered) < 2 and not soft_data_filtered:
                continue

            # Style based on whether this is optimal feed
            if is_optimal_feed:
                linewidth, alpha, zorder = 3.5, 1.0, 8
                label = f'NF = {feed} (OPTIMAL)'
            else:
                linewidth, alpha, zorder = 2.0, 0.8, 5
                label = f'NF = {feed}'

            # Plot continuous line through HARD FEASIBLE only
            hard_nts_line = [d[0] for d in hard_data_filtered]
            hard_tacs_line = [d[1] for d in hard_data_filtered]
            if len(hard_nts_line) >= 2:
                ax.plot(hard_nts_line, hard_tacs_line, '-', color=colors[i],
                       linewidth=linewidth, alpha=alpha, zorder=zorder, label=label)
            elif hard_nts_line:
                ax.plot(hard_nts_line, hard_tacs_line, 'o', color=colors[i],
                       markersize=8, alpha=alpha, zorder=zorder, label=label)

            # Hard feasible markers (filled circles)
            if hard_nts_line:
                markersize = 10 if is_optimal_feed else 6
                ax.scatter(hard_nts_line, hard_tacs_line, color=colors[i], s=markersize**2,
                          marker='o', zorder=zorder+1, edgecolors='white', linewidths=0.5)

            # Soft feasible markers (hollow circles — not connected by line)
            soft_nts_m = [d[0] for d in soft_data_filtered]
            soft_tacs_m = [d[1] for d in soft_data_filtered]
            if soft_nts_m:
                markersize = 8 if is_optimal_feed else 5
                ax.scatter(soft_nts_m, soft_tacs_m, color=colors[i], s=markersize**2,
                          marker='o', facecolors='white', edgecolors=colors[i],
                          linewidths=1.5, zorder=zorder+1)

            # Find and mark the minimum (from hard feasible only)
            if hard_nts_line:
                hard_min_idx = np.argmin(hard_tacs_line)
                feed_optima[feed] = (hard_nts_line[hard_min_idx], hard_tacs_line[hard_min_idx])

                if not is_optimal_feed:
                    ax.plot(hard_nts_line[hard_min_idx], hard_tacs_line[hard_min_idx], 'o',
                           color=colors[i], markersize=12,
                           markeredgecolor='white', markeredgewidth=2,
                           alpha=0.9, zorder=zorder+2)
        
        # Add legend entry for soft feasible markers (if any exist)
        if has_soft_feasible:
            ax.scatter([], [], color='gray', s=40, marker='o', facecolors='white',
                      edgecolors='gray', linewidths=1.5, label='○ = Soft Feasible (RR-recovered)')
            ax.scatter([], [], color='gray', s=40, marker='o',
                      label='● = Hard Feasible (natural convergence)')

        if optimal:
            ax.plot(optimal['nt'], optimal['tac']/1000, '*', color='red',
                   markersize=25, markeredgecolor='black', markeredgewidth=2,
                   zorder=10, label=f"Global Optimal: NT={optimal['nt']}, NF={optimal['feed']}")
            
            ax.annotate(f"Optimal\nTAC = ${optimal['tac']/1000:,.0f}k/yr\nNT = {optimal['nt']}, NF = {optimal['feed']}",
                       xy=(optimal['nt'], optimal['tac']/1000),
                       xytext=(optimal['nt'] + 8, optimal['tac']/1000 * 1.1),
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                edgecolor='red', alpha=0.95),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Draw trajectory of optima
        if len(feed_optima) >= 3:
            sorted_optima = sorted(feed_optima.items())
            opt_nts = [o[1][0] for o in sorted_optima]
            opt_tacs = [o[1][1] for o in sorted_optima]
            ax.plot(opt_nts, opt_tacs, ':', color='gray', linewidth=2, 
                   alpha=0.5, zorder=3, label='Optima Trajectory')
        
        # Annotations
        ax.annotate('← Energy-dominated region\n   (few stages, high reflux)',
                   xy=(min([min(d[0] for d in feed_groups[f]) for f in selected_feeds]) + 3, y_max * 0.92),
                   fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.85))
        
        ax.annotate('Capital-dominated region →\n(many stages, low reflux)',
                   xy=(max([max(d[0] for d in feed_groups[f]) for f in selected_feeds]) - 3, y_max * 0.92),
                   fontsize=10, style='italic', ha='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85))
        
        ax.set_xlabel('Number of Stages (NT)', fontsize=13, fontweight='bold')
        ax.set_ylabel('TAC (×$1,000/year)', fontsize=13, fontweight='bold')

        # Update title to include note about line styles if soft feasible exists
        title = f'{case_name} - Multiple U-Curves: Effect of Feed Stage Location\n'
        if has_soft_feasible:
            title += '(Solid = naturally converged, Dashed = RR-recovered)'
        else:
            title += '(Each curve shows TAC vs NT at a different feed stage)'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                 fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{case_name}_Multiple_UCurves.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {filepath}")
        return filepath
    
    def plot_ucurves_family(self, nt_feed_results: List, case_name: str,
                           optimal: Dict = None) -> str:
        """Plot ALL U-curves as a family with colormap."""
        if not nt_feed_results:
            return None
        
        feed_groups = defaultdict(list)
        all_tacs = []
        
        for r in nt_feed_results:
            tac = safe_get(r, 'tac')
            nt = safe_get(r, 'nt')
            feed = safe_get(r, 'feed')
            if tac and tac < 1e10:
                feed_groups[feed].append((nt, tac / 1000))
                all_tacs.append(tac / 1000)
        
        if not feed_groups:
            return None
        
        y_min, y_max = self._get_smart_ylim(all_tacs)
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        all_feeds = sorted(feed_groups.keys())
        optimal_feed = optimal.get('feed') if optimal else None
        
        norm = plt.Normalize(min(all_feeds), max(all_feeds))
        cmap = plt.cm.coolwarm
        
        for feed in all_feeds:
            data = sorted(feed_groups[feed])
            nts = [d[0] for d in data]
            tacs = [d[1] for d in data]
            
            valid_idx = [j for j, t in enumerate(tacs) if t <= y_max * 1.2]
            if len(valid_idx) < 3:
                continue
            
            nts_plot = [nts[j] for j in valid_idx]
            tacs_plot = [tacs[j] for j in valid_idx]
            
            color = cmap(norm(feed))
            
            if optimal_feed and feed == optimal_feed:
                ax.plot(nts_plot, tacs_plot, '-', color='black', 
                       linewidth=4, alpha=1.0, zorder=9)
                ax.plot(nts_plot, tacs_plot, '-', color=color, 
                       linewidth=2.5, alpha=1.0, zorder=10)
            else:
                ax.plot(nts_plot, tacs_plot, '-', color=color, 
                       linewidth=1.2, alpha=0.6, zorder=5)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Feed Stage (NF)', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)
        
        if optimal:
            ax.plot(optimal['nt'], optimal['tac']/1000, '*', color='gold',
                   markersize=25, markeredgecolor='black', markeredgewidth=2,
                   zorder=15)
            ax.annotate(f"Global Optimal\nNT={optimal['nt']}, NF={optimal['feed']}\nTAC=${optimal['tac']/1000:,.0f}k/yr",
                       xy=(optimal['nt'], optimal['tac']/1000),
                       xytext=(optimal['nt'] + 10, optimal['tac']/1000 * 1.15),
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                edgecolor='gold', linewidth=2, alpha=0.95),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        ax.set_xlabel('Number of Stages (NT)', fontsize=13, fontweight='bold')
        ax.set_ylabel('TAC (×$1,000/year)', fontsize=13, fontweight='bold')
        ax.set_title(f'{case_name} - U-Curve Family: All Feed Stage Locations\n'
                    f'(Color indicates feed stage position)',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{case_name}_UCurves_Family.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {filepath}")
        return filepath
    
    def plot_ucurves_3d(self, nt_feed_results: List, case_name: str,
                        optimal: Dict = None) -> str:
        """Plot 3D optimization landscape."""
        if not nt_feed_results:
            return None
        
        nts, feeds, tacs = [], [], []
        
        for r in nt_feed_results:
            tac = safe_get(r, 'tac')
            nt = safe_get(r, 'nt')
            feed = safe_get(r, 'feed')
            if tac and tac < 1e10:
                nts.append(nt)
                feeds.append(feed)
                tacs.append(tac / 1000)
        
        if not nts:
            return None
        
        z_max = np.percentile(np.array(tacs), 95)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(nts, feeds, tacs, c=tacs, cmap='viridis',
                            s=30, alpha=0.7, vmax=z_max)
        
        plt.colorbar(scatter, ax=ax, label='TAC (×$1,000/year)', shrink=0.6)
        
        if optimal:
            ax.scatter([optimal['nt']], [optimal['feed']], [optimal['tac']/1000],
                      c='red', marker='*', s=500, edgecolors='black', linewidth=2,
                      zorder=10, label=f"Optimal: NT={optimal['nt']}, NF={optimal['feed']}")
        
        ax.set_xlabel('Number of Stages (NT)', fontsize=11)
        ax.set_ylabel('Feed Stage (NF)', fontsize=11)
        ax.set_zlabel('TAC (×$1,000/year)', fontsize=11)
        ax.set_title(f'{case_name} - 3D Optimization Landscape',
                    fontsize=14, fontweight='bold')
        ax.set_zlim(0, z_max)
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{case_name}_UCurves_3D.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {filepath}")
        return filepath
    
    def plot_contour(self, nt_feed_results: List, case_name: str,
                     optimal: Dict = None) -> str:
        """Plot TAC contour map."""
        if not nt_feed_results:
            return None
        
        nts, feeds, tacs = [], [], []
        
        for r in nt_feed_results:
            tac = safe_get(r, 'tac')
            nt = safe_get(r, 'nt')
            feed = safe_get(r, 'feed')
            if tac and tac < 1e10:
                nts.append(nt)
                feeds.append(feed)
                tacs.append(tac / 1000)
        
        if not nts:
            return None
        
        tac_array = np.array(tacs)
        vmin = np.percentile(tac_array, 5)
        vmax = np.percentile(tac_array, 95)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(nts, feeds, c=tacs, cmap='viridis',
                            s=60, alpha=0.8, vmin=vmin, vmax=vmax,
                            edgecolors='white', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax, label='TAC (×$1,000/year)')
        
        if optimal:
            ax.plot(optimal['nt'], optimal['feed'], '*', color='red',
                   markersize=20, markeredgecolor='black', markeredgewidth=2,
                   zorder=10)
            ax.annotate(f"Optimal: NT={optimal['nt']}, NF={optimal['feed']}",
                       xy=(optimal['nt'], optimal['feed']),
                       xytext=(optimal['nt'] + 5, optimal['feed'] + 5),
                       fontsize=11, color='red', fontweight='bold')
        
        ax.set_xlabel('Number of Stages (NT)', fontsize=12)
        ax.set_ylabel('Feed Stage (NF)', fontsize=12)
        ax.set_title(f'{case_name} - TAC Contour Map',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{case_name}_Contour.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {filepath}")
        return filepath
    
    # ════════════════════════════════════════════════════════════════════════
    # SUMMARY DASHBOARD
    # ════════════════════════════════════════════════════════════════════════

    def _collect_infeasible(self, points, x_key):
        """Collect infeasible points from a sweep for plotting markers."""
        temp_xs = []
        conv_xs = []
        for pt in points:
            x_val = safe_get(pt, x_key)
            if x_val is None:
                continue
            feasibility = safe_get(pt, 'feasibility', 'feasible')
            if hasattr(feasibility, 'value'):
                feasibility = feasibility.value
            if 'temperature' in str(feasibility):
                temp_xs.append(x_val)
            elif 'convergence' in str(feasibility):
                conv_xs.append(x_val)
        return temp_xs, conv_xs

    def plot_summary(self, optimizer, case_name: str) -> str:
        """Create ISO summary dashboard."""
        result = optimizer.result
        if not result:
            return None
        
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.2, 0.8])
        
        fig.suptitle(f'{case_name} - ISO Optimization Summary', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Results box
        ax_results = fig.add_subplot(gs[0, 0])
        ax_results.axis('off')
        results_text = f"""
╔════════════════════════════════╗
║      OPTIMAL CONFIGURATION      ║
╠════════════════════════════════╣
║  NT:  {result.optimal_nt:>4} stages              ║
║  NF:  {result.optimal_feed:>4}                     ║
║  P:   {result.optimal_pressure:>6.4f} bar            ║
║  TAC: ${result.optimal_tac:>12,.0f}/yr       ║
╠════════════════════════════════╣
║  Converged: {"YES ✓" if result.converged else "NO  ":>6}              ║
║  Iterations: {result.convergence_iteration:>3}                ║
╚════════════════════════════════╝
"""
        ax_results.text(0.1, 0.9, results_text, transform=ax_results.transAxes,
                       fontsize=11, fontfamily='monospace', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # Method box
        ax_method = fig.add_subplot(gs[0, 1])
        ax_method.axis('off')
        has_styrene = getattr(optimizer, 'has_styrene', True)
        if has_styrene:
            constraint_line = f"(T_reb <= {T_REBOILER_MAX:.0f} C)"
        else:
            constraint_line = "(no T_reb constraint)"
        method_text = f"""
╔══════════════════════════════════╗
║     ISO METHODOLOGY               ║
╠══════════════════════════════════╣
║  OUTER LOOP:                      ║
║  ├─ Step 1: Optimize P            ║
║  │   {constraint_line:<30}║
║  ├─ Step 2: Optimize NT           ║
║  └─ Step 3: Optimize NF           ║
║  CHECK: Design unchanged?         ║
║  → If no, repeat                  ║
╚══════════════════════════════════╝
"""
        ax_method.text(0.1, 0.9, method_text, transform=ax_method.transAxes,
                      fontsize=10, fontfamily='monospace', verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
        
        # Stats box
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.axis('off')
        stats_text = f"""
╔════════════════════════════════╗
║         STATISTICS              ║
╠════════════════════════════════╣
║  Total evaluations: {result.total_evaluations:>5}       ║
║  Feasible:          {result.feasible_evaluations:>5}       ║
║  Infeasible:        {result.infeasible_evaluations:>5}       ║
║  Time: {result.total_time_seconds:>6.1f} seconds        ║
╚════════════════════════════════╝
"""
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, fontfamily='monospace', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9))
        
        # U-curves from final iteration
        if optimizer.iterations:
            final = optimizer.iterations[-1]
            
            ax_p = fig.add_subplot(gs[1, 0])
            points = final.pressure_sweep.points
            p_data = [(safe_get(pt, 'pressure'), safe_get(pt, 'tac', 1e12) / 1000)
                     for pt in points if safe_get(pt, 'tac', 1e12) < 1e10]
            if p_data:
                p_data.sort()
                ps, tacs = zip(*p_data)
                ax_p.plot(ps, tacs, 'o-', color=self.colors['pressure'], linewidth=2)
                ax_p.axvline(x=result.optimal_pressure, color='red', linestyle='--')
                # Infeasible markers
                temp_xs, conv_xs = self._collect_infeasible(points, 'pressure')
                y_top = max(tacs) * 1.05
                if temp_xs:
                    ax_p.scatter(temp_xs, [y_top]*len(temp_xs), marker='x',
                                color=self.colors['infeasible_temp'], s=60, zorder=8,
                                label='T_reb>{:.0f}°C'.format(T_REBOILER_MAX))
                if conv_xs:
                    ax_p.scatter(conv_xs, [y_top]*len(conv_xs), marker='x',
                                color=self.colors['infeasible_conv'], s=40, zorder=7,
                                label='No conv.')
                if temp_xs or conv_xs:
                    ax_p.legend(fontsize=7, loc='upper right')
            ax_p.set_xlabel('Pressure (bar)')
            ax_p.set_ylabel('TAC (×$1k/yr)')
            ax_p.set_title('Step 1: P Sweep', fontweight='bold')
            ax_p.grid(True, alpha=0.3)
            
            ax_nt = fig.add_subplot(gs[1, 1])
            points = final.nt_sweep.points
            nt_data = [(safe_get(pt, 'nt'), safe_get(pt, 'tac', 1e12) / 1000)
                      for pt in points if safe_get(pt, 'tac', 1e12) < 1e10]
            if nt_data:
                nt_data.sort()
                nts, tacs = zip(*nt_data)
                ax_nt.plot(nts, tacs, 's-', color=self.colors['nt'], linewidth=2)
                ax_nt.axvline(x=result.optimal_nt, color='red', linestyle='--')
                # Infeasible markers
                temp_xs, conv_xs = self._collect_infeasible(points, 'nt')
                y_top = max(tacs) * 1.05
                if temp_xs:
                    ax_nt.scatter(temp_xs, [y_top]*len(temp_xs), marker='x',
                                 color=self.colors['infeasible_temp'], s=60, zorder=8)
                if conv_xs:
                    ax_nt.scatter(conv_xs, [y_top]*len(conv_xs), marker='x',
                                 color=self.colors['infeasible_conv'], s=40, zorder=7,
                                 label='No conv.')
                if conv_xs or temp_xs:
                    ax_nt.legend(fontsize=7, loc='upper right')
            ax_nt.set_xlabel('Number of Stages (NT)')
            ax_nt.set_ylabel('TAC (×$1k/yr)')
            ax_nt.set_title('Step 2: NT Sweep', fontweight='bold')
            ax_nt.grid(True, alpha=0.3)
            
            ax_feed = fig.add_subplot(gs[1, 2])
            points = final.feed_sweep.points
            feed_data = [(safe_get(pt, 'feed'), safe_get(pt, 'tac', 1e12) / 1000)
                        for pt in points if safe_get(pt, 'tac', 1e12) < 1e10]
            if feed_data:
                feed_data.sort()
                feeds, tacs = zip(*feed_data)
                ax_feed.plot(feeds, tacs, '^-', color=self.colors['feed'], linewidth=2)
                ax_feed.axvline(x=result.optimal_feed, color='red', linestyle='--')
                # Infeasible markers
                temp_xs, conv_xs = self._collect_infeasible(points, 'feed')
                y_top = max(tacs) * 1.05
                if temp_xs:
                    ax_feed.scatter(temp_xs, [y_top]*len(temp_xs), marker='x',
                                   color=self.colors['infeasible_temp'], s=60, zorder=8)
                if conv_xs:
                    ax_feed.scatter(conv_xs, [y_top]*len(conv_xs), marker='x',
                                   color=self.colors['infeasible_conv'], s=40, zorder=7,
                                   label='No conv.')
                if conv_xs or temp_xs:
                    ax_feed.legend(fontsize=7, loc='upper right')
            ax_feed.set_xlabel('Feed Stage (NF)')
            ax_feed.set_ylabel('TAC (×$1k/yr)')
            ax_feed.set_title('Step 3: NF Sweep', fontweight='bold')
            ax_feed.grid(True, alpha=0.3)
        
        # Convergence
        ax_conv = fig.add_subplot(gs[2, :])
        if len(optimizer.iterations) > 1:
            iterations = list(range(1, len(optimizer.iterations) + 1))
            tacs = [it.optimal_tac / 1000 for it in optimizer.iterations]
            ax_conv.plot(iterations, tacs, 'D-', color=self.colors['optimal'],
                        linewidth=2.5, markersize=10)
            ax_conv.axhline(y=tacs[-1], color='red', linestyle='--', alpha=0.5)
        ax_conv.set_xlabel('ISO Iteration', fontsize=12)
        ax_conv.set_ylabel('TAC (×$1,000/year)', fontsize=12)
        ax_conv.set_title('TAC Convergence', fontsize=12, fontweight='bold')
        ax_conv.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filepath = os.path.join(self.output_dir, f"{case_name}_ISO_Summary.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {filepath}")
        return filepath
    
    # ════════════════════════════════════════════════════════════════════════
    # PLOT ALL
    # ════════════════════════════════════════════════════════════════════════
    
    def plot_all(self, optimizer, case_name: str, 
                 nt_feed_results: List = None,
                 n_ucurves: int = 5) -> List[str]:
        """
        Generate all plots.
        
        Parameters
        ----------
        optimizer : ISOOptimizer
            Optimizer with results
        case_name : str
            Case identifier
        nt_feed_results : List, optional
            NT/Feed sweep data for multiple U-curves
        n_ucurves : int
            Number of U-curves to show
        """
        plot_files = []
        
        result = optimizer.result
        optimal = None
        if result:
            optimal = {
                'nt': result.optimal_nt,
                'feed': result.optimal_feed,
                'pressure': result.optimal_pressure,
                'tac': result.optimal_tac,
            }
        
        # Standard ISO plots
        f = self.plot_pressure_sweep(optimizer, case_name)
        if f: plot_files.append(f)
        
        f = self.plot_nt_ucurve(optimizer, case_name)
        if f: plot_files.append(f)
        
        f = self.plot_feed_ucurve(optimizer, case_name)
        if f: plot_files.append(f)
        
        if len(optimizer.iterations) > 1:
            f = self.plot_convergence(optimizer, case_name)
            if f: plot_files.append(f)
        
        # NEW: Multiple U-curves (if data provided)
        if nt_feed_results:
            f = self.plot_multiple_ucurves(nt_feed_results, case_name, optimal, n_ucurves)
            if f: plot_files.append(f)
            
            f = self.plot_ucurves_family(nt_feed_results, case_name, optimal)
            if f: plot_files.append(f)
            
            f = self.plot_ucurves_3d(nt_feed_results, case_name, optimal)
            if f: plot_files.append(f)
            
            f = self.plot_contour(nt_feed_results, case_name, optimal)
            if f: plot_files.append(f)
        
        f = self.plot_summary(optimizer, case_name)
        if f: plot_files.append(f)
        
        logger.info(f"Generated {len(plot_files)} plots")
        return plot_files


# ════════════════════════════════════════════════════════════════════════════
# STANDALONE FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def plot_multiple_ucurves_standalone(data: List[Dict], case_name: str,
                                     output_dir: str = "results",
                                     n_curves: int = 5,
                                     optimal: Dict = None) -> str:
    """
    Standalone function to plot multiple U-curves.
    
    Example
    -------
    >>> data = [{'nt': 30, 'feed': 15, 'tac': 500000}, ...]
    >>> optimal = {'nt': 35, 'feed': 18, 'tac': 480000}
    >>> plot_multiple_ucurves_standalone(data, "Case1_COL2", optimal=optimal)
    """
    viz = ISOVisualizer(output_dir)
    return viz.plot_multiple_ucurves(data, case_name, optimal, n_curves)