"""
visualization_metaheuristic.py
==============================
Visualization module for PSO and GA optimization results.

Generates convergence plots showing objective function (TAC) vs iteration/generation.

Author: PSE Lab, NTUST
Version: 1.0
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')


class MetaheuristicVisualizer:
    """
    Visualizer for PSO and GA optimization results.

    Generates convergence plots and comparison charts.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_convergence(self,
                         convergence_history: List[float],
                         algorithm: str = "PSO",
                         case_name: str = "Case",
                         optimal_tac: float = None,
                         n_iterations: int = None,
                         save: bool = True) -> str:
        """
        Plot convergence curve for a single algorithm run.

        Parameters
        ----------
        convergence_history : List[float]
            Best TAC at each iteration/generation
        algorithm : str
            Algorithm name (PSO or GA)
        case_name : str
            Case identifier
        optimal_tac : float, optional
            Final optimal TAC value
        n_iterations : int, optional
            Total number of iterations
        save : bool
            Whether to save the plot

        Returns
        -------
        str : Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter out None values
        valid_history = [(i+1, tac) for i, tac in enumerate(convergence_history)
                        if tac is not None and tac < 1e10]

        if not valid_history:
            logger.warning("No valid convergence data to plot")
            plt.close()
            return None

        iterations, tacs = zip(*valid_history)

        # Plot convergence curve
        color = '#1f77b4' if algorithm == "PSO" else '#2ca02c'
        ax.plot(iterations, tacs, 'o-', color=color, linewidth=2, markersize=4,
                label=f'{algorithm} Best TAC')

        # Fill area under curve
        ax.fill_between(iterations, tacs, alpha=0.2, color=color)

        # Mark final optimum
        if optimal_tac and optimal_tac < 1e10:
            ax.axhline(y=optimal_tac, color='red', linestyle='--', linewidth=1.5,
                      label=f'Final Optimum: ${optimal_tac:,.0f}/yr')
            ax.scatter([iterations[-1]], [optimal_tac], color='red', s=100,
                      zorder=5, marker='*')

        # Labels and title
        iter_label = "Generation" if algorithm == "GA" else "Iteration"
        ax.set_xlabel(iter_label, fontsize=12)
        ax.set_ylabel('Total Annual Cost ($/year)', fontsize=12)
        ax.set_title(f'{algorithm} Optimization Convergence - {case_name}',
                    fontsize=14, fontweight='bold')

        # Format y-axis with comma separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir,
                                   f"{case_name}_{algorithm}_Convergence_{timestamp}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved convergence plot: {filename}")
            plt.close()
            return filename

        return None

    def plot_comparison(self,
                       results: Dict[str, Dict],
                       case_name: str = "Case",
                       save: bool = True) -> str:
        """
        Plot comparison of multiple algorithms.

        Parameters
        ----------
        results : Dict[str, Dict]
            Dictionary with algorithm names as keys and result dicts as values.
            Each result dict should have 'convergence_history' and 'optimal_tac'.
        case_name : str
            Case identifier
        save : bool
            Whether to save the plot

        Returns
        -------
        str : Path to saved plot file
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = {'ISO': '#ff7f0e', 'PSO': '#1f77b4', 'GA': '#2ca02c'}
        markers = {'ISO': 's', 'PSO': 'o', 'GA': '^'}

        # Left plot: Convergence curves
        ax1 = axes[0]

        for algo, data in results.items():
            history = data.get('convergence_history', [])
            if not history:
                continue

            # Filter valid values
            valid = [(i+1, tac) for i, tac in enumerate(history)
                    if tac is not None and tac < 1e10]

            if valid:
                iters, tacs = zip(*valid)
                color = colors.get(algo, 'gray')
                marker = markers.get(algo, 'o')
                ax1.plot(iters, tacs, f'{marker}-', color=color, linewidth=2,
                        markersize=4, label=algo, alpha=0.8)

        ax1.set_xlabel('Iteration / Generation', fontsize=12)
        ax1.set_ylabel('Best TAC ($/year)', fontsize=12)
        ax1.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Right plot: Final TAC comparison (bar chart)
        ax2 = axes[1]

        algos = []
        tacs = []
        bar_colors = []

        for algo, data in results.items():
            optimal_tac = data.get('optimal_tac')
            if optimal_tac and optimal_tac < 1e10:
                algos.append(algo)
                tacs.append(optimal_tac)
                bar_colors.append(colors.get(algo, 'gray'))

        if algos:
            bars = ax2.bar(algos, tacs, color=bar_colors, alpha=0.8, edgecolor='black')

            # Add value labels on bars
            for bar, tac in zip(bars, tacs):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'${tac:,.0f}', ha='center', va='bottom', fontsize=10)

            # Mark best
            best_idx = np.argmin(tacs)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)

        ax2.set_xlabel('Algorithm', fontsize=12)
        ax2.set_ylabel('Optimal TAC ($/year)', fontsize=12)
        ax2.set_title('Final TAC Comparison', fontsize=14, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir,
                                   f"{case_name}_Algorithm_Comparison_{timestamp}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison plot: {filename}")
            plt.close()
            return filename

        return None

    def plot_solution_summary(self,
                             result: Dict,
                             algorithm: str = "PSO",
                             case_name: str = "Case",
                             save: bool = True) -> str:
        """
        Plot solution summary with optimal configuration.

        Parameters
        ----------
        result : Dict
            Optimization result with optimal values
        algorithm : str
            Algorithm name
        case_name : str
            Case identifier
        save : bool
            Whether to save the plot

        Returns
        -------
        str : Path to saved plot file
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        optimal = result.get('optimal', {})
        stats = result.get('statistics', {})
        convergence = result.get('convergence', {}).get('history', [])

        # Plot 1: Convergence curve
        ax1 = axes[0]
        if convergence:
            valid = [(i+1, tac) for i, tac in enumerate(convergence)
                    if tac is not None and tac < 1e10]
            if valid:
                iters, tacs = zip(*valid)
                color = '#1f77b4' if algorithm == "PSO" else '#2ca02c'
                ax1.plot(iters, tacs, 'o-', color=color, linewidth=2, markersize=3)
                ax1.fill_between(iters, tacs, alpha=0.2, color=color)

        ax1.set_xlabel('Iteration', fontsize=10)
        ax1.set_ylabel('Best TAC ($/yr)', fontsize=10)
        ax1.set_title('Convergence', fontsize=12, fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
        ax1.grid(True, alpha=0.3)

        # Plot 2: Optimal configuration
        ax2 = axes[1]
        ax2.axis('off')

        config_text = f"""
OPTIMAL CONFIGURATION
{'='*30}

Number of Stages (NT): {optimal.get('nt', 'N/A')}
Feed Stage (NF): {optimal.get('feed', 'N/A')}
Pressure: {optimal.get('pressure', 0):.4f} bar

TAC: ${optimal.get('tac', 0):,.0f}/year
T_reb: {optimal.get('T_reb', 0):.1f}C

{'='*30}
Algorithm: {algorithm}
Case: {case_name}
"""
        ax2.text(0.1, 0.9, config_text, transform=ax2.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # Plot 3: Statistics
        ax3 = axes[2]
        ax3.axis('off')

        stats_text = f"""
STATISTICS
{'='*30}

Total Evaluations: {stats.get('total_evaluations', 'N/A')}
Feasible: {stats.get('feasible', 'N/A')}
Infeasible: {stats.get('infeasible', 'N/A')}

Time: {stats.get('time_seconds', 0):.1f} seconds

{'='*30}
"""
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

        plt.suptitle(f'{algorithm} Optimization Results - {case_name}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir,
                                   f"{case_name}_{algorithm}_Summary_{timestamp}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved summary plot: {filename}")
            plt.close()
            return filename

        return None


def generate_plots_from_result_file(result_file: str, output_dir: str = "results") -> List[str]:
    """
    Generate plots from a saved result JSON file.

    Parameters
    ----------
    result_file : str
        Path to result JSON file
    output_dir : str
        Output directory for plots

    Returns
    -------
    List[str] : Paths to generated plot files
    """
    with open(result_file, 'r') as f:
        data = json.load(f)

    algorithm = data.get('algorithm', 'Unknown')
    case_name = data.get('case_name', 'Case')

    visualizer = MetaheuristicVisualizer(output_dir)
    plot_files = []

    # Generate convergence plot
    convergence = data.get('convergence', {}).get('history', [])
    if convergence:
        optimal_tac = data.get('optimal', {}).get('tac')
        plot_file = visualizer.plot_convergence(
            convergence, algorithm, case_name, optimal_tac
        )
        if plot_file:
            plot_files.append(plot_file)

    # Generate summary plot
    summary_file = visualizer.plot_solution_summary(data, algorithm, case_name)
    if summary_file:
        plot_files.append(summary_file)

    return plot_files


if __name__ == "__main__":
    # Test with sample data
    import sys

    if len(sys.argv) > 1:
        result_file = sys.argv[1]
        plots = generate_plots_from_result_file(result_file)
        print(f"Generated {len(plots)} plots:")
        for p in plots:
            print(f"  - {p}")
    else:
        # Generate test convergence
        print("Testing MetaheuristicVisualizer...")

        visualizer = MetaheuristicVisualizer()

        # Fake PSO convergence data
        pso_history = [500000 - i*5000 + np.random.randint(-2000, 2000)
                      for i in range(50)]

        plot = visualizer.plot_convergence(
            pso_history, "PSO", "Test_Case",
            optimal_tac=min(pso_history)
        )
        print(f"Generated: {plot}")
