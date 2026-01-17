"""
comparison_plots.py
===================
Comparison Visualization Tools

Publication-quality plots for algorithm comparison:
- Convergence curves
- Boxplots
- Efficiency scatter plots
- Statistical comparison heatmaps

Usage:
    from comparison_plots import ComparisonVisualizer

    viz = ComparisonVisualizer()
    viz.plot_convergence_comparison(convergence_data, "Case1_COL2")
    viz.plot_boxplot(results_dict, "Case1_COL2")

Author: PSE Lab, NTUST
Version: 1.0
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')


# ════════════════════════════════════════════════════════════════════════════
# COMPARISON VISUALIZER
# ════════════════════════════════════════════════════════════════════════════

class ComparisonVisualizer:
    """
    Publication-quality visualization for algorithm comparison.

    Parameters
    ----------
    output_dir : str
        Directory for output files
    dpi : int
        Resolution for saved figures
    figsize : tuple
        Default figure size
    """

    # Color palette for algorithms
    COLORS = {
        'ISO': '#1f77b4',  # Blue
        'PSO': '#ff7f0e',  # Orange
        'GA': '#2ca02c',   # Green
        'DE': '#d62728',   # Red
        'SA': '#9467bd',   # Purple
    }

    MARKERS = {
        'ISO': 's',  # Square
        'PSO': 'o',  # Circle
        'GA': '^',   # Triangle
        'DE': 'D',   # Diamond
        'SA': 'v',   # Inverted triangle
    }

    def __init__(self, output_dir: str = "results", dpi: int = 300, figsize: tuple = (10, 7)):
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize
        os.makedirs(output_dir, exist_ok=True)

    def _get_color(self, algorithm: str) -> str:
        """Get color for algorithm."""
        return self.COLORS.get(algorithm.upper(), '#7f7f7f')

    def _get_marker(self, algorithm: str) -> str:
        """Get marker for algorithm."""
        return self.MARKERS.get(algorithm.upper(), 'o')

    def plot_convergence_comparison(
        self,
        convergence_data: Dict[str, List[float]],
        case_name: str,
        title: Optional[str] = None,
        xlabel: str = "Iteration/Generation",
        ylabel: str = "Best TAC (×$1,000/year)",
    ) -> str:
        """
        Plot convergence curves for multiple algorithms.

        Parameters
        ----------
        convergence_data : Dict[str, List[float]]
            Dictionary mapping algorithm names to convergence history
            e.g., {"ISO": [150000, 140000, ...], "PSO": [...]}
        case_name : str
            Case name for title and filename
        title : str, optional
            Custom title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label

        Returns
        -------
        str
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for alg, history in convergence_data.items():
            if not history:
                continue

            # Convert to thousands
            history_k = [t / 1000 if t and t < 1e10 else np.nan for t in history]
            iterations = range(1, len(history_k) + 1)

            ax.plot(
                iterations,
                history_k,
                label=alg,
                color=self._get_color(alg),
                linewidth=2,
                marker=self._get_marker(alg),
                markersize=4,
                markevery=max(1, len(history_k) // 10),
            )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(
                f'{case_name} - Algorithm Convergence Comparison',
                fontsize=14,
                fontweight='bold'
            )

        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Save
        filepath = os.path.join(self.output_dir, f"{case_name}_Convergence_Comparison.png")
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_boxplot(
        self,
        results: Dict[str, List[float]],
        case_name: str,
        title: Optional[str] = None,
        ylabel: str = "TAC (×$1,000/year)",
    ) -> str:
        """
        Generate boxplot comparing algorithm TAC distributions.

        Parameters
        ----------
        results : Dict[str, List[float]]
            Dictionary mapping algorithm names to TAC values
        case_name : str
            Case name for title and filename
        title : str, optional
            Custom title
        ylabel : str
            Y-axis label

        Returns
        -------
        str
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        algorithms = list(results.keys())
        data = [np.array([v for v in results[alg] if v < 1e10]) / 1000 for alg in algorithms]

        # Filter out empty arrays
        valid_idx = [i for i, d in enumerate(data) if len(d) > 0]
        algorithms = [algorithms[i] for i in valid_idx]
        data = [data[i] for i in valid_idx]

        if not data:
            plt.close()
            return ""

        bp = ax.boxplot(data, labels=algorithms, patch_artist=True)

        # Color boxes
        for i, (patch, alg) in enumerate(zip(bp['boxes'], algorithms)):
            patch.set_facecolor(self._get_color(alg))
            patch.set_alpha(0.6)

        # Style whiskers, caps, medians
        for whisker in bp['whiskers']:
            whisker.set(color='gray', linewidth=1.5)
        for cap in bp['caps']:
            cap.set(color='gray', linewidth=1.5)
        for median in bp['medians']:
            median.set(color='black', linewidth=2)

        ax.set_ylabel(ylabel, fontsize=12)

        # Calculate n for each algorithm
        n_runs = [len([v for v in results[alg] if v < 1e10]) for alg in algorithms]
        subtitle = ', '.join([f"{alg}: n={n}" for alg, n in zip(algorithms, n_runs)])

        if title:
            ax.set_title(f"{title}\n({subtitle})", fontsize=14, fontweight='bold')
        else:
            ax.set_title(
                f'{case_name} - Algorithm Performance Comparison\n({subtitle})',
                fontsize=14,
                fontweight='bold'
            )

        ax.grid(True, alpha=0.3, axis='y')

        # Save
        filepath = os.path.join(self.output_dir, f"{case_name}_Boxplot_Comparison.png")
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_efficiency_scatter(
        self,
        results: List[Dict],
        case_name: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Scatter plot: Number of evaluations vs Final TAC.

        Shows trade-off between computational cost and solution quality.

        Parameters
        ----------
        results : List[Dict]
            List of result dictionaries with keys:
            - algorithm: str
            - evaluations: int
            - tac: float
        case_name : str
            Case name for title and filename

        Returns
        -------
        str
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Group by algorithm for legend
        algorithms_plotted = set()

        for r in results:
            alg = r.get('algorithm', 'Unknown')
            evals = r.get('evaluations', 0)
            tac = r.get('tac', 0)

            if tac > 1e10:
                continue

            label = alg if alg not in algorithms_plotted else None
            algorithms_plotted.add(alg)

            ax.scatter(
                evals,
                tac / 1000,
                marker=self._get_marker(alg),
                color=self._get_color(alg),
                s=100,
                alpha=0.7,
                label=label,
            )

        ax.set_xlabel('Number of Evaluations', fontsize=12)
        ax.set_ylabel('Final TAC (×$1,000/year)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(
                f'{case_name} - Efficiency vs Quality Trade-off',
                fontsize=14,
                fontweight='bold'
            )

        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Save
        filepath = os.path.join(self.output_dir, f"{case_name}_Efficiency_Comparison.png")
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_violin(
        self,
        results: Dict[str, List[float]],
        case_name: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Violin plot showing distribution shape.

        Parameters
        ----------
        results : Dict[str, List[float]]
            Dictionary mapping algorithm names to TAC values
        case_name : str
            Case name

        Returns
        -------
        str
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        algorithms = list(results.keys())
        data = [np.array([v for v in results[alg] if v < 1e10]) / 1000 for alg in algorithms]

        # Filter empty
        valid_idx = [i for i, d in enumerate(data) if len(d) > 0]
        algorithms = [algorithms[i] for i in valid_idx]
        data = [data[i] for i in valid_idx]

        if not data:
            plt.close()
            return ""

        parts = ax.violinplot(data, positions=range(len(algorithms)), showmeans=True, showmedians=True)

        # Color violins
        for i, (pc, alg) in enumerate(zip(parts['bodies'], algorithms)):
            pc.set_facecolor(self._get_color(alg))
            pc.set_alpha(0.6)

        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms)
        ax.set_ylabel('TAC (×$1,000/year)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(
                f'{case_name} - Distribution Comparison',
                fontsize=14,
                fontweight='bold'
            )

        ax.grid(True, alpha=0.3, axis='y')

        filepath = os.path.join(self.output_dir, f"{case_name}_Violin_Comparison.png")
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_time_comparison(
        self,
        time_data: Dict[str, List[float]],
        case_name: str,
    ) -> str:
        """
        Bar chart comparing computation times.

        Parameters
        ----------
        time_data : Dict[str, List[float]]
            Dictionary mapping algorithm names to computation times (seconds)
        case_name : str
            Case name

        Returns
        -------
        str
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        algorithms = list(time_data.keys())
        means = [np.mean(time_data[alg]) for alg in algorithms]
        stds = [np.std(time_data[alg]) for alg in algorithms]
        colors = [self._get_color(alg) for alg in algorithms]

        x = range(len(algorithms))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.set_ylabel('Computation Time (seconds)', fontsize=12)
        ax.set_title(f'{case_name} - Computation Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.1,
                f'{mean:.1f}s',
                ha='center',
                fontsize=10,
            )

        filepath = os.path.join(self.output_dir, f"{case_name}_Time_Comparison.png")
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_summary_dashboard(
        self,
        results: Dict[str, List[float]],
        convergence_data: Dict[str, List[float]],
        case_name: str,
    ) -> str:
        """
        Create a summary dashboard with multiple plots.

        Parameters
        ----------
        results : Dict[str, List[float]]
            TAC values per algorithm
        convergence_data : Dict[str, List[float]]
            Convergence history per algorithm
        case_name : str
            Case name

        Returns
        -------
        str
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 10))

        # Create 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Boxplot (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        algorithms = list(results.keys())
        data = [np.array([v for v in results[alg] if v < 1e10]) / 1000 for alg in algorithms]

        if any(len(d) > 0 for d in data):
            bp = ax1.boxplot(
                [d for d in data if len(d) > 0],
                labels=[a for a, d in zip(algorithms, data) if len(d) > 0],
                patch_artist=True
            )
            for patch, alg in zip(bp['boxes'], algorithms):
                if len([v for v in results[alg] if v < 1e10]) > 0:
                    patch.set_facecolor(self._get_color(alg))
                    patch.set_alpha(0.6)

        ax1.set_ylabel('TAC (×$1,000/year)')
        ax1.set_title('TAC Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Convergence (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        for alg, history in convergence_data.items():
            if not history:
                continue
            history_k = [t / 1000 if t and t < 1e10 else np.nan for t in history]
            ax2.plot(history_k, label=alg, color=self._get_color(alg), linewidth=2)

        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Best TAC (×$1,000/year)')
        ax2.set_title('Convergence', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Summary statistics (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        stats_text = []
        for alg in algorithms:
            vals = [v for v in results[alg] if v < 1e10]
            if vals:
                stats_text.append(f"{alg}:")
                stats_text.append(f"  Mean: ${np.mean(vals):,.0f}")
                stats_text.append(f"  Std:  ${np.std(vals):,.0f}")
                stats_text.append(f"  Best: ${np.min(vals):,.0f}")
                stats_text.append("")

        ax3.text(0.1, 0.9, '\n'.join(stats_text), transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        ax3.axis('off')
        ax3.set_title('Summary Statistics', fontweight='bold')

        # 4. Bar chart of means (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        means = []
        stds = []
        valid_algs = []
        for alg in algorithms:
            vals = [v for v in results[alg] if v < 1e10]
            if vals:
                means.append(np.mean(vals) / 1000)
                stds.append(np.std(vals) / 1000)
                valid_algs.append(alg)

        if means:
            x = range(len(valid_algs))
            bars = ax4.bar(x, means, yerr=stds, capsize=5,
                          color=[self._get_color(a) for a in valid_algs], alpha=0.7)
            ax4.set_xticks(x)
            ax4.set_xticklabels(valid_algs)

        ax4.set_ylabel('Mean TAC (×$1,000/year)')
        ax4.set_title('Mean Comparison', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'{case_name} - Algorithm Comparison Summary',
                    fontsize=16, fontweight='bold', y=0.98)

        filepath = os.path.join(self.output_dir, f"{case_name}_Comparison_Dashboard.png")
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath


# ════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def quick_comparison_plot(results: Dict[str, List[float]], case_name: str, output_dir: str = "results") -> List[str]:
    """
    Generate all comparison plots quickly.

    Parameters
    ----------
    results : Dict[str, List[float]]
        TAC values per algorithm
    case_name : str
        Case name
    output_dir : str
        Output directory

    Returns
    -------
    List[str]
        List of generated file paths
    """
    viz = ComparisonVisualizer(output_dir=output_dir)

    files = []
    files.append(viz.plot_boxplot(results, case_name))
    files.append(viz.plot_violin(results, case_name))

    return [f for f in files if f]


# ════════════════════════════════════════════════════════════════════════════
# MAIN (EXAMPLE USAGE)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Comparison Plots Module")
    print("=" * 40)

    # Create sample data
    np.random.seed(42)

    results = {
        "ISO": list(np.random.normal(125000, 2000, 10)),
        "PSO": list(np.random.normal(124000, 2500, 10)),
        "GA": list(np.random.normal(126000, 3000, 10)),
    }

    convergence = {
        "ISO": list(np.linspace(150000, 125000, 20)),
        "PSO": list(np.linspace(160000, 124000, 50)),
        "GA": list(np.linspace(155000, 126000, 100)),
    }

    # Create visualizer
    viz = ComparisonVisualizer(output_dir="results")

    # Generate plots
    print("Generating comparison plots...")

    f1 = viz.plot_boxplot(results, "Demo")
    print(f"  Boxplot: {f1}")

    f2 = viz.plot_convergence_comparison(convergence, "Demo")
    print(f"  Convergence: {f2}")

    f3 = viz.plot_violin(results, "Demo")
    print(f"  Violin: {f3}")

    f4 = viz.plot_summary_dashboard(results, convergence, "Demo")
    print(f"  Dashboard: {f4}")

    print("\nDone!")
