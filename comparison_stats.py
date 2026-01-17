"""
comparison_stats.py
===================
Statistical Comparison Tools for Algorithm Analysis

Includes Wilcoxon tests, effect size calculations, and LaTeX table generation
for thesis research.

Usage:
    from comparison_stats import AlgorithmComparison

    comparison = AlgorithmComparison()
    comparison.add_results("ISO", [125000, 126000, ...])
    comparison.add_results("PSO", [124500, 125500, ...])

    result = comparison.wilcoxon_test("ISO", "PSO")
    print(comparison.generate_latex_table())

Author: PSE Lab, NTUST
Version: 1.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os

# Try to import scipy for statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class StatisticalTestResult:
    """Result from a statistical test."""
    algorithm_1: str
    algorithm_2: str
    test_name: str
    statistic: float
    p_value: float
    significant_005: bool
    significant_001: bool
    effect_size: float
    effect_interpretation: str
    winner: str  # Which algorithm is better (lower TAC)


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM COMPARISON
# ════════════════════════════════════════════════════════════════════════════

class AlgorithmComparison:
    """
    Statistical comparison tools for optimization algorithms.

    Supports:
    - Wilcoxon signed-rank test (paired)
    - Mann-Whitney U test (unpaired)
    - Effect size calculation
    - LaTeX table generation
    - Ranking algorithms
    """

    def __init__(self):
        self.results: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}

    def add_results(self, algorithm: str, tac_values: List[float], **metadata):
        """
        Add results for an algorithm.

        Parameters
        ----------
        algorithm : str
            Algorithm name (e.g., "ISO", "PSO", "GA")
        tac_values : List[float]
            TAC values from multiple runs
        **metadata
            Additional metadata (case_name, n_iterations, etc.)
        """
        self.results[algorithm] = np.array(tac_values)
        self.metadata[algorithm] = metadata

    def load_from_batch_result(self, batch_result):
        """Load results from a BatchResult object."""
        tacs = [r.tac for r in batch_result.all_results if r.tac < 1e10]
        self.add_results(
            batch_result.algorithm,
            tacs,
            case_name=batch_result.case_name,
            n_runs=batch_result.n_runs,
        )

    def load_from_json(self, filepath: str):
        """Load results from a comparison JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for comp in data.get("comparison", []):
            tacs = [r["tac"] for r in comp.get("individual_runs", []) if r["tac"] < 1e10]
            self.add_results(
                comp["algorithm"],
                tacs,
                case_name=comp.get("case_name", ""),
            )

    def wilcoxon_test(self, alg1: str, alg2: str) -> StatisticalTestResult:
        """
        Perform Wilcoxon signed-rank test (paired).

        H0: No difference between algorithms
        H1: Algorithms differ in performance

        Parameters
        ----------
        alg1 : str
            First algorithm name
        alg2 : str
            Second algorithm name

        Returns
        -------
        StatisticalTestResult
            Test results including p-value and effect size
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for statistical tests")

        x = self.results[alg1]
        y = self.results[alg2]

        if len(x) != len(y):
            raise ValueError("Results must have same length for paired test")

        # Perform Wilcoxon test
        statistic, p_value = stats.wilcoxon(x, y)

        # Calculate effect size (r = Z / sqrt(N))
        n = len(x)
        # Approximate Z from statistic
        z = (statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        effect_size = abs(z) / np.sqrt(n)

        # Determine winner
        winner = alg1 if np.mean(x) < np.mean(y) else alg2

        return StatisticalTestResult(
            algorithm_1=alg1,
            algorithm_2=alg2,
            test_name="Wilcoxon signed-rank",
            statistic=statistic,
            p_value=p_value,
            significant_005=p_value < 0.05,
            significant_001=p_value < 0.01,
            effect_size=effect_size,
            effect_interpretation=self._interpret_effect(effect_size),
            winner=winner,
        )

    def mann_whitney_test(self, alg1: str, alg2: str) -> StatisticalTestResult:
        """
        Perform Mann-Whitney U test (unpaired).

        Use when number of runs differs between algorithms.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for statistical tests")

        x = self.results[alg1]
        y = self.results[alg2]

        statistic, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')

        # Effect size (r = Z / sqrt(N))
        n1, n2 = len(x), len(y)
        z = (statistic - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        effect_size = abs(z) / np.sqrt(n1 + n2)

        winner = alg1 if np.mean(x) < np.mean(y) else alg2

        return StatisticalTestResult(
            algorithm_1=alg1,
            algorithm_2=alg2,
            test_name="Mann-Whitney U",
            statistic=statistic,
            p_value=p_value,
            significant_005=p_value < 0.05,
            significant_001=p_value < 0.01,
            effect_size=effect_size,
            effect_interpretation=self._interpret_effect(effect_size),
            winner=winner,
        )

    def _interpret_effect(self, r: float) -> str:
        """Interpret effect size (Cohen's guidelines)."""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"

    def rank_algorithms(self) -> List[Tuple[str, float, float]]:
        """
        Rank algorithms by mean TAC (lower is better).

        Returns
        -------
        List[Tuple[str, float, float]]
            List of (algorithm, mean_tac, std_tac) sorted by mean
        """
        rankings = []
        for alg, values in self.results.items():
            rankings.append((alg, np.mean(values), np.std(values)))
        return sorted(rankings, key=lambda x: x[1])

    def get_summary_stats(self, algorithm: str) -> Dict:
        """Get summary statistics for an algorithm."""
        values = self.results[algorithm]
        return {
            'algorithm': algorithm,
            'n': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
        }

    def pairwise_comparison(self) -> List[StatisticalTestResult]:
        """
        Perform pairwise comparison between all algorithms.

        Returns
        -------
        List[StatisticalTestResult]
            Results for all pairs
        """
        algorithms = list(self.results.keys())
        results = []

        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                alg1, alg2 = algorithms[i], algorithms[j]
                try:
                    result = self.wilcoxon_test(alg1, alg2)
                    results.append(result)
                except ValueError:
                    # Fall back to Mann-Whitney for unequal sizes
                    result = self.mann_whitney_test(alg1, alg2)
                    results.append(result)

        return results

    def generate_latex_table(self) -> str:
        """
        Generate LaTeX table for thesis.

        Returns
        -------
        str
            LaTeX table code
        """
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Algorithm Comparison Results}",
            r"\label{tab:algorithm_comparison}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Algorithm & Mean TAC & Std & Best & Worst & Median \\",
            r"\midrule",
        ]

        for alg, values in self.results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            med_val = np.median(values)

            line = f"{alg} & {mean_val:,.0f} & {std_val:,.0f} & "
            line += f"{min_val:,.0f} & {max_val:,.0f} & {med_val:,.0f} \\\\"
            lines.append(line)

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def generate_statistical_test_table(self) -> str:
        """Generate LaTeX table for statistical tests."""
        comparisons = self.pairwise_comparison()

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Statistical Comparison (Wilcoxon Test)}",
            r"\label{tab:statistical_tests}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Comparison & Test & p-value & Sig. & Effect & Winner \\",
            r"\midrule",
        ]

        for comp in comparisons:
            sig = "**" if comp.significant_001 else ("*" if comp.significant_005 else "n.s.")
            line = f"{comp.algorithm_1} vs {comp.algorithm_2} & "
            line += f"{comp.test_name[:10]} & "
            line += f"{comp.p_value:.4f} & {sig} & "
            line += f"{comp.effect_interpretation} & {comp.winner} \\\\"
            lines.append(line)

        lines.extend([
            r"\bottomrule",
            r"\multicolumn{6}{l}{\small *p<0.05, **p<0.01, n.s.=not significant}",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def generate_markdown_table(self) -> str:
        """Generate Markdown table."""
        lines = [
            "| Algorithm | Mean TAC | Std | Best | Worst | Median |",
            "|-----------|----------|-----|------|-------|--------|",
        ]

        for alg, values in self.results.items():
            line = f"| {alg} | ${np.mean(values):,.0f} | ${np.std(values):,.0f} | "
            line += f"${np.min(values):,.0f} | ${np.max(values):,.0f} | ${np.median(values):,.0f} |"
            lines.append(line)

        return "\n".join(lines)

    def save_report(self, filepath: str):
        """Save comprehensive comparison report."""
        report = {
            "summary": {},
            "rankings": [],
            "statistical_tests": [],
        }

        # Summary stats
        for alg in self.results:
            report["summary"][alg] = self.get_summary_stats(alg)

        # Rankings
        report["rankings"] = [
            {"rank": i + 1, "algorithm": alg, "mean_tac": mean, "std_tac": std}
            for i, (alg, mean, std) in enumerate(self.rank_algorithms())
        ]

        # Statistical tests
        for comp in self.pairwise_comparison():
            report["statistical_tests"].append({
                "comparison": f"{comp.algorithm_1} vs {comp.algorithm_2}",
                "test": comp.test_name,
                "statistic": comp.statistic,
                "p_value": comp.p_value,
                "significant": comp.significant_005,
                "effect_size": comp.effect_size,
                "effect_interpretation": comp.effect_interpretation,
                "winner": comp.winner,
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        return filepath


# ════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def compare_from_json(filepath: str) -> AlgorithmComparison:
    """Load and create comparison from JSON file."""
    comparison = AlgorithmComparison()
    comparison.load_from_json(filepath)
    return comparison


def quick_compare(results_dict: Dict[str, List[float]]) -> str:
    """
    Quick comparison from dictionary.

    Parameters
    ----------
    results_dict : Dict[str, List[float]]
        Dictionary mapping algorithm names to TAC values

    Returns
    -------
    str
        Summary string
    """
    comparison = AlgorithmComparison()
    for alg, values in results_dict.items():
        comparison.add_results(alg, values)

    lines = ["Algorithm Comparison Summary", "=" * 40]

    # Rankings
    lines.append("\nRankings (by mean TAC):")
    for i, (alg, mean, std) in enumerate(comparison.rank_algorithms()):
        lines.append(f"  {i + 1}. {alg}: ${mean:,.0f} +/- ${std:,.0f}")

    # Statistical tests
    if SCIPY_AVAILABLE and len(results_dict) > 1:
        lines.append("\nStatistical Tests:")
        for comp in comparison.pairwise_comparison():
            sig = "significant" if comp.significant_005 else "not significant"
            lines.append(f"  {comp.algorithm_1} vs {comp.algorithm_2}: p={comp.p_value:.4f} ({sig})")
            lines.append(f"    Effect: {comp.effect_interpretation}, Winner: {comp.winner}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# MAIN (EXAMPLE USAGE)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example usage
    print("Comparison Statistics Module")
    print("=" * 40)

    # Create sample data
    np.random.seed(42)
    iso_results = np.random.normal(125000, 2000, 10)
    pso_results = np.random.normal(124000, 2500, 10)
    ga_results = np.random.normal(126000, 3000, 10)

    # Create comparison
    comparison = AlgorithmComparison()
    comparison.add_results("ISO", iso_results.tolist())
    comparison.add_results("PSO", pso_results.tolist())
    comparison.add_results("GA", ga_results.tolist())

    # Print summary
    print(quick_compare({
        "ISO": iso_results.tolist(),
        "PSO": pso_results.tolist(),
        "GA": ga_results.tolist(),
    }))

    print("\n" + "=" * 40)
    print("LaTeX Table:")
    print(comparison.generate_latex_table())

    print("\n" + "=" * 40)
    print("Markdown Table:")
    print(comparison.generate_markdown_table())
