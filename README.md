# Distillation Column Optimization Platform

A comprehensive optimization platform for distillation column design using multiple algorithms: **ISO (Iterative Sequential Optimization)**, **PSO (Particle Swarm Optimization)**, and **GA (Genetic Algorithm)**.

Developed for thesis research at **PSE Lab, NTUST**.

---

## Features

- **Three Optimization Algorithms:**
  - ISO - Iterative Sequential Optimization (deterministic, sequential P->NT->NF)
  - PSO - Particle Swarm Optimization (metaheuristic, swarm-based)
  - GA - Genetic Algorithm (metaheuristic, evolutionary via pymoo)

- **Web Dashboard:** Streamlit-based UI for running optimizations and viewing results
- **Demo Mode:** Test without Aspen Plus using mock evaluator
- **Live Progress Tracking:** Real-time convergence charts for PSO/GA
- **Automatic Visualization:** U-curves for ISO, convergence plots for PSO/GA
- **TAC Economic Model:** Turton et al. (2018) unified costing with CEPCI scaling (base 397, current 800)
- **Soft/Hard Feasibility:** Distinguishes Design Spec-converged (hard) vs forward-mode RR-recovered (soft) feasible points
- **Auto-detection:** Automatic purity targets and styrene-in-feed detection from Aspen at startup
- **Pressure Refinement:** Bisection-based pressure boundary search for ISO optimizer
- **Early Termination:** Smart stopping in NT/NF sweeps after 5 consecutive rising-TAC points
- **Global Best Tracking:** Remembers the best solution across all ISO iterations, preventing loss from oscillation
- **Pressure Carry-Forward:** Injects previous iteration's optimal pressure into the next sweep grid to avoid missing narrow feasibility bands

---

## Project Structure

```
iso-test-case/
├── Core Modules
│   ├── aspen_interface.py      # Aspen Plus COM interface + auto-detection
│   ├── tac_calculator.py       # TAC economic model (Turton v4.0)
│   ├── tac_evaluator.py        # Evaluation wrapper with RR recovery
│   └── config.py               # Case configurations (bounds, column specs)
│
├── Optimization Algorithms
│   ├── iso_optimizer.py        # ISO algorithm (P→NT→NF sequential)
│   ├── pso_optimizer.py        # PSO algorithm with soft feasible rejection
│   ├── ga_optimizer.py         # GA algorithm with soft feasible rejection (pymoo)
│   └── sequential_optimizer.py # Sequential optimization helper
│
├── Entry Points
│   ├── main_sequential_optimizer.py  # CLI entry point
│   ├── server.py                     # FastAPI backend server
│   └── dashboard_streamlit.py        # Streamlit web dashboard
│
├── Visualization
│   ├── visualization_iso.py          # U-curves with soft/hard feasible distinction
│   └── visualization_metaheuristic.py # Convergence plots (for PSO/GA)
│
├── Demo & Testing
│   ├── demo_evaluator.py       # Mock evaluator (no Aspen required)
│   ├── demo_runner.py          # Demo script
│   └── test_client.py          # API test client
│
├── Utilities
│   ├── batch_runner.py         # Batch optimization runner
│   ├── comparison_plots.py     # Multi-algorithm comparison plots
│   └── comparison_stats.py     # Statistical comparison
│
└── results/                    # Output directory for results and plots
```

---

## Installation

### Requirements

```bash
pip install numpy matplotlib streamlit fastapi uvicorn requests pandas
pip install pymoo  # Optional: for advanced GA (falls back to simple GA if not installed)
pip install pywin32  # Required for Aspen Plus connection on Windows
```

### Aspen Plus (Optional)

For real simulations, Aspen Plus must be installed on the same Windows machine.
Demo mode works without Aspen.

---

## Usage

### 1. Web Dashboard (Recommended)

Start the backend server:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Start the dashboard:
```bash
streamlit run dashboard_streamlit.py
```

Then open http://localhost:8501 in your browser.

**Dashboard Features:**
- Select algorithm (ISO, PSO, GA)
- Select case configuration
- Toggle Demo Mode (no Aspen required)
- Adjust algorithm parameters (particles, iterations, etc.)
- View live convergence chart (PSO/GA)
- Browse results and plots

### 2. Command Line

**ISO Optimization:**
```bash
python main_sequential_optimizer.py Case1_COL2 --demo
python main_sequential_optimizer.py Case1_COL2  # With Aspen
```

**PSO Optimization:**
```bash
python pso_optimizer.py Case1_COL2 --demo --n-particles 20 --n-iterations 50
python pso_optimizer.py Case1_COL2 --n-particles 30 --n-iterations 100
```

**GA Optimization:**
```bash
python ga_optimizer.py Case1_COL2 --demo --pop-size 50 --n-generations 100
python ga_optimizer.py Case1_COL2 --pop-size 50 --n-generations 100
```

---

## Algorithm Details

### ISO (Iterative Sequential Optimization)
- Optimizes variables sequentially: Pressure -> NT -> NF
- Iterates until convergence (all three variables stabilize)
- Pressure refinement via bisection at feasibility boundaries
- Generates U-curves showing TAC vs each variable
- Early termination after 5 consecutive rising-TAC feasible points
- Best for understanding variable relationships

### PSO (Particle Swarm Optimization)
- Swarm-based metaheuristic
- Particles explore solution space simultaneously
- Adaptive inertia weight
- Rejects soft feasible (forward-mode) points to ensure fair TAC comparison
- Generates convergence plots (TAC vs iteration)

### GA (Genetic Algorithm)
- Population-based evolutionary algorithm
- Uses pymoo library (or fallback simple GA)
- Selection, crossover, mutation operators
- Rejects soft feasible (forward-mode) points to ensure fair TAC comparison
- Generates convergence plots (TAC vs generation)

---

## TAC Costing Model

The platform uses **Turton et al. (2018)** unified costing (v4.0):

- **Column vessel:** Turton Table A.1 vertical vessel correlation
- **Sieve trays:** Turton Table A.1 with Fq quantity discount factor
- **Heat exchangers:** Turton correlation for condenser and reboiler
- **Vacuum equipment:** Seider Table 22.32 (liquid-ring pump, steam ejector)
- **CEPCI scaling:** Base year 2001 (CEPCI=397), scaled to current (CEPCI=800)
- **Material factors:** Fm=1.7 (SS vessel), Fm=1.189 (SS trays)
- **Pressure factors:** Fp=1.25 (vacuum <0.5 bar), Fp=1.0 (atmospheric)

---

## Configuration

Edit `config.py` to add or modify cases:

```python
CASES = {
    'Case1_COL2': {
        'file_path': r'path\to\aspen\file.apw',
        'column': {
            'block_name': 'COL2',
            'feed_stream': 'FEED',
        },
        'bounds': {
            'nt_bounds': (20, 70),
            'feed_bounds': (10, 60),
            'pressure_bounds': (0.1, 0.5),
        },
        'min_section_stages': 3,
        'T_reb_max': 120.0,
    },
}
```

---

## Output

Results are saved to the `results/` directory:

- **JSON files:** Complete optimization results
  - `iso_result_YYYYMMDD_HHMMSS.json`
  - `pso_result_YYYYMMDD_HHMMSS.json`
  - `ga_result_YYYYMMDD_HHMMSS.json`

- **Plot files (PNG):**
  - ISO: U-curves, pressure sweeps, summary plots
  - PSO/GA: Convergence plots, solution summary

---

## Temperature Constraint

Columns with styrene in their feed stream enforce the reboiler temperature constraint:
```
T_reb <= 120°C
```

Styrene presence is auto-detected from Aspen feed stream composition at startup. This constraint is handled via penalty methods in PSO/GA and constraint checking in ISO.

---

## Recent Changes

### v4.1 - Soft Feasible Rejection in PSO/GA (2026-03-07)
- PSO and GA now reject soft feasible (RR-recovered forward-mode) points
- Forward-mode Q_reb is systematically lower than Design Spec mode, producing artificially low TAC
- All three algorithms now use consistent hard-feasible-only evaluation
- Added `soft_feasible_evaluations` tracking to PSO/GA statistics and JSON output
- Finer sweep resolution: default NT and feed sweep steps changed from 2 to 1
- Fixed ISO config passthrough for column settings

### v4.0 - Turton Unified Costing (2026-02-28)
- Replaced Guthrie (M&S-based, 1969) with Turton et al. (2018) for column vessel and trays
- Corrected CEPCI base from 500 to 397 for heat exchangers
- Added pressure refinement bisection for ISO pressure sweep
- Extended pressure bounds for non-styrene columns

### v3.0 - Soft/Hard Feasibility & Robustness (2026-02-22)
- Soft vs hard feasible TAC distinction in ISO optimizer
- RR warm-starting from both hard and soft feasible points
- Early termination in NT/NF sweeps (5 consecutive rising-TAC points)
- Fixed RR escalation bug (unconditional RR restoration)
- Auto-detection of purity targets and styrene-in-feed from Aspen
- Dual MASS-RECOV Design Spec handling for middle-split columns

### v2.0 - Multi-Algorithm Platform
- Added PSO and GA optimization algorithms
- Web dashboard with live convergence tracking
- Demo mode for testing without Aspen Plus
- Batch optimization runner
- Multi-algorithm comparison tools

---

## License

For academic and research use. PSE Lab, NTUST.