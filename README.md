# Distillation Column Optimization Platform

A comprehensive optimization platform for distillation column design using multiple algorithms: **ISO (Iterative Sequential Optimization)**, **PSO (Particle Swarm Optimization)**, and **GA (Genetic Algorithm)**.

Developed for thesis research at **PSE Lab, NTUST**.

---

## Features

- **Three Optimization Algorithms:**
  - ISO - Iterative Sequential Optimization (deterministic, sequential variable optimization)
  - PSO - Particle Swarm Optimization (metaheuristic, swarm-based)
  - GA - Genetic Algorithm (metaheuristic, evolutionary)

- **Web Dashboard:** Streamlit-based UI for running optimizations and viewing results
- **Demo Mode:** Test without Aspen Plus using mock evaluator
- **Live Progress Tracking:** Real-time convergence charts for PSO/GA
- **Automatic Visualization:** U-curves for ISO, convergence plots for PSO/GA
- **TAC Economic Model:** Total Annualized Cost calculation

---

## Project Structure

```
iso-test-case/
├── Core Modules
│   ├── aspen_interface.py      # Aspen Plus COM interface
│   ├── tac_calculator.py       # TAC economic model
│   ├── tac_evaluator.py        # Evaluation wrapper
│   └── config.py               # Case configurations (bounds, column specs)
│
├── Optimization Algorithms
│   ├── iso_optimizer.py        # ISO algorithm implementation
│   ├── pso_optimizer.py        # PSO algorithm implementation
│   ├── ga_optimizer.py         # GA algorithm (using pymoo)
│   └── sequential_optimizer.py # Sequential optimization helper
│
├── Entry Points
│   ├── main_sequential_optimizer.py  # CLI entry point
│   ├── server.py                     # FastAPI backend server
│   └── dashboard_streamlit.py        # Streamlit web dashboard
│
├── Visualization
│   ├── visualization_iso.py          # U-curves, pressure sweeps (for ISO)
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
- Optimizes variables sequentially: NT -> Feed -> Pressure
- Iterates until convergence
- Generates U-curves showing TAC vs each variable
- Best for understanding variable relationships

### PSO (Particle Swarm Optimization)
- Swarm-based metaheuristic
- Particles explore solution space simultaneously
- Adaptive inertia weight
- Generates convergence plots (TAC vs iteration)

### GA (Genetic Algorithm)
- Population-based evolutionary algorithm
- Uses pymoo library (or fallback simple GA)
- Selection, crossover, mutation operators
- Generates convergence plots (TAC vs generation)

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

All algorithms enforce the reboiler temperature constraint:
```
T_reb <= 120°C
```

This is handled via penalty methods in PSO/GA and constraint checking in ISO.

---

## Recent Changes

### New Features
- Added PSO and GA optimization algorithms
- Web dashboard with live convergence tracking
- Demo mode for testing without Aspen Plus
- Auto-refresh dashboard during optimization
- Convergence plots for metaheuristic algorithms

### Bug Fixes
- Fixed Unicode encoding issues when running from dashboard
- Fixed None value formatting in results display
- Added COM initialization for subprocess compatibility
- Fixed demo mode fallback in optimizers

---

## License

For academic and research use. PSE Lab, NTUST.