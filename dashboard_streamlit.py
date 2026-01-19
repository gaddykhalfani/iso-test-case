"""
Streamlit dashboard for ISO / PSO / GA optimization (repo: iso-test-case)

Features:
- Support for 4 concurrent optimization runs
- Per-run configuration editing (multi-run safe)
- Per-run case and algorithm selection
- Live status and convergence tracking for all runs
- Results browser with plot viewing

Usage:
    streamlit run dashboard_streamlit.py

Notes:
- Connect to FastAPI backend at http://localhost:8000
- For real Aspen runs, server must run on Windows machine with Aspen
- Demo mode works without Aspen for testing/demonstration
- Config edits are scoped per-run, not global (multi-run safe)
"""
import streamlit as st
import os
import time
import glob
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Number of concurrent run slots
MAX_CONCURRENT_RUNS = 4

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Try to import repo modules for case listing
try:
    from config import list_available_cases, get_case_config, get_column_template, COLUMN_TEMPLATES
    from iso_optimizer import T_REBOILER_MAX
except Exception as e:
    list_available_cases = lambda: []
    get_case_config = lambda name: None
    get_column_template = lambda name: None
    COLUMN_TEMPLATES = {}
    T_REBOILER_MAX = 120.0
    print(f"Warning: could not import full repo modules: {e}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Column Optimization Dashboard",
    page_icon="🧪",
    layout="wide"
)

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION - Multi-Run Safe
# ════════════════════════════════════════════════════════════════════════════

def init_run_slot(slot_id: int):
    """Initialize session state for a single run slot."""
    prefix = f"run_{slot_id}_"

    if f"{prefix}job_id" not in st.session_state:
        st.session_state[f"{prefix}job_id"] = None
    if f"{prefix}logs" not in st.session_state:
        st.session_state[f"{prefix}logs"] = []
    if f"{prefix}job_status" not in st.session_state:
        st.session_state[f"{prefix}job_status"] = None
    if f"{prefix}last_progress" not in st.session_state:
        st.session_state[f"{prefix}last_progress"] = {}
    if f"{prefix}convergence_history" not in st.session_state:
        st.session_state[f"{prefix}convergence_history"] = []
    if f"{prefix}config_overrides" not in st.session_state:
        st.session_state[f"{prefix}config_overrides"] = {}
    if f"{prefix}selected_case" not in st.session_state:
        st.session_state[f"{prefix}selected_case"] = None
    if f"{prefix}algorithm" not in st.session_state:
        st.session_state[f"{prefix}algorithm"] = "ISO"
    if f"{prefix}demo_mode" not in st.session_state:
        st.session_state[f"{prefix}demo_mode"] = True

# Initialize all run slots
for slot in range(MAX_CONCURRENT_RUNS):
    init_run_slot(slot)

# Global mode selection
if "mode" not in st.session_state:
    st.session_state.mode = "Run Optimization"

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def check_api_health():
    """Check if the API backend is running."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False


def start_optimization(case: str, algorithm: str, demo: bool, **kwargs) -> Optional[Dict]:
    """Start an optimization job via the API."""
    payload = {
        "case": case,
        "algorithm": algorithm,
        "demo": demo,
        **kwargs
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/run", json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            # Max concurrent runs reached
            st.warning(f"Maximum concurrent runs ({MAX_CONCURRENT_RUNS}) reached. Please wait for a run to complete.")
            return None
        else:
            st.error(f"Failed to start job: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE_URL}. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Error starting job: {e}")
        return None


def stop_optimization(job_id: str) -> Optional[Dict]:
    """Stop a running optimization job."""
    try:
        resp = requests.post(f"{API_BASE_URL}/kill/{job_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        st.error(f"Error stopping job: {e}")
        return None


def get_job_status(job_id: str) -> Optional[Dict]:
    """Get status of a job."""
    try:
        resp = requests.get(f"{API_BASE_URL}/status/{job_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None


def get_convergence_data(job_id: str) -> Optional[Dict]:
    """Fetch convergence history from API for live charting."""
    try:
        resp = requests.get(f"{API_BASE_URL}/convergence/{job_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None


def fetch_results_list():
    """Fetch list of results from API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/results", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("results", [])
        return []
    except:
        # Fallback to local file listing
        results = []
        for f in sorted(glob.glob(str(RESULTS_DIR / "*.json")), reverse=True):
            results.append({"filename": os.path.basename(f), "filepath": f})
        return results


def get_column_defaults(case_name: str) -> dict:
    """
    Get default column configuration values for a case.
    Returns a dict with bounds and initial values.
    """
    if not case_name or '_' not in case_name:
        return {}

    column_name = case_name.split('_')[1]
    template = get_column_template(column_name)
    if not template:
        return {}

    return {
        'nt_bounds': template.get('nt_bounds', (15, 80)),
        'feed_bounds': template.get('feed_bounds', (10, 70)),
        'pressure_bounds': template.get('pressure_bounds', (0.1, 1.0)),
        'initial_nt': template.get('initial_nt', 25),
        'initial_feed': template.get('initial_feed', 15),
        'initial_pressure': template.get('initial_pressure', 0.2),
        'description': template.get('description', ''),
    }


def build_config_overrides_from_ui(defaults: dict, ui_values: dict) -> dict:
    """
    Build config overrides dict from UI values, only including changed values.
    Returns empty dict if nothing changed from defaults.
    """
    overrides = {}

    # Check bounds
    if ui_values.get('nt_min') is not None and ui_values.get('nt_max') is not None:
        new_nt_bounds = (int(ui_values['nt_min']), int(ui_values['nt_max']))
        if new_nt_bounds != defaults.get('nt_bounds'):
            overrides['nt_bounds'] = new_nt_bounds

    if ui_values.get('feed_min') is not None and ui_values.get('feed_max') is not None:
        new_feed_bounds = (int(ui_values['feed_min']), int(ui_values['feed_max']))
        if new_feed_bounds != defaults.get('feed_bounds'):
            overrides['feed_bounds'] = new_feed_bounds

    if ui_values.get('pressure_min') is not None and ui_values.get('pressure_max') is not None:
        new_pressure_bounds = (float(ui_values['pressure_min']), float(ui_values['pressure_max']))
        if new_pressure_bounds != defaults.get('pressure_bounds'):
            overrides['pressure_bounds'] = new_pressure_bounds

    # Check initial values
    if ui_values.get('initial_nt') is not None:
        if int(ui_values['initial_nt']) != defaults.get('initial_nt'):
            overrides['initial_nt'] = int(ui_values['initial_nt'])

    if ui_values.get('initial_feed') is not None:
        if int(ui_values['initial_feed']) != defaults.get('initial_feed'):
            overrides['initial_feed'] = int(ui_values['initial_feed'])

    if ui_values.get('initial_pressure') is not None:
        if float(ui_values['initial_pressure']) != defaults.get('initial_pressure'):
            overrides['initial_pressure'] = float(ui_values['initial_pressure'])

    return overrides


def get_active_runs_count() -> int:
    """Count how many runs are currently active."""
    count = 0
    for slot in range(MAX_CONCURRENT_RUNS):
        status = st.session_state.get(f"run_{slot}_job_status")
        if status == "running":
            count += 1
    return count


# ════════════════════════════════════════════════════════════════════════════
# RUN SLOT COMPONENT - Renders a single run panel with config editing
# ════════════════════════════════════════════════════════════════════════════

def render_run_slot(slot_id: int, available_cases: list, api_ok: bool):
    """
    Render a single run slot with all controls and status.

    Each slot is completely independent with its own:
    - Case selection
    - Algorithm selection
    - Config overrides (multi-run safe)
    - Start/Stop controls
    - Status display
    """
    prefix = f"run_{slot_id}_"

    # Get current state for this slot
    job_id = st.session_state.get(f"{prefix}job_id")
    job_status = st.session_state.get(f"{prefix}job_status")
    last_progress = st.session_state.get(f"{prefix}last_progress", {})

    # Status color indicator
    status_colors = {
        "running": "🟢",
        "done": "✅",
        "failed": "🔴",
        "killed": "🟠",
        None: "⚪"
    }
    status_icon = status_colors.get(job_status, "⚪")

    with st.container():
        st.subheader(f"{status_icon} Run Slot {slot_id + 1}")

        # Row 1: Case, Algorithm, Demo mode selection
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            case = st.selectbox(
                "Case",
                available_cases,
                key=f"{prefix}case_select",
                disabled=job_status == "running"
            )
            # Update selected case in session state
            if case != st.session_state.get(f"{prefix}selected_case"):
                st.session_state[f"{prefix}selected_case"] = case
                st.session_state[f"{prefix}config_overrides"] = {}  # Reset overrides on case change

        with col2:
            algorithm = st.selectbox(
                "Algorithm",
                ["ISO", "PSO", "GA"],
                key=f"{prefix}algo_select",
                disabled=job_status == "running"
            )
            st.session_state[f"{prefix}algorithm"] = algorithm

        with col3:
            demo_mode = st.checkbox(
                "Demo Mode",
                value=st.session_state.get(f"{prefix}demo_mode", True),
                key=f"{prefix}demo_checkbox",
                disabled=job_status == "running",
                help="Run without Aspen"
            )
            st.session_state[f"{prefix}demo_mode"] = demo_mode

        # Row 2: Config Editor (expandable)
        defaults = get_column_defaults(case) if case else {}
        config_overrides = {}

        if defaults:
            with st.expander("Edit Config (bounds & initial values)", expanded=False):
                st.caption("Changes apply to THIS run only (multi-run safe)")

                # NT bounds
                st.markdown("**Number of Trays (NT)**")
                nt_col1, nt_col2, nt_col3 = st.columns(3)
                with nt_col1:
                    nt_min = st.number_input(
                        "Min",
                        min_value=1, max_value=200,
                        value=int(defaults['nt_bounds'][0]),
                        key=f"{prefix}nt_min",
                        disabled=job_status == "running"
                    )
                with nt_col2:
                    nt_max = st.number_input(
                        "Max",
                        min_value=1, max_value=200,
                        value=int(defaults['nt_bounds'][1]),
                        key=f"{prefix}nt_max",
                        disabled=job_status == "running"
                    )
                with nt_col3:
                    initial_nt = st.number_input(
                        "Initial",
                        min_value=int(nt_min), max_value=int(nt_max),
                        value=min(max(int(defaults['initial_nt']), int(nt_min)), int(nt_max)),
                        key=f"{prefix}initial_nt",
                        disabled=job_status == "running"
                    )

                # Feed bounds
                st.markdown("**Feed Stage**")
                feed_col1, feed_col2, feed_col3 = st.columns(3)
                with feed_col1:
                    feed_min = st.number_input(
                        "Min",
                        min_value=1, max_value=200,
                        value=int(defaults['feed_bounds'][0]),
                        key=f"{prefix}feed_min",
                        disabled=job_status == "running"
                    )
                with feed_col2:
                    feed_max = st.number_input(
                        "Max",
                        min_value=1, max_value=200,
                        value=int(defaults['feed_bounds'][1]),
                        key=f"{prefix}feed_max",
                        disabled=job_status == "running"
                    )
                with feed_col3:
                    initial_feed = st.number_input(
                        "Initial",
                        min_value=int(feed_min), max_value=int(feed_max),
                        value=min(max(int(defaults['initial_feed']), int(feed_min)), int(feed_max)),
                        key=f"{prefix}initial_feed",
                        disabled=job_status == "running"
                    )

                # Pressure bounds
                st.markdown("**Pressure (bar)**")
                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1:
                    pressure_min = st.number_input(
                        "Min",
                        min_value=0.001, max_value=10.0,
                        value=float(defaults['pressure_bounds'][0]),
                        step=0.01, format="%.3f",
                        key=f"{prefix}pressure_min",
                        disabled=job_status == "running"
                    )
                with p_col2:
                    pressure_max = st.number_input(
                        "Max",
                        min_value=0.001, max_value=10.0,
                        value=float(defaults['pressure_bounds'][1]),
                        step=0.01, format="%.3f",
                        key=f"{prefix}pressure_max",
                        disabled=job_status == "running"
                    )
                with p_col3:
                    initial_pressure = st.number_input(
                        "Initial",
                        min_value=float(pressure_min), max_value=float(pressure_max),
                        value=min(max(float(defaults['initial_pressure']), float(pressure_min)), float(pressure_max)),
                        step=0.01, format="%.3f",
                        key=f"{prefix}initial_pressure",
                        disabled=job_status == "running"
                    )

                # Build overrides
                ui_values = {
                    'nt_min': nt_min, 'nt_max': nt_max,
                    'feed_min': feed_min, 'feed_max': feed_max,
                    'pressure_min': pressure_min, 'pressure_max': pressure_max,
                    'initial_nt': initial_nt, 'initial_feed': initial_feed,
                    'initial_pressure': initial_pressure,
                }
                config_overrides = build_config_overrides_from_ui(defaults, ui_values)

                if config_overrides:
                    st.markdown("**Modified:**")
                    for key, value in config_overrides.items():
                        st.caption(f"• {key}: {value}")
                else:
                    st.caption("Using default values")

        # Algorithm-specific options
        algo_kwargs = {}
        if algorithm in ["PSO", "GA"]:
            opt_col1, opt_col2, opt_col3 = st.columns(3)
            with opt_col1:
                n_particles = st.number_input(
                    "Particles/Pop",
                    min_value=5, max_value=100, value=20,
                    key=f"{prefix}n_particles",
                    disabled=job_status == "running"
                )
            with opt_col2:
                n_iterations = st.number_input(
                    "Iterations",
                    min_value=10, max_value=500, value=50,
                    key=f"{prefix}n_iterations",
                    disabled=job_status == "running"
                )
            with opt_col3:
                seed = st.number_input(
                    "Seed",
                    min_value=0, max_value=9999, value=42,
                    key=f"{prefix}seed",
                    disabled=job_status == "running"
                )
            algo_kwargs = {"n_particles": n_particles, "n_iterations": n_iterations, "seed": seed}
        else:
            no_sweep = st.checkbox(
                "Skip post-sweep",
                value=False,
                key=f"{prefix}no_sweep",
                disabled=job_status == "running"
            )
            algo_kwargs = {"no_sweep": no_sweep}

        # Row 3: Start/Stop buttons
        btn_col1, btn_col2, status_col = st.columns([1, 1, 2])

        with btn_col1:
            start_disabled = not api_ok or job_status == "running"
            if st.button("Start", key=f"{prefix}start_btn", type="primary", disabled=start_disabled):
                # Prepare kwargs
                kwargs = {**algo_kwargs}
                if config_overrides:
                    kwargs["config_overrides"] = config_overrides

                result = start_optimization(case, algorithm, demo_mode, **kwargs)
                if result:
                    st.session_state[f"{prefix}job_id"] = result.get("job_id")
                    st.session_state[f"{prefix}logs"] = []
                    st.session_state[f"{prefix}job_status"] = "running"
                    st.session_state[f"{prefix}convergence_history"] = []
                    st.session_state[f"{prefix}last_progress"] = {}
                    st.rerun()

        with btn_col2:
            stop_disabled = job_status != "running"
            if st.button("Stop", key=f"{prefix}stop_btn", type="secondary", disabled=stop_disabled):
                if job_id:
                    stop_optimization(job_id)
                    st.session_state[f"{prefix}job_status"] = "killed"
                    st.rerun()

        with status_col:
            if job_status:
                status_text = f"**Status:** {job_status}"
                if job_status == "running":
                    st.info(status_text)
                elif job_status == "done":
                    st.success(status_text)
                elif job_status in ("failed", "killed"):
                    st.error(status_text)
                else:
                    st.warning(status_text)

        # Row 4: Live status (if running or has result)
        if job_id:
            status = get_job_status(job_id)
            if status:
                st.session_state[f"{prefix}job_status"] = status.get("status")
                last_progress = status.get("last_progress", {})
                st.session_state[f"{prefix}last_progress"] = last_progress

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    elapsed = status.get("elapsed_seconds")
                    st.metric("Elapsed", f"{elapsed:.0f}s" if elapsed else "-")
                with m2:
                    best_tac = last_progress.get("best_tac")
                    st.metric("Best TAC", f"${best_tac:,.0f}" if best_tac else "-")
                with m3:
                    phase = last_progress.get("phase", "-")
                    st.metric("Phase", phase[:12] if phase else "-")
                with m4:
                    iteration = last_progress.get("iteration", 0)
                    st.metric("Iteration", iteration)

                # Progress bar
                if status.get("status") == "running" and last_progress:
                    current = last_progress.get("current", 0)
                    total = last_progress.get("total", 100)
                    if total > 0:
                        st.progress(min(current / total, 1.0))

                # Mini convergence chart (for PSO/GA)
                conv_data = get_convergence_data(job_id)
                if conv_data and conv_data.get("convergence_history"):
                    history = conv_data["convergence_history"]
                    algo = conv_data.get("algorithm", "")

                    if algo in ["PSO", "GA"] and len(history) > 2:
                        import pandas as pd
                        chart_data = pd.DataFrame(history)
                        if "iteration" in chart_data.columns and "best_tac" in chart_data.columns:
                            chart_data = chart_data[chart_data["best_tac"] < 1e10]
                            if len(chart_data) > 0:
                                chart_data = chart_data.set_index("iteration")
                                st.line_chart(chart_data[["best_tac"]], height=150)

                # Result file link
                if status.get("result_file"):
                    st.caption(f"Results: `{status['result_file']}`")

        st.divider()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

st.sidebar.title("Column Optimization")
st.sidebar.caption(f"Multi-Run Dashboard (up to {MAX_CONCURRENT_RUNS} concurrent)")
st.sidebar.markdown("---")

# Mode selection
mode = st.sidebar.radio(
    "Mode",
    ["Run Optimization", "Results Browser", "Demo Gallery"],
    index=0
)
st.session_state.mode = mode

st.sidebar.markdown("---")

# API status indicator
api_ok = check_api_health()
if api_ok:
    st.sidebar.success("API Connected")
    active_count = get_active_runs_count()
    st.sidebar.caption(f"Active runs: {active_count}/{MAX_CONCURRENT_RUNS}")
else:
    st.sidebar.warning("API Not Available")
    st.sidebar.caption(f"Start server: `uvicorn server:app`")

st.sidebar.markdown("---")
st.sidebar.caption(f"T_reb limit: {T_REBOILER_MAX}C")

# Quick actions
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Actions")

if st.sidebar.button("Refresh All Status"):
    st.rerun()

if st.sidebar.button("Stop All Runs"):
    for slot in range(MAX_CONCURRENT_RUNS):
        job_id = st.session_state.get(f"run_{slot}_job_id")
        if job_id and st.session_state.get(f"run_{slot}_job_status") == "running":
            stop_optimization(job_id)
            st.session_state[f"run_{slot}_job_status"] = "killed"
    st.rerun()

if st.sidebar.button("Clear Completed"):
    for slot in range(MAX_CONCURRENT_RUNS):
        status = st.session_state.get(f"run_{slot}_job_status")
        if status in ("done", "failed", "killed"):
            st.session_state[f"run_{slot}_job_id"] = None
            st.session_state[f"run_{slot}_job_status"] = None
            st.session_state[f"run_{slot}_logs"] = []
            st.session_state[f"run_{slot}_last_progress"] = {}
            st.session_state[f"run_{slot}_convergence_history"] = []
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════

st.title("Distillation Column Optimization Dashboard")

# Get available cases
available_cases = list_available_cases()
if not available_cases:
    available_cases = ["Case1_COL2", "Case1_COL3", "Case8_COL2", "Case9_COL2"]

if mode == "Run Optimization":
    st.header("Multi-Run Optimization")
    st.caption("Configure and run up to 4 optimizations concurrently. Each run has isolated configuration (multi-run safe).")

    # Check for any running jobs and auto-refresh
    any_running = any(
        st.session_state.get(f"run_{slot}_job_status") == "running"
        for slot in range(MAX_CONCURRENT_RUNS)
    )

    # Render run slots in a 2x2 grid
    col_left, col_right = st.columns(2)

    with col_left:
        render_run_slot(0, available_cases, api_ok)
        render_run_slot(2, available_cases, api_ok)

    with col_right:
        render_run_slot(1, available_cases, api_ok)
        render_run_slot(3, available_cases, api_ok)

    # Auto-refresh if any job is running
    if any_running:
        st.caption("Dashboard auto-refreshes while jobs are running...")
        time.sleep(3)
        st.rerun()


elif mode == "Results Browser":
    st.header("Results Browser")

    # Fetch results
    results = fetch_results_list()

    if not results:
        st.info("No results found. Run an optimization first!")
    else:
        # Results table
        st.subheader(f"Found {len(results)} result files")

        # Filter by algorithm
        algorithms_found = list(set(r.get("algorithm", "ISO") for r in results if "algorithm" in r))
        if algorithms_found:
            filter_algo = st.multiselect("Filter by algorithm", algorithms_found, default=algorithms_found)
            results = [r for r in results if r.get("algorithm", "ISO") in filter_algo]

        # Display results
        for r in results[:20]:  # Limit to 20
            tac_display = r.get('optimal_tac') or 0
            with st.expander(f"{r.get('filename', 'Unknown')} - TAC: ${tac_display:,.0f}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Case:** {r.get('case_name', 'N/A')}")
                    st.write(f"**Algorithm:** {r.get('algorithm', 'ISO')}")
                    st.write(f"**Optimal NT:** {r.get('optimal_nt', 'N/A')}")
                    st.write(f"**Optimal Feed:** {r.get('optimal_feed', 'N/A')}")
                    st.write(f"**Optimal Pressure:** {r.get('optimal_pressure', 'N/A')}")

                with col2:
                    tac_val = r.get('optimal_tac') or 0
                    st.write(f"**TAC:** ${tac_val:,.0f}/year")
                    st.write(f"**Evaluations:** {r.get('total_evaluations', 'N/A')}")
                    time_val = r.get('time_seconds') or 0
                    st.write(f"**Time:** {time_val:.1f}s")
                    st.write(f"**Modified:** {r.get('modified', 'N/A')}")

                # Load full JSON
                if st.button(f"Load JSON", key=f"load_{r.get('filename')}"):
                    try:
                        filepath = r.get('filepath', os.path.join(RESULTS_DIR, r.get('filename')))
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        st.json(data)
                    except Exception as e:
                        st.error(f"Error loading: {e}")

    # Plots section
    st.markdown("---")
    st.subheader("Plots")

    png_files = sorted(glob.glob(str(RESULTS_DIR / "*.png")), key=os.path.getmtime, reverse=True)

    if png_files:
        # Group by type
        st.write(f"Found {len(png_files)} plot files")

        # Filter
        plot_filter = st.text_input("Filter plots by name", "")

        filtered_plots = [p for p in png_files if plot_filter.lower() in os.path.basename(p).lower()]

        # Display in grid
        cols = st.columns(2)
        for i, plot in enumerate(filtered_plots[:10]):
            with cols[i % 2]:
                st.image(plot, caption=os.path.basename(plot), use_container_width=True)
    else:
        st.info("No plots found in results folder.")


elif mode == "Demo Gallery":
    st.header("Demo Gallery")
    st.markdown("Pre-generated results and plots for demonstration.")

    # Show sample images
    st.subheader("Sample U-Curves")

    png_files = sorted(glob.glob(str(RESULTS_DIR / "*.png")), key=os.path.getmtime, reverse=True)

    if png_files:
        # Categories
        ucurve_plots = [p for p in png_files if "UCurve" in os.path.basename(p) or "U_curve" in os.path.basename(p)]
        pressure_plots = [p for p in png_files if "Pressure" in os.path.basename(p)]
        summary_plots = [p for p in png_files if "Summary" in os.path.basename(p)]
        other_plots = [p for p in png_files if p not in ucurve_plots + pressure_plots + summary_plots]

        tab1, tab2, tab3, tab4 = st.tabs(["U-Curves", "Pressure Sweeps", "Summaries", "Other"])

        with tab1:
            if ucurve_plots:
                for plot in ucurve_plots[:6]:
                    st.image(plot, caption=os.path.basename(plot), use_container_width=True)
            else:
                st.info("No U-curve plots found")

        with tab2:
            if pressure_plots:
                for plot in pressure_plots[:6]:
                    st.image(plot, caption=os.path.basename(plot), use_container_width=True)
            else:
                st.info("No pressure sweep plots found")

        with tab3:
            if summary_plots:
                for plot in summary_plots[:6]:
                    st.image(plot, caption=os.path.basename(plot), use_container_width=True)
            else:
                st.info("No summary plots found")

        with tab4:
            if other_plots:
                cols = st.columns(2)
                for i, plot in enumerate(other_plots[:8]):
                    with cols[i % 2]:
                        st.image(plot, caption=os.path.basename(plot), use_container_width=True)
            else:
                st.info("No other plots found")

    else:
        st.info("No plots found. Run an optimization to generate plots!")

    # Sample JSON
    st.markdown("---")
    st.subheader("Sample Results")

    json_files = sorted(glob.glob(str(RESULTS_DIR / "*.json")), key=os.path.getmtime, reverse=True)

    if json_files:
        selected_json = st.selectbox("Select result file", json_files[:10])

        if selected_json:
            try:
                with open(selected_json, 'r') as f:
                    data = json.load(f)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Optimal Configuration:**")
                    st.json(data.get("optimal", {}))

                with col2:
                    st.write("**Statistics:**")
                    st.json(data.get("statistics", {}))

                with st.expander("Full JSON"):
                    st.json(data)

            except Exception as e:
                st.error(f"Error loading: {e}")
    else:
        st.info("No result files found.")


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    f"Column Optimization Dashboard | "
    f"API: {API_BASE_URL} | "
    f"Results: {RESULTS_DIR} | "
    f"Max Concurrent: {MAX_CONCURRENT_RUNS}"
)