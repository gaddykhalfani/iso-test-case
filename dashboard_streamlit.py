"""
Streamlit dashboard for ISO / PSO / GA optimization (repo: iso-test-case)

Features:
- Select case and algorithm (ISO, PSO, GA)
- Two modes:
    * Demo / Visualize saved results (load JSON or images)
    * Run optimization (requires Aspen on same Windows machine or demo mode)
- Live log streaming from FastAPI backend
- Start/Stop run controls
- Results browser with plot viewing

Usage:
    streamlit run dashboard_streamlit.py

Notes:
- Connect to FastAPI backend at http://localhost:8000
- For real Aspen runs, server must run on Windows machine with Aspen
- Demo mode works without Aspen for testing/demonstration
"""
import streamlit as st
import os
import time
import glob
import json
import requests
from pathlib import Path
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Try to import repo modules for case listing
try:
    from config import list_available_cases, get_case_config
    from iso_optimizer import T_REBOILER_MAX
except Exception as e:
    list_available_cases = lambda: []
    get_case_config = lambda name: None
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
# SESSION STATE INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "job_status" not in st.session_state:
    st.session_state.job_status = None
if "last_progress" not in st.session_state:
    st.session_state.last_progress = {}
if "convergence_history" not in st.session_state:
    st.session_state.convergence_history = []


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


def start_optimization(case: str, algorithm: str, demo: bool, **kwargs):
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
            data = resp.json()
            st.session_state.job_id = data.get("job_id")
            st.session_state.logs = []
            st.session_state.job_status = "running"
            return data
        else:
            st.error(f"Failed to start job: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE_URL}. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Error starting job: {e}")
        return None


def stop_optimization(job_id: str):
    """Stop a running optimization job."""
    try:
        resp = requests.post(f"{API_BASE_URL}/kill/{job_id}", timeout=5)
        if resp.status_code == 200:
            st.session_state.job_status = "killed"
            return resp.json()
        return None
    except Exception as e:
        st.error(f"Error stopping job: {e}")
        return None


def get_job_status(job_id: str):
    """Get status of a job."""
    try:
        resp = requests.get(f"{API_BASE_URL}/status/{job_id}", timeout=5)
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


def parse_progress_line(line: str):
    """Parse a [PROGRESS] JSON line."""
    if "[PROGRESS]" in line:
        try:
            json_str = line.split("[PROGRESS]")[1].strip()
            return json.loads(json_str)
        except:
            pass
    return None


def get_convergence_data(job_id: str):
    """Fetch convergence history from API for live charting."""
    try:
        resp = requests.get(f"{API_BASE_URL}/convergence/{job_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

st.sidebar.title("Column Optimization")
st.sidebar.markdown("---")

# Mode selection
mode = st.sidebar.radio(
    "Mode",
    ["Run Optimization", "Results Browser", "Demo Gallery"],
    index=0
)

st.sidebar.markdown("---")

# API status indicator
api_ok = check_api_health()
if api_ok:
    st.sidebar.success("API Connected")
else:
    st.sidebar.warning("API Not Available")
    st.sidebar.caption(f"Start server: `uvicorn server:app`")

st.sidebar.markdown("---")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["ISO", "PSO", "GA"],
    help="ISO: Iterative Sequential Optimization\nPSO: Particle Swarm\nGA: Genetic Algorithm"
)

# Case selection
available_cases = list_available_cases()
if not available_cases:
    available_cases = ["Case1_COL2", "Case1_COL3", "Case8_COL2", "Case9_COL2"]

case = st.sidebar.selectbox("Case", available_cases)

# Demo mode toggle
demo_mode = st.sidebar.checkbox(
    "Demo Mode (no Aspen)",
    value=True,
    help="Run with mock evaluator for testing"
)

# Algorithm-specific options
st.sidebar.markdown("---")
st.sidebar.subheader("Algorithm Options")

if algorithm in ["PSO", "GA"]:
    n_particles = st.sidebar.slider("Particles/Population", 10, 50, 20)
    n_iterations = st.sidebar.slider("Iterations/Generations", 20, 200, 50)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)
else:
    n_particles = None
    n_iterations = None
    seed = None
    no_sweep = st.sidebar.checkbox("Skip post-ISO sweep", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(f"T_reb limit: {T_REBOILER_MAX}°C")


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════

st.title("Distillation Column Optimization Dashboard")

if mode == "Run Optimization":
    st.header("Run Optimization")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"**Case:** `{case}` | **Algorithm:** `{algorithm}` | **Demo:** `{demo_mode}`")

    with col2:
        start_btn = st.button(
            "Start Optimization",
            type="primary",
            disabled=st.session_state.job_status == "running"
        )

    with col3:
        stop_btn = st.button(
            "Stop",
            type="secondary",
            disabled=st.session_state.job_status != "running"
        )

    # Handle button clicks
    if start_btn:
        kwargs = {}
        if algorithm == "ISO":
            kwargs["no_sweep"] = no_sweep if 'no_sweep' in dir() else False
        else:
            kwargs["n_particles"] = n_particles
            kwargs["n_iterations"] = n_iterations
            kwargs["seed"] = seed

        result = start_optimization(case, algorithm, demo_mode, **kwargs)
        if result:
            st.success(f"Job started: {result.get('job_id', '')[:8]}...")
            st.rerun()

    if stop_btn and st.session_state.job_id:
        stop_optimization(st.session_state.job_id)
        st.warning("Job stopped")
        st.rerun()

    st.markdown("---")

    # Job status and progress
    if st.session_state.job_id:
        status = get_job_status(st.session_state.job_id)

        if status:
            st.session_state.job_status = status.get("status")

            # Status display
            status_col1, status_col2, status_col3, status_col4 = st.columns(4)

            with status_col1:
                status_text = status.get("status", "unknown")
                if status_text == "running":
                    st.info(f"Status: {status_text}")
                elif status_text == "done":
                    st.success(f"Status: {status_text}")
                elif status_text in ("failed", "killed"):
                    st.error(f"Status: {status_text}")
                else:
                    st.warning(f"Status: {status_text}")

            with status_col2:
                elapsed = status.get("elapsed_seconds")
                if elapsed:
                    st.metric("Elapsed", f"{elapsed:.0f}s")

            with status_col3:
                progress = st.session_state.last_progress
                if progress.get("best_tac"):
                    st.metric("Best TAC", f"${progress['best_tac']:,.0f}")

            with status_col4:
                if progress.get("phase"):
                    st.metric("Phase", progress.get("phase", "")[:15])

            # Progress bar
            if status.get("status") == "running" and progress:
                current = progress.get("current", 0)
                total = progress.get("total", 100)
                if total > 0:
                    st.progress(min(current / total, 1.0))

            # Live Convergence Chart (for PSO/GA)
            conv_data = get_convergence_data(st.session_state.job_id)
            if conv_data and conv_data.get("convergence_history"):
                history = conv_data["convergence_history"]
                algo = conv_data.get("algorithm", "")

                if algo in ["PSO", "GA"] and len(history) > 0:
                    st.subheader(f"Live Convergence ({algo})")

                    # Prepare data for chart
                    import pandas as pd
                    chart_data = pd.DataFrame(history)

                    if "iteration" in chart_data.columns and "best_tac" in chart_data.columns:
                        # Filter valid TAC values
                        chart_data = chart_data[chart_data["best_tac"] < 1e10]

                        if len(chart_data) > 0:
                            # Use Streamlit's line chart
                            chart_data = chart_data.set_index("iteration")
                            st.line_chart(
                                chart_data[["best_tac"]],
                                use_container_width=True
                            )

                            # Show current best
                            best_row = chart_data.loc[chart_data["best_tac"].idxmin()]
                            st.caption(
                                f"Current best TAC: ${best_row['best_tac']:,.0f}/year "
                                f"at iteration {chart_data['best_tac'].idxmin()}"
                            )

        # Live logs display
        st.subheader("Live Logs")

        log_container = st.empty()

        # Poll for updates if job is running
        if st.session_state.job_status == "running":
            # Note: For true live streaming, we would use WebSocket
            # This is a simplified polling approach
            with st.spinner("Fetching logs..."):
                try:
                    # Use requests to poll status
                    # In production, connect to WebSocket for real-time updates
                    pass
                except:
                    pass

            # Auto-refresh for live updates
            st.caption("Dashboard auto-refreshes every 3 seconds while job is running.")

            # Auto-refresh using st.rerun with a delay
            time.sleep(3)
            st.rerun()

        # Display accumulated logs
        if st.session_state.logs:
            log_text = "\n".join(st.session_state.logs[-100:])  # Last 100 lines
            st.code(log_text, language="text")

        # Result file link
        if status and status.get("result_file"):
            st.success(f"Results saved to: `{status['result_file']}`")

    else:
        st.info("Click 'Start Optimization' to begin a new run.")

        # Show recent jobs
        st.subheader("Recent Jobs")
        try:
            resp = requests.get(f"{API_BASE_URL}/jobs", timeout=5)
            if resp.status_code == 200:
                jobs = resp.json()
                if jobs:
                    for job in jobs[:5]:
                        st.write(f"- `{job['job_id'][:8]}...` | {job.get('case', 'N/A')} | {job.get('status', 'unknown')}")
                else:
                    st.caption("No recent jobs")
        except:
            st.caption("Cannot fetch job list (API unavailable)")


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
    f"Results: {RESULTS_DIR}"
)
