"""
Streamlit dashboard for ISO / Sequential optimization (repo: iso-test-case)

Features:
- Select case (from config.list_available_cases)
- Two modes:
    * Demo / Visualize saved results (load JSON or images)
    * Run optimization (requires Aspen on same Windows machine)
- Show progress with spinner, show resulting plots and summary
- Option to run post-ISO NT-feed sweep and display multiple-U-curves
Notes:
- Running optimization will call run_iso_optimization() which uses Aspen COM.
  That function must be run on the same machine that has Aspen and pywin32.
- For lightweight demos, use demo mode and load JSON from results/*.json.
"""
import streamlit as st
import os
import time
import glob
import json
import threading
from pathlib import Path

# Try to import repo modules (must run app with repo root as cwd)
try:
    from config import list_available_cases, get_case_config
    from main_sequential_optimizer import run_iso_optimization, T_REBOILER_MAX
    from visualization_iso import ISOVisualizer
except Exception as e:
    # If imports fail, we still allow demo mode
    list_available_cases = lambda: []
    get_case_config = lambda name: None
    run_iso_optimization = None
    ISOVisualizer = None
    st.warning(f"Warning: could not import full repo modules: {e}")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="ISO Dashboard", layout="wide")

st.title("ISO / Sequential Optimization Dashboard")
st.markdown(
    """
    Use this dashboard to run ISO optimizations (requires Aspen on the machine)
    or to visualize saved results and U-curves.
    """
)

# Sidebar controls
st.sidebar.header("Controls")

mode = st.sidebar.selectbox("Mode", ["Demo / Visualize", "Run Optimization (Aspen required)"])

# list available cases from config (if import succeeded)
available_cases = list_available_cases()
if not available_cases:
    # If config import failed, try to detect saved case names from results folder
    saved_jsons = glob.glob(str(RESULTS_DIR / "*iso_result_*.json"))
    saved_cases = []
    for p in saved_jsons:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
                saved_cases.append(d.get('metadata', {}).get('case_name', os.path.basename(p)))
        except:
            saved_cases.append(os.path.basename(p))
    available_cases = saved_cases or ["Case1_COL2"]

case = st.sidebar.selectbox("Select case", available_cases)

run_post_sweep = st.sidebar.checkbox("Run post-ISO NT-Feed sweep (multiple U-curves)", value=True)
n_ucurves = st.sidebar.slider("Number of U-curves to show (post-sweep)", 1, 12, 7)

st.sidebar.markdown("---")
st.sidebar.markdown("T_reb safety limit (from code)")
st.sidebar.write(f"{T_REBOILER_MAX if 'T_REBOILER_MAX' in globals() else 120.0} °C")

# Main area
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Available results")
    json_files = sorted(glob.glob(str(RESULTS_DIR / "*.json")), reverse=True)
    img_files = sorted(glob.glob(str(RESULTS_DIR / "*.png")), reverse=True)
    st.write(f"{len(json_files)} saved JSON result files, {len(img_files)} images in {RESULTS_DIR}")

    chosen_json = st.selectbox("Open previous results JSON", ["(none)"] + json_files)
    if chosen_json != "(none)":
        try:
            with open(chosen_json, 'r', encoding='utf-8') as f:
                prev = json.load(f)
            st.json(prev.get('optimal', {}))
            if 'iterations' in prev:
                st.write(f"Iterations: {len(prev['iterations'])}")
        except Exception as e:
            st.error(f"Failed to load JSON: {e}")

with col1:
    st.subheader("Actions")

    if mode == "Demo / Visualize":
        st.markdown("Visualize previously generated plots or import a saved NT-Feed sweep JSON.")
        uploaded = st.file_uploader("Upload NT-Feed sweep JSON (optional)", type=["json"])
        if uploaded:
            try:
                data = json.load(uploaded)
                st.success("Loaded uploaded sweep JSON")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                data = None

        # Show image gallery
        st.markdown("### Image gallery (results/)")
        gallery = img_files[:50]
        for img in gallery:
            st.image(img, use_column_width=True)
        st.markdown("You can also use the standalone plotting function from visualization_iso.py "
                    "or the ISOVisualizer class to re-generate plots from sweep JSON.")

    else:
        st.markdown("Run optimization on this machine (REQUIRES Aspen + pywin32).")
        st.warning("This will call the optimizer which uses Aspen COM; run only on a Windows machine with Aspen installed.")
        run_btn = st.button("Run ISO Optimization now")

        if run_btn:
            if run_iso_optimization is None:
                st.error("Optimizer import failed. Make sure you run this app from the repo root and that Python path is correct.")
            else:
                status = st.empty()
                progress = st.progress(0)
                # Run optimizer in thread to avoid blocking Streamlit main thread UI completely
                result_container = {"done": False, "result": None, "error": None}

                def _run():
                    try:
                        # call run_iso_optimization (this will block until finished)
                        res = run_iso_optimization(case + "_COL2" if "_" not in case else case,
                                                  run_post_sweep=run_post_sweep,
                                                  sweep_nt_step=2,
                                                  sweep_feed_step=2,
                                                  nt_range_around_opt=20,
                                                  feed_range_around_opt=10)
                        result_container["result"] = res
                    except Exception as e:
                        result_container["error"] = str(e)
                    finally:
                        result_container["done"] = True

                thread = threading.Thread(target=_run, daemon=True)
                thread.start()
                t0 = time.time()
                # simple polling loop with progress animation
                while not result_container["done"]:
                    elapsed = time.time() - t0
                    # progress estimate: just animate
                    p = (int(elapsed) % 100) / 100.0
                    progress.progress(min(100, int(p * 100)))
                    status.info(f"Running... elapsed: {int(elapsed)}s")
                    time.sleep(1)

                progress.progress(100)
                if result_container["error"]:
                    st.error(f"Run failed: {result_container['error']}")
                else:
                    st.success("Optimization finished")
                    st.write("Result summary:")
                    st.json(result_container["result"].get('optimal') if result_container["result"] else {})

# Bottom: load latest images and display primary plots
st.markdown("---")
st.header("Visualization: Load latest plot set from results/")

def _latest_by_pattern(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None

latest_summary = _latest_by_pattern(str(RESULTS_DIR / "*ISO_Summary.png"))
latest_pressure = _latest_by_pattern(str(RESULTS_DIR / "*Pressure_Sweep.png"))
latest_ucurves = _latest_by_pattern(str(RESULTS_DIR / "*Multiple_UCurves.png"))
latest_family = _latest_by_pattern(str(RESULTS_DIR / "*UCurves_Family.png"))

col_a, col_b = st.columns(2)
with col_a:
    if latest_summary:
        st.subheader("Latest ISO Summary")
        st.image(latest_summary, use_column_width=True)
    else:
        st.info("No ISO summary image found yet.")

with col_b:
    if latest_ucurves:
        st.subheader("Latest Multiple U-Curves")
        st.image(latest_ucurves, use_column_width=True)
    else:
        st.info("No multiple U-curves image found yet.")

st.markdown("If you want real-time logs, consider modifying run_iso_optimization to write incremental JSON or logs that the dashboard can tail and show.")
st.write("Note: For production use, move long-running simulations to a background worker and show progress via a job queue (Redis/Celery/RQ) or a subprocess with logs streamed to the UI.")