"""
Streamlit dashboard for ISO / PSO / GA optimization (repo: iso-test-case)

Features:
- Support for 4 concurrent optimization runs
- Per-run configuration editing (multi-run safe)
- Per-run case and algorithm selection
- Live status and convergence tracking for all runs
- Combined convergence chart overlay
- 3D surface plots and heatmaps
- Pareto front visualization
- Statistics dashboard with analytics
- Historical run comparison
- Export to CSV/ZIP
- Discord notifications
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
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from io import BytesIO

# Plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Pandas for data manipulation
import pandas as pd

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
# CUSTOM CSS STYLING
# ════════════════════════════════════════════════════════════════════════════

# Algorithm colors (matching visualization modules)
ALGO_COLORS = {
    'ISO': '#4fc3f7',  # Electric blue
    'PSO': '#ffb74d',  # Warm amber
    'GA': '#66bb6a',   # Emerald green
}

STATUS_COLORS = {
    'running': '#4cdf8a',   # Bright green
    'done': '#4fc3f7',      # Electric blue
    'failed': '#ef5350',    # Bright red
    'killed': '#ffb74d',    # Warm amber
}

st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════════
       KEYFRAME ANIMATIONS
       ═══════════════════════════════════════════════════════════════ */
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.1); }
    }

    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ═══════════════════════════════════════════════════════════════
       BASE & CONTAINER STYLES
       ═══════════════════════════════════════════════════════════════ */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }

    .stApp {
        background-color: #0f1419;
    }

    /* ═══════════════════════════════════════════════════════════════
       GLASS-MORPHISM RUN SLOT CARDS
       ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background: linear-gradient(135deg, #1e2030 0%, #1a1c2e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow:
            0 4px 24px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.2s ease;
    }

    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:hover {
        border-color: rgba(79, 195, 247, 0.15);
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.4),
            0 0 20px rgba(79, 195, 247, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
        transform: translateY(-1px);
    }

    /* ═══════════════════════════════════════════════════════════════
       SIDEBAR STYLING
       ═══════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 40%, #0d1117 100%);
        border-right: 1px solid rgba(79, 195, 247, 0.08);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e6edf3 !important;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #8b949e !important;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label {
        color: #c9d1d9 !important;
        transition: color 0.2s ease;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        color: #4fc3f7 !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       METRIC CARDS
       ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #161a26 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 3px solid #4fc3f7;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }

    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
    }

    /* ═══════════════════════════════════════════════════════════════
       BUTTON STYLING
       ═══════════════════════════════════════════════════════════════ */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 50%, #0288d1 100%);
        border: none;
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.78rem;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(79, 195, 247, 0.25);
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 195, 247, 0.35);
        background: linear-gradient(135deg, #81d4fa 0%, #4fc3f7 50%, #29b6f6 100%);
    }

    .stButton > button[kind="primary"]:active {
        transform: translateY(0px);
        box-shadow: 0 2px 6px rgba(79, 195, 247, 0.2);
    }

    .stButton > button[kind="secondary"] {
        background: transparent;
        border: 2px solid #ef5350;
        color: #ef5350;
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.78rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton > button[kind="secondary"]:hover {
        background: rgba(239, 83, 80, 0.12);
        border-color: #ef5350;
        color: #ff8a80;
        box-shadow: 0 0 15px rgba(239, 83, 80, 0.2);
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(79, 195, 247, 0.08);
        border: 1px solid rgba(79, 195, 247, 0.2);
        color: #c9d1d9;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(79, 195, 247, 0.15);
        border-color: rgba(79, 195, 247, 0.4);
        color: #4fc3f7;
    }

    /* ═══════════════════════════════════════════════════════════════
       EXPANDER STYLING
       ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stExpander"] {
        background: rgba(30, 32, 48, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        overflow: hidden;
        backdrop-filter: blur(8px);
    }

    div[data-testid="stExpander"] summary {
        font-weight: 600;
        color: #c9d1d9;
        padding: 0.75rem 1rem;
    }

    div[data-testid="stExpander"] summary:hover {
        color: #4fc3f7;
    }

    /* ═══════════════════════════════════════════════════════════════
       SELECT BOX & INPUTS
       ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stSelectbox"] > div {
        border-radius: 10px;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 8px;
        background: #161a26;
        border-color: rgba(255, 255, 255, 0.08);
        color: #e6edf3;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #4fc3f7;
        box-shadow: 0 0 0 2px rgba(79, 195, 247, 0.15);
    }

    /* ═══════════════════════════════════════════════════════════════
       PROGRESS BAR
       ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stProgress"] > div {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #4fc3f7, #29b6f6, #4fc3f7, #81d4fa);
        background-size: 200% 100%;
        border-radius: 12px;
        animation: shimmer 2.5s ease-in-out infinite;
    }

    /* ═══════════════════════════════════════════════════════════════
       ALERT BOXES
       ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border-left-width: 4px;
        background: rgba(30, 32, 48, 0.6);
    }

    /* ═══════════════════════════════════════════════════════════════
       DIVIDER
       ═══════════════════════════════════════════════════════════════ */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(79, 195, 247, 0.15), transparent);
        margin: 1.5rem 0;
    }

    /* ═══════════════════════════════════════════════════════════════
       TABS
       ═══════════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(30, 32, 48, 0.4);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        color: #8b949e;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #e6edf3;
        background: rgba(79, 195, 247, 0.08);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(79, 195, 247, 0.12) !important;
        color: #4fc3f7 !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       CHART CONTAINER
       ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stArrowVegaLiteChart"] {
        background: rgba(22, 26, 38, 0.8);
        border-radius: 12px;
        padding: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }

    /* ═══════════════════════════════════════════════════════════════
       CAPTION & TEXT
       ═══════════════════════════════════════════════════════════════ */
    .stCaption, span[data-testid="stCaptionContainer"] {
        color: #8b949e !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       SCROLLBAR
       ═══════════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0f1419;
    }

    ::-webkit-scrollbar-thumb {
        background: #2d333b;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #444c56;
    }

    /* ═══════════════════════════════════════════════════════════════
       DROPDOWN POPOVER
       ═══════════════════════════════════════════════════════════════ */
    [data-baseweb="popover"] {
        background: #1e2030 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }

    [data-baseweb="menu"] {
        background: #1e2030 !important;
    }
</style>
""", unsafe_allow_html=True)

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
# STYLING HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def get_algorithm_badge(algorithm: str) -> str:
    """Generate an HTML badge for the algorithm type."""
    color = ALGO_COLORS.get(algorithm, '#8b949e')
    return f'''<span style="
        background: {color}18;
        color: {color};
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-left: 0.5rem;
        display: inline-block;
        border: 1px solid {color}40;
        text-transform: uppercase;
    ">{algorithm}</span>'''


def get_status_badge(status: str) -> str:
    """Generate an HTML badge for the job status."""
    colors = {
        'running': ('#4cdf8a', '#4cdf8a15'),
        'done': ('#4fc3f7', '#4fc3f715'),
        'failed': ('#ef5350', '#ef535015'),
        'killed': ('#ffb74d', '#ffb74d15'),
    }
    text_color, bg_color = colors.get(status, ('#8b949e', '#8b949e15'))
    icon = {'running': '&#9679;', 'done': '&#10003;', 'failed': '&#10007;', 'killed': '&#8856;'}.get(status, '&#9675;')
    animation_style = 'animation: pulse-glow 1.5s ease-in-out infinite;' if status == 'running' else ''
    return f'''<span style="
        background: {bg_color};
        color: {text_color};
        padding: 0.3rem 0.7rem;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid {text_color}30;
        letter-spacing: 0.03em;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    "><span style="{animation_style} display: inline-block;">{icon}</span> {status.upper()}</span>'''


def get_slot_header_style(algorithm: str) -> str:
    """Get the border color for run slot based on algorithm."""
    color = ALGO_COLORS.get(algorithm, '#4fc3f7')
    return f"border-left: 3px solid {color}; padding-left: 0.75rem;"


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


def get_all_convergence() -> Optional[Dict]:
    """Fetch convergence history for all jobs (for combined chart)."""
    try:
        resp = requests.get(f"{API_BASE_URL}/convergence/all", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None


@st.cache_data(ttl=5)
def get_logs(job_id: str, tail: int = 500) -> Optional[Dict]:
    """Fetch logs for a specific job (cached for 5 seconds)."""
    try:
        params = {"tail": tail}
        resp = requests.get(f"{API_BASE_URL}/logs/{job_id}", params=params, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None


@st.cache_data(ttl=30)
def get_statistics() -> Optional[Dict]:
    """Fetch aggregated statistics from all historical results."""
    try:
        resp = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None


def set_discord_webhook(webhook_url: str) -> bool:
    """Set the Discord webhook URL on the server."""
    try:
        resp = requests.post(f"{API_BASE_URL}/discord/webhook",
                           json={"webhook_url": webhook_url}, timeout=5)
        return resp.status_code == 200
    except:
        return False


def test_discord_webhook(webhook_url: str = None) -> Dict:
    """Test the Discord webhook by sending a test notification."""
    try:
        payload = {}
        if webhook_url:
            payload["webhook_url"] = webhook_url
        resp = requests.post(f"{API_BASE_URL}/discord/test", json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"success": False, "message": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@st.cache_data(ttl=300)
def load_sweep_data(result_folder: str) -> Optional[List[Dict]]:
    """Load sweep data from a result folder for 3D plots."""
    sweep_files = glob.glob(os.path.join(result_folder, "*_NT_Feed_Sweep.json"))
    if not sweep_files:
        return None

    all_points = []
    for sweep_file in sweep_files:
        try:
            with open(sweep_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_points.extend(data)
        except:
            continue

    return all_points if all_points else None


def compute_pareto_frontier(points: List[Dict], x_key: str, y_key: str) -> List[Dict]:
    """
    Compute the Pareto frontier for minimizing both objectives.

    Args:
        points: List of evaluation points with x_key and y_key values
        x_key: Key for first objective (e.g., 'tac')
        y_key: Key for second objective (e.g., 'q_reb')

    Returns:
        List of non-dominated points forming the Pareto frontier
    """
    # Filter valid points
    valid = [p for p in points if p.get(x_key) and p.get(y_key)
             and p[x_key] < 1e10 and p[y_key] < 1e10]

    if not valid:
        return []

    # Sort by x_key
    sorted_pts = sorted(valid, key=lambda p: p[x_key])

    # Build Pareto frontier (for minimization of both)
    pareto = [sorted_pts[0]]
    for p in sorted_pts[1:]:
        if p[y_key] < pareto[-1][y_key]:
            pareto.append(p)

    return pareto


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATION COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

def render_combined_convergence_chart():
    """Render a combined convergence chart showing all active/recent runs."""
    conv_data = get_all_convergence()
    if not conv_data or not conv_data.get("jobs"):
        return

    jobs_with_data = [j for j in conv_data["jobs"] if j.get("convergence_history")]
    if not jobs_with_data:
        return

    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    ">
        <span style="font-size: 1.1rem; color: #4fc3f7;">📈</span>
        <h4 style="margin: 0; color: #e6edf3; font-weight: 600;">Live Convergence Comparison</h4>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()

    for job in jobs_with_data:
        history = job["convergence_history"]
        algo = job.get("algorithm", "ISO")
        case = job.get("case", "Unknown")
        status = job.get("status", "")

        # Extract iterations and TAC values
        iterations = [h.get("iteration", i) for i, h in enumerate(history)]
        tac_values = [h.get("best_tac", 0) for h in history]

        # Filter out invalid TAC values
        valid_data = [(it, tac) for it, tac in zip(iterations, tac_values) if tac and tac < 1e10]
        if not valid_data:
            continue

        iterations, tac_values = zip(*valid_data)

        color = ALGO_COLORS.get(algo, '#8b949e')
        dash = 'solid' if status == 'running' else 'dot'

        fig.add_trace(go.Scatter(
            x=list(iterations),
            y=list(tac_values),
            mode='lines+markers',
            name=f"{case} ({algo})",
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=4),
            hovertemplate=f"<b>{case}</b><br>Iter: %{{x}}<br>TAC: $%{{y:,.0f}}<extra>{algo}</extra>"
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 32, 48, 0.6)',
        xaxis_title="Iteration",
        yaxis_title="Best TAC ($/year)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 32, 48, 0.8)'
        ),
        margin=dict(l=60, r=20, t=40, b=40),
        height=300,
        yaxis=dict(tickformat="$,.0f"),
    )

    st.plotly_chart(fig, width='stretch')


def render_3d_surface_plot(sweep_data: List[Dict], pressure_value: float = None):
    """Render a 3D surface plot of TAC as a function of NT and Feed."""
    if not sweep_data:
        st.info("No sweep data available for 3D visualization.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(sweep_data)

    # Filter by pressure if specified
    if pressure_value is not None and 'pressure' in df.columns:
        df = df[abs(df['pressure'] - pressure_value) < 0.01]

    if df.empty or 'nt' not in df.columns or 'feed' not in df.columns:
        st.info("Insufficient data for 3D surface plot.")
        return

    # Get unique NT and Feed values
    nt_values = sorted(df['nt'].unique())
    feed_values = sorted(df['feed'].unique())

    if len(nt_values) < 2 or len(feed_values) < 2:
        st.info("Need more data points for 3D surface.")
        return

    # Create TAC matrix
    tac_matrix = np.full((len(feed_values), len(nt_values)), np.nan)
    t_reb_matrix = np.full((len(feed_values), len(nt_values)), np.nan)

    for _, row in df.iterrows():
        try:
            nt_idx = nt_values.index(row['nt'])
            feed_idx = feed_values.index(row['feed'])
            tac = row.get('tac', row.get('TAC', np.nan))
            if tac and tac < 1e10:
                tac_matrix[feed_idx, nt_idx] = tac
                t_reb_matrix[feed_idx, nt_idx] = row.get('T_reb', np.nan)
        except (ValueError, KeyError):
            continue

    # Create 3D surface
    fig = go.Figure(data=[go.Surface(
        x=nt_values,
        y=feed_values,
        z=tac_matrix,
        colorscale='Viridis',
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        ),
        hovertemplate="NT: %{x}<br>Feed: %{y}<br>TAC: $%{z:,.0f}<extra></extra>"
    )])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis_title='Number of Trays (NT)',
            yaxis_title='Feed Stage',
            zaxis_title='TAC ($/year)',
            bgcolor='rgba(30, 32, 48, 0.6)',
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
    )

    st.plotly_chart(fig, width='stretch')


def render_heatmap(sweep_data: List[Dict], z_key: str = 'tac', pressure_value: float = None):
    """Render a 2D heatmap of NT vs Feed."""
    if not sweep_data:
        st.info("No data available for heatmap.")
        return

    df = pd.DataFrame(sweep_data)

    # Filter by pressure if specified
    if pressure_value is not None and 'pressure' in df.columns:
        df = df[abs(df['pressure'] - pressure_value) < 0.01]

    if df.empty or 'nt' not in df.columns or 'feed' not in df.columns:
        st.info("Insufficient data for heatmap.")
        return

    # Pivot for heatmap
    z_column = z_key if z_key in df.columns else 'tac' if 'tac' in df.columns else 'TAC'

    # Filter valid values
    df = df[df[z_column] < 1e10]

    if df.empty:
        st.info("No valid data points for heatmap.")
        return

    pivot = df.pivot_table(index='feed', columns='nt', values=z_column, aggfunc='mean')

    # Color scale based on metric
    if z_key in ['T_reb', 't_reb']:
        colorscale = 'RdYlBu_r'  # Red for hot, blue for cold
        title = 'Reboiler Temperature (°C)'
    else:
        colorscale = 'RdYlGn_r'  # Red for high TAC, green for low
        title = 'TAC ($/year)'

    fig = go.Figure(data=go.Heatmap(
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        z=pivot.values,
        colorscale=colorscale,
        hovertemplate='NT: %{x}<br>Feed: %{y}<br>Value: %{z:,.1f}<extra></extra>'
    ))

    # Add contour lines
    fig.update_traces(
        contours=dict(
            showlines=True,
            linewidth=1,
            linecolor='rgba(255,255,255,0.3)'
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 32, 48, 0.6)',
        xaxis_title='Number of Trays (NT)',
        yaxis_title='Feed Stage',
        coloraxis_colorbar_title=title,
        margin=dict(l=60, r=20, t=30, b=40),
        height=400,
    )

    st.plotly_chart(fig, width='stretch')


def render_pareto_front(points: List[Dict]):
    """Render Pareto front visualization for TAC vs Energy."""
    if not points:
        st.info("No data available for Pareto front.")
        return

    # Filter valid points
    valid = [p for p in points if p.get('tac') and p.get('q_reb')
             and p['tac'] < 1e10 and p['q_reb'] < 1e10]

    if len(valid) < 3:
        st.info("Need more data points for Pareto front visualization.")
        return

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(valid, 'tac', 'q_reb')

    fig = go.Figure()

    # All points
    fig.add_trace(go.Scatter(
        x=[p['tac'] for p in valid],
        y=[p['q_reb'] for p in valid],
        mode='markers',
        name='All Evaluations',
        marker=dict(size=6, color='rgba(139, 148, 158, 0.4)'),
        hovertemplate='TAC: $%{x:,.0f}<br>Q_reb: %{y:.1f} kW<extra></extra>'
    ))

    # Pareto frontier
    if pareto:
        fig.add_trace(go.Scatter(
            x=[p['tac'] for p in pareto],
            y=[p['q_reb'] for p in pareto],
            mode='lines+markers',
            name='Pareto Frontier',
            line=dict(color='#ef5350', width=3),
            marker=dict(size=10, color='#ef5350'),
            hovertemplate='<b>Pareto Optimal</b><br>TAC: $%{x:,.0f}<br>Q_reb: %{y:.1f} kW<extra></extra>'
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 32, 48, 0.6)',
        xaxis_title='Total Annual Cost ($/year)',
        yaxis_title='Reboiler Energy (kW)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(30, 32, 48, 0.8)'
        ),
        margin=dict(l=60, r=20, t=30, b=40),
        height=400,
        xaxis=dict(tickformat="$,.0f"),
    )

    st.plotly_chart(fig, width='stretch')


def render_run_timeline(results: List[Dict]):
    """Render a timeline of optimization runs."""
    if not results:
        st.info("No runs to display in timeline.")
        return

    # Parse timestamps and prepare data
    timeline_data = []
    for r in results:
        try:
            modified = r.get('modified', '')
            if modified:
                timestamp = datetime.fromisoformat(modified.replace('Z', '+00:00'))
            else:
                continue

            timeline_data.append({
                'timestamp': timestamp,
                'case': r.get('case_name', r.get('case', 'Unknown')),
                'algorithm': r.get('algorithm', 'ISO'),
                'tac': r.get('optimal_tac', 0),
                'filename': r.get('filename', ''),
            })
        except:
            continue

    if not timeline_data:
        st.info("No valid timestamps for timeline.")
        return

    df = pd.DataFrame(timeline_data)
    df = df.sort_values('timestamp')

    fig = go.Figure()

    for algo in ['ISO', 'PSO', 'GA']:
        algo_df = df[df['algorithm'] == algo]
        if algo_df.empty:
            continue

        fig.add_trace(go.Scatter(
            x=algo_df['timestamp'],
            y=[algo] * len(algo_df),
            mode='markers',
            name=algo,
            marker=dict(
                size=15,
                color=ALGO_COLORS.get(algo, '#8b949e'),
                symbol='circle',
            ),
            hovertemplate='<b>%{customdata[0]}</b><br>TAC: $%{customdata[1]:,.0f}<br>%{x}<extra>' + algo + '</extra>',
            customdata=list(zip(algo_df['case'], algo_df['tac'])),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 32, 48, 0.6)',
        xaxis_title='Time',
        yaxis_title='Algorithm',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=20, t=40, b=40),
        height=250,
    )

    st.plotly_chart(fig, width='stretch')


def render_algorithm_comparison(stats: Dict):
    """Render algorithm performance comparison charts."""
    if not stats:
        return

    by_algo = stats.get("by_algorithm", {})

    # Prepare data for comparison
    algos = []
    avg_tacs = []
    min_tacs = []
    avg_times = []
    counts = []

    for algo in ['ISO', 'PSO', 'GA']:
        algo_stats = by_algo.get(algo, {})
        if isinstance(algo_stats, dict) and algo_stats.get("count", 0) > 0:
            algos.append(algo)
            avg_tacs.append(algo_stats.get("avg_tac", 0))
            min_tacs.append(algo_stats.get("min_tac", 0))
            avg_times.append(algo_stats.get("avg_time", 0))
            counts.append(algo_stats.get("count", 0))

    if not algos:
        st.info("No algorithm data available for comparison.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # TAC comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=algos,
            y=avg_tacs,
            name='Average TAC',
            marker_color=[ALGO_COLORS.get(a, '#8b949e') for a in algos],
            text=[f'${t:,.0f}' if t else 'N/A' for t in avg_tacs],
            textposition='outside',
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30, 32, 48, 0.6)',
            title="Average TAC by Algorithm",
            yaxis_title="TAC ($/year)",
            yaxis=dict(tickformat="$,.0f"),
            height=300,
            margin=dict(l=60, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Time comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=algos,
            y=avg_times,
            name='Average Time',
            marker_color=[ALGO_COLORS.get(a, '#8b949e') for a in algos],
            text=[f'{t:.1f}s' if t else 'N/A' for t in avg_times],
            textposition='outside',
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30, 32, 48, 0.6)',
            title="Average Time by Algorithm",
            yaxis_title="Time (seconds)",
            height=300,
            margin=dict(l=60, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    # Run count pie chart
    fig = go.Figure(data=[go.Pie(
        labels=algos,
        values=counts,
        hole=0.4,
        marker=dict(colors=[ALGO_COLORS.get(a, '#8b949e') for a in algos]),
    )])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        title="Runs by Algorithm",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, width='stretch')


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

    # Get current algorithm for styling
    current_algo = st.session_state.get(f"{prefix}algorithm", "ISO")

    with st.container():
        # Styled header with algorithm color accent
        header_style = get_slot_header_style(current_algo)
        algo_badge = get_algorithm_badge(current_algo)
        status_badge = get_status_badge(job_status) if job_status else ''

        st.markdown(
            f'''<div style="{header_style}">
                <h3 style="margin: 0; display: inline-block; color: #e6edf3; font-weight: 700; font-size: 1.05rem;">Run Slot {slot_id + 1}</h3>
                {algo_badge}
                {status_badge}
            </div>''',
            unsafe_allow_html=True
        )
        st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)

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
            # Only update session state on actual change to avoid refresh
            if algorithm != st.session_state.get(f"{prefix}algorithm"):
                st.session_state[f"{prefix}algorithm"] = algorithm

        with col3:
            demo_mode = st.checkbox(
                "Demo Mode",
                value=st.session_state.get(f"{prefix}demo_mode", True),
                key=f"{prefix}demo_checkbox",
                disabled=job_status == "running",
                help="Run without Aspen"
            )
            # Only update session state on actual change to avoid refresh
            if demo_mode != st.session_state.get(f"{prefix}demo_mode"):
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
            # Status shown in header badge, show additional info here if running
            if job_status == "running":
                st.caption("Optimization in progress...")
            elif job_status == "done":
                st.caption("Completed successfully")
            elif job_status == "failed":
                st.caption("Run failed - check logs")
            elif job_status == "killed":
                st.caption("Stopped by user")

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

# Styled sidebar header
st.sidebar.markdown("""
<div style="
    text-align: center;
    padding: 1.25rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(79, 195, 247, 0.1);
">
    <div style="
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #4fc3f7, #81d4fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    ">&#9879;</div>
    <h2 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #e6edf3;">Column Optimizer</h2>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.75rem; color: #8b949e; letter-spacing: 0.08em; text-transform: uppercase;">Multi-Run Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Mode selection
mode = st.sidebar.radio(
    "Mode",
    ["Run Optimization", "Statistics", "Analysis", "Results Browser"],
    index=0
)
st.session_state.mode = mode

st.sidebar.markdown("---")

# API status indicator with enhanced styling
api_ok = check_api_health()
if api_ok:
    active_count = get_active_runs_count()
    st.sidebar.markdown(f"""
    <div style="
        background: rgba(76, 223, 138, 0.06);
        border: 1px solid rgba(76, 223, 138, 0.2);
        border-radius: 10px;
        padding: 0.85rem;
        margin: 0.5rem 0;
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #4cdf8a; font-size: 0.6rem; animation: pulse-glow 2s ease-in-out infinite;">&#9679;</span>
            <span style="font-weight: 600; color: #4cdf8a; font-size: 0.85rem;">API Connected</span>
        </div>
        <div style="font-size: 0.75rem; margin-top: 0.35rem; color: #8b949e;">
            Active runs: <span style="color: #e6edf3; font-weight: 600;">{active_count}/{MAX_CONCURRENT_RUNS}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="
        background: rgba(255, 183, 77, 0.06);
        border: 1px solid rgba(255, 183, 77, 0.2);
        border-radius: 10px;
        padding: 0.85rem;
        margin: 0.5rem 0;
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #ffb74d; font-size: 1rem;">&#9888;</span>
            <span style="font-weight: 600; color: #ffb74d; font-size: 0.85rem;">API Offline</span>
        </div>
        <div style="font-size: 0.72rem; margin-top: 0.35rem; color: #8b949e;">
            Run: <code style="background: rgba(255,255,255,0.06); padding: 0.15rem 0.4rem; border-radius: 4px; color: #c9d1d9;">uvicorn server:app</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# Config info styled
st.sidebar.markdown(f"""
<div style="
    background: rgba(79, 195, 247, 0.04);
    border: 1px solid rgba(79, 195, 247, 0.1);
    border-radius: 8px;
    padding: 0.6rem 0.85rem;
    font-size: 0.78rem;
">
    <span style="color: #8b949e;">T_reb limit:</span>
    <span style="font-weight: 700; color: #e6edf3; font-variant-numeric: tabular-nums;">{T_REBOILER_MAX}&#176;C</span>
</div>
""", unsafe_allow_html=True)

# Quick actions with styled header
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
">
    &#9881; Quick Actions
</div>
""", unsafe_allow_html=True)

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

# Export section
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
">
    &#128229; Export
</div>
""", unsafe_allow_html=True)

# CSV Export button
if st.sidebar.button("Download Results (CSV)"):
    try:
        resp = requests.get(f"{API_BASE_URL}/export/results/csv", timeout=30)
        if resp.status_code == 200:
            st.sidebar.download_button(
                label="Save CSV",
                data=resp.content,
                file_name="optimization_results.csv",
                mime="text/csv",
                key="csv_download"
            )
        else:
            st.sidebar.error("Failed to export CSV")
    except Exception as e:
        st.sidebar.error(f"Export error: {e}")

# ZIP Plots button
if st.sidebar.button("Download Plots (ZIP)"):
    try:
        resp = requests.get(f"{API_BASE_URL}/export/plots/zip", timeout=60)
        if resp.status_code == 200:
            st.sidebar.download_button(
                label="Save ZIP",
                data=resp.content,
                file_name=f"plots_{datetime.now().strftime('%Y%m%d')}.zip",
                mime="application/zip",
                key="zip_download"
            )
        else:
            st.sidebar.error("No plots found or export failed")
    except Exception as e:
        st.sidebar.error(f"Export error: {e}")

# Discord Notifications section
st.sidebar.markdown("---")
with st.sidebar.expander("Discord Notifications", expanded=False):
    st.caption("Get notified when runs complete")

    # Initialize webhook URL in session state
    if "discord_webhook" not in st.session_state:
        st.session_state["discord_webhook"] = ""

    webhook_url = st.text_input(
        "Webhook URL",
        value=st.session_state.get("discord_webhook", ""),
        type="password",
        placeholder="https://discord.com/api/webhooks/...",
        key="discord_webhook_input"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save", key="save_webhook"):
            if webhook_url:
                if set_discord_webhook(webhook_url):
                    st.session_state["discord_webhook"] = webhook_url
                    st.success("Saved!")
                else:
                    st.error("Failed to save")
            else:
                set_discord_webhook("")
                st.session_state["discord_webhook"] = ""
                st.info("Cleared")

    with col2:
        if st.button("Test", key="test_webhook"):
            result = test_discord_webhook(webhook_url or st.session_state.get("discord_webhook"))
            if result.get("success"):
                st.success("Sent!")
            else:
                st.error(result.get("message", "Failed"))

    st.caption("""
    **Setup:**
    1. Open Discord Server Settings
    2. Go to Integrations → Webhooks
    3. Create New Webhook
    4. Copy the URL and paste above
    """)


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════

# Styled main header
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 30%, #0d3150 60%, #0f1419 100%);
    background-size: 200% 200%;
    animation: gradient-shift 8s ease infinite;
    color: #e6edf3;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(79, 195, 247, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
">
    <div style="
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 20% 50%, rgba(79, 195, 247, 0.08) 0%, transparent 60%);
        pointer-events: none;
    "></div>
    <h1 style="margin: 0; font-size: 1.7rem; font-weight: 800; position: relative; letter-spacing: -0.02em;">
        &#9879; Distillation Column Optimization
    </h1>
    <p style="margin: 0.6rem 0 0 0; color: #8b949e; font-size: 0.95rem; position: relative;">
        Multi-algorithm optimizer &mdash;
        <span style="color: #4fc3f7;">ISO</span> &bull;
        <span style="color: #ffb74d;">PSO</span> &bull;
        <span style="color: #66bb6a;">GA</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Get available cases
available_cases = list_available_cases()
if not available_cases:
    available_cases = ["Case1_COL2", "Case1_COL3", "Case8_COL2", "Case9_COL2"]

if mode == "Run Optimization":
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    ">
        <span style="font-size: 1.3rem; color: #4fc3f7;">&#9654;</span>
        <h2 style="margin: 0; color: #e6edf3; font-weight: 700; letter-spacing: -0.01em;">Multi-Run Optimization</h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Configure and run up to 4 optimizations concurrently. Each run has isolated configuration (multi-run safe).")

    # Check for any running jobs
    any_running = any(
        st.session_state.get(f"run_{slot}_job_status") == "running"
        for slot in range(MAX_CONCURRENT_RUNS)
    )

    # Combined Convergence Chart (if any jobs have data)
    if any_running or any(st.session_state.get(f"run_{slot}_job_id") for slot in range(MAX_CONCURRENT_RUNS)):
        render_combined_convergence_chart()

    # Render run slots in a 2x2 grid
    col_left, col_right = st.columns(2)
    with col_left:
        render_run_slot(0, available_cases, api_ok)
        render_run_slot(2, available_cases, api_ok)
    with col_right:
        render_run_slot(1, available_cases, api_ok)
        render_run_slot(3, available_cases, api_ok)

    # Detailed Logs Panel
    st.markdown("---")
    with st.expander("Detailed Logs", expanded=False):
        log_tabs = st.tabs([f"Run {i+1}" for i in range(MAX_CONCURRENT_RUNS)])

        for slot, tab in enumerate(log_tabs):
            with tab:
                job_id = st.session_state.get(f"run_{slot}_job_id")
                if job_id:
                    filter_text = st.text_input("Filter", key=f"log_filter_{slot}", placeholder="Search logs...")
                    # Fetch logs without filter (avoids API call on every keystroke)
                    log_data = get_logs(job_id, tail=200)
                    if log_data and log_data.get("logs"):
                        logs = log_data["logs"]
                        # Apply filter client-side
                        if filter_text:
                            logs = [line for line in logs if filter_text.lower() in line.lower()]
                        st.caption(f"Showing {len(logs)} of {log_data['total_lines']} lines")
                        log_text = "\n".join(logs)
                        st.code(log_text, language="text")
                    else:
                        st.caption("No logs available")
                else:
                    st.caption("No job running in this slot")

    # Auto-refresh while jobs are running (5 second interval)
    if any_running:
        time.sleep(5)
        st.rerun()


elif mode == "Statistics":
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    ">
        <span style="font-size: 1.3rem; color: #66bb6a;">&#128202;</span>
        <h2 style="margin: 0; color: #e6edf3; font-weight: 700; letter-spacing: -0.01em;">Statistics Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Aggregated metrics and performance analysis across all historical runs.")

    # Fetch statistics
    stats = get_statistics()

    if not stats or stats.get("total_runs", 0) == 0:
        st.info("No historical runs found. Run some optimizations first!")
    else:
        # Top metrics row
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Total Runs", stats.get("total_runs", 0))
        with m2:
            best_tac = stats.get("best_tac_ever")
            st.metric("Best TAC Ever", f"${best_tac:,.0f}" if best_tac else "N/A")
        with m3:
            st.metric("Total Evaluations", f"{stats.get('total_evaluations', 0):,}")
        with m4:
            total_time = stats.get("total_time_seconds", 0)
            st.metric("Total Compute Time", f"{total_time/3600:.1f} hrs" if total_time > 3600 else f"{total_time:.0f}s")

        # Best run info
        best_run = stats.get("best_run")
        if best_run:
            st.markdown("---")
            st.markdown(f"""
            <div style="
                background: rgba(79, 195, 247, 0.08);
                border: 1px solid rgba(79, 195, 247, 0.2);
                border-radius: 10px;
                padding: 1rem;
                margin: 0.5rem 0;
            ">
                <div style="font-weight: 600; color: #4fc3f7; margin-bottom: 0.5rem;">Best Run</div>
                <div style="color: #c9d1d9;">
                    <strong>{best_run.get('case', 'Unknown')}</strong> using <strong>{best_run.get('algorithm', 'ISO')}</strong><br>
                    TAC: <strong>${stats.get('best_tac_ever', 0):,.0f}/year</strong><br>
                    Config: NT={best_run.get('optimal', {}).get('nt', '?')}, Feed={best_run.get('optimal', {}).get('feed', '?')}, P={best_run.get('optimal', {}).get('pressure', 0):.3f} bar
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Algorithm comparison
        st.subheader("Algorithm Performance Comparison")
        render_algorithm_comparison(stats)

        # Run history timeline
        st.markdown("---")
        st.subheader("Run History Timeline")
        results = fetch_results_list()
        render_run_timeline(results)


elif mode == "Analysis":
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    ">
        <span style="font-size: 1.3rem; color: #e040fb;">&#128300;</span>
        <h2 style="margin: 0; color: #e6edf3; font-weight: 700; letter-spacing: -0.01em;">Analysis Tools</h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption("3D surface plots, sensitivity heatmaps, and Pareto front analysis.")

    # Get result folders with sweep data
    result_folders = sorted(glob.glob(str(RESULTS_DIR / "Case*")), key=os.path.getmtime, reverse=True)

    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
        "3D Surface Plot", "Sensitivity Heatmap", "Pareto Front", "Run Comparison"
    ])

    with analysis_tab1:
        st.subheader("TAC Surface: f(NT, Feed)")

        if result_folders:
            selected_folder = st.selectbox("Select Result", result_folders,
                format_func=lambda x: os.path.basename(x))

            if selected_folder:
                sweep_data = load_sweep_data(selected_folder)
                if sweep_data:
                    # Check if there are multiple pressure values
                    pressures = list(set(p.get('pressure') for p in sweep_data if p.get('pressure')))
                    if len(pressures) > 1:
                        selected_pressure = st.select_slider("Pressure (bar)", options=sorted(pressures))
                    else:
                        selected_pressure = pressures[0] if pressures else None

                    render_3d_surface_plot(sweep_data, selected_pressure)
                else:
                    st.info("No sweep data found in this result folder.")
        else:
            st.info("No result folders found. Run an ISO optimization to generate sweep data.")

    with analysis_tab2:
        st.subheader("Parameter Sensitivity Heatmap")

        if result_folders:
            selected_folder2 = st.selectbox("Select Result", result_folders,
                format_func=lambda x: os.path.basename(x), key="heatmap_folder")

            if selected_folder2:
                sweep_data = load_sweep_data(selected_folder2)
                if sweep_data:
                    metric = st.radio("Metric", ["TAC", "T_reb (Temperature)"], horizontal=True)
                    z_key = 'tac' if metric == "TAC" else 'T_reb'
                    render_heatmap(sweep_data, z_key=z_key)
                else:
                    st.info("No sweep data found in this result folder.")
        else:
            st.info("No result folders found.")

    with analysis_tab3:
        st.subheader("Pareto Front: TAC vs Energy")

        if result_folders:
            selected_folder3 = st.selectbox("Select Result", result_folders,
                format_func=lambda x: os.path.basename(x), key="pareto_folder")

            if selected_folder3:
                sweep_data = load_sweep_data(selected_folder3)
                if sweep_data:
                    # Check if we have q_reb data
                    has_energy = any(p.get('q_reb') or p.get('Q_reb') for p in sweep_data)
                    if has_energy:
                        # Normalize key names
                        for p in sweep_data:
                            if 'Q_reb' in p and 'q_reb' not in p:
                                p['q_reb'] = p['Q_reb']
                            if 'TAC' in p and 'tac' not in p:
                                p['tac'] = p['TAC']
                        render_pareto_front(sweep_data)
                    else:
                        st.info("Energy data (Q_reb) not available in this sweep. Run with full evaluation to get energy data.")
                else:
                    st.info("No sweep data found in this result folder.")
        else:
            st.info("No result folders found.")

    with analysis_tab4:
        st.subheader("Historical Run Comparison")

        results = fetch_results_list()
        if results:
            # Filter out config files
            results = [r for r in results if "run_config" not in r.get("filename", "")]

            if len(results) >= 2:
                selected_runs = st.multiselect(
                    "Select runs to compare (up to 3)",
                    options=[r.get("filename") for r in results],
                    max_selections=3,
                    default=[results[0].get("filename")] if results else []
                )

                if selected_runs:
                    cols = st.columns(len(selected_runs))

                    # Find best TAC among selected for comparison
                    selected_data = [r for r in results if r.get("filename") in selected_runs]
                    best_tac = min(r.get("optimal_tac", float('inf')) for r in selected_data if r.get("optimal_tac"))

                    for i, (run_name, col) in enumerate(zip(selected_runs, cols)):
                        run_data = next((r for r in results if r.get("filename") == run_name), {})
                        with col:
                            st.markdown(f"**{run_name[:30]}...**" if len(run_name) > 30 else f"**{run_name}**")
                            tac = run_data.get("optimal_tac", 0)
                            delta = tac - best_tac if i > 0 and tac and best_tac else None
                            st.metric("TAC", f"${tac:,.0f}" if tac else "N/A",
                                     delta=f"+${delta:,.0f}" if delta else None,
                                     delta_color="inverse")
                            st.write(f"**Case:** {run_data.get('case_name', 'N/A')}")
                            st.write(f"**Algorithm:** {run_data.get('algorithm', 'ISO')}")
                            st.write(f"**NT:** {run_data.get('optimal_nt', 'N/A')}")
                            st.write(f"**Feed:** {run_data.get('optimal_feed', 'N/A')}")
                            st.write(f"**Pressure:** {run_data.get('optimal_pressure', 'N/A')}")
                            st.write(f"**Time:** {run_data.get('time_seconds', 0):.1f}s")
            else:
                st.info("Need at least 2 results to compare.")
        else:
            st.info("No results found.")


elif mode == "Results Browser":
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    ">
        <span style="font-size: 1.3rem; color: #ffb74d;">&#128193;</span>
        <h2 style="margin: 0; color: #e6edf3; font-weight: 700; letter-spacing: -0.01em;">Results Browser</h2>
    </div>
    """, unsafe_allow_html=True)

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
                st.image(plot, caption=os.path.basename(plot), width='stretch')
    else:
        st.info("No plots found in results folder.")


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style="
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    padding: 1.2rem;
    background: rgba(30, 32, 48, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    font-size: 0.75rem;
    color: #8b949e;
">
    <span><span style="color: #4fc3f7;">&#9679;</span> <strong style="color: #c9d1d9;">API:</strong> {API_BASE_URL}</span>
    <span><span style="color: #66bb6a;">&#9679;</span> <strong style="color: #c9d1d9;">Results:</strong> {RESULTS_DIR}</span>
    <span><span style="color: #ffb74d;">&#9679;</span> <strong style="color: #c9d1d9;">Max Concurrent:</strong> {MAX_CONCURRENT_RUNS}</span>
</div>
""", unsafe_allow_html=True)