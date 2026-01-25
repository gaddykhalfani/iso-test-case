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
    ["Run Optimization", "Results Browser", "Demo Gallery"],
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
                st.image(plot, caption=os.path.basename(plot), use_container_width=True)
    else:
        st.info("No plots found in results folder.")


elif mode == "Demo Gallery":
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    ">
        <span style="font-size: 1.3rem; color: #66bb6a;">&#127912;</span>
        <h2 style="margin: 0; color: #e6edf3; font-weight: 700; letter-spacing: -0.01em;">Demo Gallery</h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Pre-generated results and plots for demonstration.")

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