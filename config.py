# config.py
"""
Configuration Module for Sequential Parametric Optimizer
=========================================================
Defines column configurations, optimization bounds, and case settings.

Author: PSE Lab, NTUST
Version: 4.0 - Sequential Parametric Approach
"""

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# ════════════════════════════════════════════════════════════════════════════
# BASE DIRECTORY CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

BASE_DIR = r"C:/Users/pse/Desktop/test/Fix yang ini/iso test case/iso-test-case/"

# ════════════════════════════════════════════════════════════════════════════
# SEQUENCE FILES - Map sequence cases to their Aspen files
# ════════════════════════════════════════════════════════════════════════════

SEQUENCE_FILES = {
    'Case1': os.path.join(BASE_DIR, "case1.apw"),
    'Case8': os.path.join(BASE_DIR, "case8.apw"),
    'Case9': os.path.join(BASE_DIR, "case9.apw"),
    'Case11': os.path.join(BASE_DIR, "case11.apw"),
    'Case12': os.path.join(BASE_DIR, "case12.apw"),
}

# ════════════════════════════════════════════════════════════════════════════
# COLUMN TEMPLATES - Define column properties for each position
# ════════════════════════════════════════════════════════════════════════════

COLUMN_TEMPLATES = {
    'COL2': {
        'block_name': 'COL2',
        'feed_stream': 'LIQPROD1',
        'description': 'EB/SM Separation',
        'nt_bounds': (15, 80),
        'feed_bounds': (10, 70),
        'pressure_bounds': (0.1, 1.0),
        'initial_nt': 25,
        'initial_feed': 15,
        'initial_pressure': 0.2,
    },
    'COL3': {
        'block_name': 'COL3',
        'feed_stream': 'LIQPROD2',
        'description': 'Benzene/Toluene Separation',
        'nt_bounds': (15, 45),
        'feed_bounds': (5, 30),
        'pressure_bounds': (0.1, 0.2),
        'initial_nt': 45,
        'initial_feed': 23,
        'initial_pressure': 0.2,
    },
    'COL4': {
        'block_name': 'COL4',
        'feed_stream': 'LIQPROD',
        'description': 'Light Ends Removal',
        'nt_bounds': (15, 85),
        'feed_bounds': (10, 80),
        'pressure_bounds': (0.01, 0.1),
        'initial_nt': 95,
        'initial_feed': 40,
        'initial_pressure': 0.1,
    },
    'COL5': {
        'block_name': 'COL5',
        'feed_stream': 'HEAVY1',
        'description': 'Heavy Ends Separation',
        'nt_bounds': (15, 45),
        'feed_bounds': (5, 40),
        'pressure_bounds': (0.5, 1.5),
        'initial_nt': 25,
        'initial_feed': 12,
        'initial_pressure': 1.0,
    },
}

# ════════════════════════════════════════════════════════════════════════════
# PURITY SPECIFICATIONS - Stream/Component mapping for each Case+Column
# ════════════════════════════════════════════════════════════════════════════

PURITY_SPECS = {
    # Case 1 - Direct Sequence
    'Case1_COL2': {
        'stream': 'TOLL',
        'component': 'TOLUE-01',
        'fraction_type': 'MASSFRAC',
        'target': 0.99,
    },
    'Case1_COL3': {
        'stream': 'EB',
        'component': 'ETHYL-01',
        'fraction_type': 'MASSFRAC',
        'target': 0.99,
    },
    'Case1_COL4': {
        'stream': 'STY',
        'component': 'STYRE-01',
        'fraction_type': 'MASSFRAC',
        'target': 0.999,
    },
    'Case1_COL5': {
        'stream': 'AMS',
        'component': 'ALPHA-01',
        'fraction_type': 'MASSFRAC',
        'target': 0.999,
    },

    # Case 8 - Indirect Sequence (EB before STY)
    'Case8_COL2': {
        'stream': 'EBTOL',  # Top product (middle split - no purity spec)
        'component': None,   # Middle split has no single key component
        'fraction_type': 'MASSFRAC',
        'target': None,      # No purity target for middle split
        'is_middle_split': True,
        'bottom_stream': 'STYAMS',
        'top_components': ['TOLUE-01', 'ETHYL-01', 'STYRE-01'],  # Lights go UP
        'bottom_components': ['ALPHA-01', 'STY-DI', 'STY-TRI', 'DPP'],  # Heavies go DOWN
    },
    'Case8_COL3': {
        'stream': 'LIQPROD2',
        'component': 'TOLUE-01',
        'fraction_type': 'MASSFRAC',
        'target': 0.99,
    },
    'Case8_COL4': {
        'stream': 'EBB',
        'component': 'STYRE-01',
        'fraction_type': 'MASSFRAC',
        'target': 0.997,
        # Alternative spec for EB recovery:
        'alt_stream': 'TOLL',
        'alt_component': 'ETHYL-01',
        'alt_target': 0.99,
    },
    'Case8_COL5': {
        'stream': 'AMSS',
        'component': 'ALPHA-01',
        'fraction_type': 'MASSFRAC',
        'target': 0.999,
    },

    # Case 9 - Similar to Case 8
    'Case9_COL2': {
        'stream': 'EBTOL',
        'component': None,
        'fraction_type': 'MASSFRAC',
        'target': None,
        'is_middle_split': True,
        'bottom_stream': 'STYAMS',
        'top_components': ['TOLUE-01', 'ETHYL-01', 'STYRE-01'],  # Lights go UP
        'bottom_components': ['ALPHA-01', 'STY-DI', 'STY-TRI', 'DPP'],  # Heavies go DOWN
    },
    'Case9_COL3': {
        'stream': 'LIQPROD2',
        'component': 'TOLUE-01',
        'fraction_type': 'MASSFRAC',
        'target': 0.99,
    },
    'Case9_COL4': {
        'stream': 'EBB',
        'component': 'STYRE-01',
        'fraction_type': 'MASSFRAC',
        'target': 0.997,
    },
    'Case9_COL5': {
        'stream': 'AMSS',
        'component': 'ALPHA-01',
        'fraction_type': 'MASSFRAC',
        'target': 0.999,
    },
}


def get_purity_spec(case_name: str) -> Optional[Dict]:
    """
    Get purity specification for a case.

    Parameters
    ----------
    case_name : str
        Case identifier (e.g., 'Case1_COL2', 'Case8_COL4')

    Returns
    -------
    dict : Purity spec with keys: stream, component, fraction_type, target
           Returns None if case not found
    """
    return PURITY_SPECS.get(case_name)


# ════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SequentialOptConfig:
    """Configuration for Sequential Parametric Optimization."""
    pressure_sweep_points: int = 9
    pressure_refine: bool = True
    pressure_refine_tol: float = 0.005
    pressure_sweep_use_midpoint: bool = True
    nt_step: int = 2
    feed_step: int = 2
    min_rectifying_stages: int = 3
    min_stripping_stages: int = 3
    output_dir: str = "results"
    generate_ucurves: bool = True
    save_pressure_sweep: bool = True
    save_nt_feed_sweep: bool = True


@dataclass  
class ColumnConfig:
    """Configuration for a specific column."""
    block_name: str
    feed_stream: str
    description: str
    nt_bounds: Tuple[int, int]
    feed_bounds: Tuple[int, int]
    pressure_bounds: Tuple[float, float]
    initial_nt: int
    initial_feed: int
    initial_pressure: float


@dataclass
class TACConfig:
    """Configuration for TAC calculations."""
    material: str = 'SS'
    tray_spacing: float = 0.6096
    payback_period: int = 3
    cepci: float = 800
    cepci_base: float = 500
    operating_hours: int = 8000


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def get_case_config(case_name: str) -> Optional[Dict]:
    """
    Get complete configuration for a case.
    
    Parameters
    ----------
    case_name : str
        Case identifier in format 'CaseX_COLY' (e.g., 'Case1_COL2')
        
    Returns
    -------
    dict : Complete configuration dictionary or None if invalid
    """
    parts = case_name.split('_')
    if len(parts) != 2:
        print(f"Invalid case name format: {case_name}")
        print("Expected format: CaseX_COLY (e.g., Case1_COL2)")
        return None
    
    sequence_name = parts[0]
    column_name = parts[1]
    
    if sequence_name not in SEQUENCE_FILES:
        print(f"Unknown sequence: {sequence_name}")
        print(f"Available: {list(SEQUENCE_FILES.keys())}")
        return None
    
    if column_name not in COLUMN_TEMPLATES:
        print(f"Unknown column: {column_name}")
        print(f"Available: {list(COLUMN_TEMPLATES.keys())}")
        return None
    
    file_path = SEQUENCE_FILES[sequence_name]
    col_template = COLUMN_TEMPLATES[column_name]
    
    if not os.path.exists(file_path):
        print(f"WARNING: Aspen file not found: {file_path}")
    
    config = {
        'case_name': case_name,
        'sequence': sequence_name,
        'column_name': column_name,
        'file_path': file_path,
        'column': {
            'block_name': col_template['block_name'],
            'feed_stream': col_template['feed_stream'],
            'description': col_template['description'],
        },
        'bounds': {
            'nt_bounds': col_template['nt_bounds'],
            'feed_bounds': col_template['feed_bounds'],
            'pressure_bounds': col_template['pressure_bounds'],
        },
        'initial': {
            'nt': col_template['initial_nt'],
            'feed': col_template['initial_feed'],
            'pressure': col_template['initial_pressure'],
        },
        'iso': {
            'nt_bounds': col_template['nt_bounds'],
            'feed_bounds': col_template['feed_bounds'],
            'pressure_bounds': col_template['pressure_bounds'],
        },
    }
    
    return config


def list_available_cases() -> list:
    """List all available case configurations."""
    cases = []
    for seq in SEQUENCE_FILES.keys():
        for col in COLUMN_TEMPLATES.keys():
            cases.append(f"{seq}_{col}")
    return cases


def get_column_config(column_name: str) -> Optional[ColumnConfig]:
    """Get ColumnConfig dataclass for a column."""
    if column_name not in COLUMN_TEMPLATES:
        return None
    
    t = COLUMN_TEMPLATES[column_name]
    return ColumnConfig(
        block_name=t['block_name'],
        feed_stream=t['feed_stream'],
        description=t['description'],
        nt_bounds=t['nt_bounds'],
        feed_bounds=t['feed_bounds'],
        pressure_bounds=t['pressure_bounds'],
        initial_nt=t['initial_nt'],
        initial_feed=t['initial_feed'],
        initial_pressure=t['initial_pressure'],
    )


# ════════════════════════════════════════════════════════════════════════════
# RUN-SPECIFIC CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

def create_run_config(case_name: str, overrides: Optional[Dict] = None) -> Optional[Dict]:
    """
    Create a run-specific configuration by combining base config with overrides.

    This function creates an ISOLATED copy of the configuration that is safe
    for concurrent multi-run scenarios. The original COLUMN_TEMPLATES remain
    unmodified.

    Parameters
    ----------
    case_name : str
        Case identifier (e.g., 'Case1_COL2')
    overrides : dict, optional
        Override values for bounds and initial values. Expected structure:
        {
            'nt_bounds': (min, max),
            'feed_bounds': (min, max),
            'pressure_bounds': (min, max),
            'initial_nt': int,
            'initial_feed': int,
            'initial_pressure': float,
        }

    Returns
    -------
    dict : Run-specific configuration (deep copy with overrides applied)
           Returns None if case_name is invalid
    """
    import copy

    # Get base config (this already creates a new dict)
    base_config = get_case_config(case_name)
    if base_config is None:
        return None

    # Deep copy to ensure complete isolation
    run_config = copy.deepcopy(base_config)

    # Apply overrides if provided
    if overrides:
        # Bounds overrides
        if 'nt_bounds' in overrides:
            run_config['bounds']['nt_bounds'] = tuple(overrides['nt_bounds'])
            run_config['iso']['nt_bounds'] = tuple(overrides['nt_bounds'])
        if 'feed_bounds' in overrides:
            run_config['bounds']['feed_bounds'] = tuple(overrides['feed_bounds'])
            run_config['iso']['feed_bounds'] = tuple(overrides['feed_bounds'])
        if 'pressure_bounds' in overrides:
            run_config['bounds']['pressure_bounds'] = tuple(overrides['pressure_bounds'])
            run_config['iso']['pressure_bounds'] = tuple(overrides['pressure_bounds'])

        # Initial values overrides
        if 'initial_nt' in overrides:
            run_config['initial']['nt'] = overrides['initial_nt']
        if 'initial_feed' in overrides:
            run_config['initial']['feed'] = overrides['initial_feed']
        if 'initial_pressure' in overrides:
            run_config['initial']['pressure'] = overrides['initial_pressure']

    return run_config


def save_run_config(run_config: Dict, run_id: str, base_dir: str = "results") -> str:
    """
    Save a run-specific configuration to a JSON file.

    Parameters
    ----------
    run_config : dict
        The run-specific configuration to save
    run_id : str
        Unique identifier for this run (e.g., job_id)
    base_dir : str
        Base directory for results (default: 'results')

    Returns
    -------
    str : Path to the saved config file
    """
    import json

    os.makedirs(base_dir, exist_ok=True)
    config_path = os.path.join(base_dir, f"run_config_{run_id}.json")

    # Convert tuples to lists for JSON serialization
    def serialize_config(obj):
        if isinstance(obj, dict):
            return {k: serialize_config(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    serializable_config = serialize_config(run_config)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_config, f, indent=2)

    return config_path


def load_run_config(config_path: str) -> Optional[Dict]:
    """
    Load a run-specific configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the config JSON file

    Returns
    -------
    dict : Loaded configuration with tuples restored, or None if failed
    """
    import json

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Convert lists back to tuples for bounds
        def restore_tuples(obj, tuple_keys={'nt_bounds', 'feed_bounds', 'pressure_bounds'}):
            if isinstance(obj, dict):
                return {
                    k: tuple(v) if k in tuple_keys and isinstance(v, list)
                    else restore_tuples(v, tuple_keys)
                    for k, v in obj.items()
                }
            return obj

        return restore_tuples(config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def get_column_template(column_name: str) -> Optional[Dict]:
    """
    Get a copy of the column template (for dashboard display).

    Returns a COPY to prevent accidental modification of global templates.

    Parameters
    ----------
    column_name : str
        Column identifier (e.g., 'COL2')

    Returns
    -------
    dict : Copy of column template or None if not found
    """
    import copy

    if column_name not in COLUMN_TEMPLATES:
        return None

    return copy.deepcopy(COLUMN_TEMPLATES[column_name])


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY HELPER
# ════════════════════════════════════════════════════════════════════════════

def create_run_output_dir(case_name: str, base_dir: str = "results") -> str:
    """
    Create a timestamped output directory for a specific run.

    Parameters
    ----------
    case_name : str
        Case identifier (e.g., 'Case1_COL2')
    base_dir : str
        Base results directory (default: 'results')

    Returns
    -------
    str : Path to the created directory (e.g., 'results/Case1_COL2_20260118_120000/')
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{case_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


# ════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATIONS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_SEQ_OPT_CONFIG = SequentialOptConfig()
DEFAULT_TAC_CONFIG = TACConfig()


# ════════════════════════════════════════════════════════════════════════════
# TESTING
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("CONFIGURATION MODULE - Sequential Parametric Optimizer")
    print("=" * 70)
    
    print("\nAvailable Cases:")
    for case in list_available_cases():
        print(f"  - {case}")
    
    print("\n" + "-" * 70)
    print("Example: Case1_COL2")
    print("-" * 70)
    
    config = get_case_config("Case1_COL2")
    if config:
        print(f"\nSequence: {config['sequence']}")
        print(f"Column: {config['column']['block_name']} ({config['column']['description']})")
        print(f"File: {config['file_path']}")
        print(f"\nBounds:")
        print(f"  NT: {config['bounds']['nt_bounds']}")
        print(f"  Feed: {config['bounds']['feed_bounds']}")
        print(f"  Pressure: {config['bounds']['pressure_bounds']} bar")