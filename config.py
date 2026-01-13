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

BASE_DIR = r"C:/Users/pse/Desktop/test/Fix yang ini/iso test case"

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
        'feed_stream': 'EBTOL',
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
        'feed_stream': 'STYRENE',
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
        'feed_stream': 'LIQPROD4',
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