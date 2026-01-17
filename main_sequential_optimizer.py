# main_sequential_optimizer.py
"""
Main Entry Point for TRUE Iterative Sequential Optimization (ISO)
=================================================================

WORKFLOW:
=========
PHASE 1: ISO Optimization (Professor's Method)
    - Variables optimized ONE AT A TIME: P → NT → NF
    - Outer iteration loop until convergence
    - Temperature constraint: T_reb ≤ 120°C

PHASE 2: Post-ISO Parametric Sweep (For Multiple U-Curves)
    - At optimal pressure P*, sweep all NT-Feed combinations
    - Generates data for publication-quality multiple U-curves

Author: PSE Lab, NTUST
Version: 5.2 - ISO + Multiple U-Curves
"""

import logging
import sys
import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

# Initialize COM for Windows subprocess compatibility
try:
    import pythoncom
    pythoncom.CoInitialize()
except ImportError:
    pass  # Not on Windows or pythoncom not available

# ════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir="results"):
    """Setup logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"iso_opt_{timestamp}.log")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    return log_filename, timestamp

# Initialize logging
log_file, run_timestamp = setup_logging()
logger = logging.getLogger(__name__)

# Import modules
from config import get_case_config, list_available_cases, create_run_output_dir
from aspen_interface import AspenEnergyOptimizer
from tac_calculator import TACCalculator
from tac_evaluator import TACEvaluator
from iso_optimizer import ISOOptimizer, T_REBOILER_MAX
from visualization_iso import ISOVisualizer


# ════════════════════════════════════════════════════════════════════════════
# POST-ISO PARAMETRIC SWEEP FOR MULTIPLE U-CURVES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SweepPoint:
    """Data point from NT-Feed sweep."""
    nt: int
    feed: int
    pressure: float
    tac: float
    converged: bool
    T_reb: float = 0.0


def run_nt_feed_sweep(evaluator, optimal_pressure: float, 
                      nt_bounds: tuple, feed_bounds: tuple,
                      optimal_nt: int = None, optimal_feed: int = None,
                      nt_step: int = 2, feed_step: int = 2,
                      min_section_stages: int = 3,
                      nt_range_around_opt: int = 20,
                      feed_range_around_opt: int = 10) -> List[Dict]:
    """
    Run NT-Feed parametric sweep AROUND the optimal point.
    
    This generates data for multiple U-curves visualization,
    focused on the region around the ISO optimal for cleaner plots.
    
    Parameters
    ----------
    evaluator : TACEvaluator
        The TAC evaluator with Aspen connection
    optimal_pressure : float
        Fixed pressure from ISO optimization (bar)
    nt_bounds : tuple
        (min_nt, max_nt) absolute bounds for number of stages
    feed_bounds : tuple
        (min_feed, max_feed) absolute bounds for feed stage
    optimal_nt : int
        Optimal NT from ISO (sweep will center around this)
    optimal_feed : int
        Optimal feed from ISO (sweep will center around this)
    nt_step : int
        Step size for NT sweep
    feed_step : int
        Step size for feed sweep
    min_section_stages : int
        Minimum stages in rectifying/stripping sections
    nt_range_around_opt : int
        How many stages above/below optimal NT to sweep (default: ±20)
    feed_range_around_opt : int
        How many stages above/below optimal feed to sweep (default: ±10)
        
    Returns
    -------
    List[Dict] : List of sweep results with keys: nt, feed, tac, pressure
    """
    # Calculate focused ranges around optimal
    if optimal_nt is not None:
        nt_min = max(nt_bounds[0], optimal_nt - nt_range_around_opt)
        nt_max = min(nt_bounds[1], optimal_nt + nt_range_around_opt)
    else:
        nt_min, nt_max = nt_bounds
    
    if optimal_feed is not None:
        feed_min = max(feed_bounds[0], optimal_feed - feed_range_around_opt)
        feed_max = min(feed_bounds[1], optimal_feed + feed_range_around_opt)
    else:
        feed_min, feed_max = feed_bounds
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("POST-ISO PHASE: NT-FEED PARAMETRIC SWEEP")
    logger.info("=" * 70)
    logger.info(f"  Purpose: Generate data for multiple U-curves")
    logger.info(f"  Fixed pressure: {optimal_pressure:.4f} bar (from ISO)")
    logger.info("")
    logger.info(f"  Optimal from ISO: NT={optimal_nt}, NF={optimal_feed}")
    logger.info(f"  Sweep range: NT ± {nt_range_around_opt}, Feed ± {feed_range_around_opt}")
    logger.info("")
    logger.info(f"  NT range: {nt_min} to {nt_max}, step={nt_step}")
    logger.info(f"  Feed range: {feed_min} to {feed_max}, step={feed_step}")
    logger.info("")
    
    results = []
    total_points = 0
    feasible_points = 0
    
    # Generate NT values (focused around optimal)
    nt_values = list(range(nt_min, nt_max + 1, nt_step))
    
    # Make sure optimal NT is included
    if optimal_nt and optimal_nt not in nt_values:
        nt_values.append(optimal_nt)
        nt_values.sort()
    
    # Count total expected points for progress
    for nt in nt_values:
        valid_feed_min = max(feed_min, min_section_stages + 1)
        valid_feed_max = min(feed_max, nt - min_section_stages)
        if valid_feed_max >= valid_feed_min:
            n_feeds = len(range(valid_feed_min, valid_feed_max + 1, feed_step))
            # Add 1 if optimal_feed not in range but should be included
            if optimal_feed and valid_feed_min <= optimal_feed <= valid_feed_max:
                if optimal_feed not in range(valid_feed_min, valid_feed_max + 1, feed_step):
                    n_feeds += 1
            total_points += n_feeds
    
    logger.info(f"  Expected evaluations: ~{total_points}")
    logger.info("-" * 70)
    
    eval_count = 0
    start_time = time.time()
    
    for nt in nt_values:
        # Determine valid feed range for this NT (using focused range)
        valid_feed_min = max(feed_min, min_section_stages + 1)
        valid_feed_max = min(feed_max, nt - min_section_stages)
        
        if valid_feed_max < valid_feed_min:
            continue
        
        feed_values = list(range(valid_feed_min, valid_feed_max + 1, feed_step))
        
        # Make sure optimal feed is included
        if optimal_feed and valid_feed_min <= optimal_feed <= valid_feed_max:
            if optimal_feed not in feed_values:
                feed_values.append(optimal_feed)
                feed_values.sort()
        
        for feed in feed_values:
            eval_count += 1
            
            # Progress update every 10 evaluations
            if eval_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = eval_count / elapsed if elapsed > 0 else 0
                remaining = (total_points - eval_count) / rate if rate > 0 else 0
                logger.info(f"  Progress: {eval_count}/{total_points} "
                           f"({100*eval_count/total_points:.0f}%) - "
                           f"ETA: {remaining/60:.1f} min")
            
            # Evaluate this point
            try:
                result = evaluator.evaluate(
                    nt=nt, 
                    feed=feed, 
                    pressure=optimal_pressure
                )
                
                # evaluator.evaluate() returns a dict, not a tuple
                tac = result.get('TAC', 1e12)
                converged = result.get('converged', False)
                T_reb = result.get('T_reb', 0)
                
                if converged and tac < 1e10:
                    results.append({
                        'nt': nt,
                        'feed': feed,
                        'pressure': optimal_pressure,
                        'tac': tac,
                        'T_reb': T_reb if T_reb else 0,
                        'converged': True
                    })
                    feasible_points += 1
                else:
                    results.append({
                        'nt': nt,
                        'feed': feed,
                        'pressure': optimal_pressure,
                        'tac': 1e12,
                        'T_reb': T_reb if T_reb else 0,
                        'converged': False
                    })
                    
            except Exception as e:
                logger.warning(f"  Error at NT={nt}, NF={feed}: {e}")
                results.append({
                    'nt': nt,
                    'feed': feed,
                    'pressure': optimal_pressure,
                    'tac': 1e12,
                    'T_reb': 0,
                    'converged': False
                })
    
    elapsed = time.time() - start_time
    
    logger.info("-" * 70)
    logger.info(f"  Sweep complete!")
    logger.info(f"  Total evaluations: {eval_count}")
    logger.info(f"  Feasible points: {feasible_points}")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)
    
    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_iso_optimization(case_name: str, config_overrides: dict = None,
                         run_post_sweep: bool = True,
                         sweep_nt_step: int = 2,
                         sweep_feed_step: int = 2,
                         nt_range_around_opt: int = 20,
                         feed_range_around_opt: int = 10):
    """
    Run TRUE Iterative Sequential Optimization with Multiple U-Curves.
    
    Parameters
    ----------
    case_name : str
        Case identifier (e.g., 'Case1_COL2')
    config_overrides : dict, optional
        Override default configuration values
    run_post_sweep : bool
        Whether to run post-ISO NT-Feed sweep for U-curves (default: True)
    sweep_nt_step : int
        NT step size for post-ISO sweep (default: 2)
    sweep_feed_step : int
        Feed step size for post-ISO sweep (default: 2)
    nt_range_around_opt : int
        Sweep NT within ± this many stages of optimal (default: 20)
    feed_range_around_opt : int
        Sweep Feed within ± this many stages of optimal (default: 10)
        
    Returns
    -------
    dict : Optimization results
    """
    # Get configuration
    config = get_case_config(case_name)
    if config is None:
        logger.error(f"Unknown case: {case_name}")
        return None

    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if key in config['bounds']:
                config['bounds'][key] = value
            else:
                config[key] = value

    # Create run-specific output directory
    run_output_dir = create_run_output_dir(case_name)
    logger.info(f"Output directory: {run_output_dir}")

    # Print header
    print("")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "  TRUE ITERATIVE SEQUENTIAL OPTIMIZATION (ISO)".center(68) + "|")
    print("|" + "  + Multiple U-Curves Generation".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")
    print("")
    print(f"  Case: {case_name}")
    print(f"  Column: {config['column']['block_name']} ({config['column']['description']})")
    print("")
    print(f"  Temperature constraint: T_reb <= {T_REBOILER_MAX}C")
    print("")
    print("  Bounds:")
    print(f"    NT: {config['bounds']['nt_bounds']}")
    print(f"    Feed: {config['bounds']['feed_bounds']}")
    print(f"    Pressure: {config['bounds']['pressure_bounds']} bar")
    print("")
    print(f"  Post-ISO sweep for U-curves: {'YES' if run_post_sweep else 'NO'}")
    if run_post_sweep:
        print(f"    Sweep range: NT ± {nt_range_around_opt}, Feed ± {feed_range_around_opt}")
    print(f"  Log file: {log_file}")
    print("=" * 70)
    print("")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONNECT TO ASPEN
    # ─────────────────────────────────────────────────────────────────────────
    
    logger.info("Connecting to Aspen Plus...")
    
    aspen = AspenEnergyOptimizer(config['file_path'])
    
    if not aspen.connect_and_open(visible=True):
        logger.error("Failed to connect to Aspen Plus!")
        return None
    
    logger.info("Connected successfully!")
    aspen.get_column_info(config['column']['block_name'])
    
    # ─────────────────────────────────────────────────────────────────────────
    # SETUP TAC CALCULATOR AND EVALUATOR
    # ─────────────────────────────────────────────────────────────────────────
    
    tac_calc = TACCalculator(
        material='SS',
        cepci=800,
        cepci_base=500,
        operating_hours=8000,
        payback_period=3,
    )
    
    evaluator = TACEvaluator(
        aspen_interface=aspen,
        tac_calculator=tac_calc,
        block_name=config['column']['block_name'],
        feed_stream=config['column']['feed_stream'],
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # CREATE ISO OPTIMIZER
    # ─────────────────────────────────────────────────────────────────────────
    
    iso_config = {
        'bounds': config['bounds'],
        'initial': config.get('initial', {}),
        
        # Sweep settings
        'pressure_points': 9,
        'nt_step': 2,
        'feed_step': 2,
        'min_section_stages': 3,
        
        # Temperature constraint (professor's requirement)
        'T_reb_max': T_REBOILER_MAX,
        
        # Convergence settings
        'tac_tolerance': 500,  # $/year
        'max_iterations': 10,
    }
    
    optimizer = ISOOptimizer(evaluator, iso_config)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: RUN ISO OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    nt_feed_sweep_results = None
    
    try:
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 1: ISO OPTIMIZATION")
        logger.info("=" * 70)
        logger.info("  Variables optimized ONE AT A TIME: P -> NT -> NF")
        logger.info(f"  Temperature constraint: T_reb <= {T_REBOILER_MAX}C")
        
        result = optimizer.run(case_name=case_name)
        
        # ─────────────────────────────────────────────────────────────────────
        # PHASE 2: POST-ISO NT-FEED SWEEP FOR MULTIPLE U-CURVES
        # ─────────────────────────────────────────────────────────────────────
        
        if run_post_sweep and result:
            nt_feed_sweep_results = run_nt_feed_sweep(
                evaluator=evaluator,
                optimal_pressure=result.optimal_pressure,
                nt_bounds=config['bounds']['nt_bounds'],
                feed_bounds=config['bounds']['feed_bounds'],
                optimal_nt=result.optimal_nt,
                optimal_feed=result.optimal_feed,
                nt_step=sweep_nt_step,
                feed_step=sweep_feed_step,
                min_section_stages=3,
                nt_range_around_opt=nt_range_around_opt,
                feed_range_around_opt=feed_range_around_opt
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # GENERATE PLOTS
        # ─────────────────────────────────────────────────────────────────────
        
        logger.info("")
        logger.info("Generating plots...")

        visualizer = ISOVisualizer(output_dir=run_output_dir)

        # Get optimal dict for visualization
        optimal = None
        if result:
            optimal = {
                'nt': result.optimal_nt,
                'feed': result.optimal_feed,
                'pressure': result.optimal_pressure,
                'tac': result.optimal_tac,
            }
        
        # Generate all plots including multiple U-curves
        plot_files = visualizer.plot_all(
            optimizer, 
            case_name,
            nt_feed_results=nt_feed_sweep_results,  # Pass sweep data for U-curves
            n_ucurves=7  # Number of U-curves to display
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # SAVE RESULTS
        # ─────────────────────────────────────────────────────────────────────
        
        logger.info("")
        logger.info("Saving results...")
        
        results_file = optimizer.save_results(output_dir=run_output_dir)

        # Save sweep data separately if generated
        if nt_feed_sweep_results:
            import json
            sweep_file = os.path.join(run_output_dir, f"{case_name}_NT_Feed_Sweep.json")
            with open(sweep_file, 'w') as f:
                json.dump(nt_feed_sweep_results, f, indent=2)
            logger.info(f"  Saved sweep data: {sweep_file}")
        
        # ─────────────────────────────────────────────────────────────────────
        # PRINT FINAL SUMMARY
        # ─────────────────────────────────────────────────────────────────────
        
        print("")
        print("+" + "=" * 68 + "+")
        print("|" + "  ISO OPTIMIZATION COMPLETE".center(68) + "|")
        print("+" + "=" * 68 + "+")
        print("")
        print("  OPTIMAL CONFIGURATION:")
        print("  " + "-" * 40)
        print(f"    Number of Stages (NT): {result.optimal_nt}")
        print(f"    Feed Stage (NF):       {result.optimal_feed}")
        print(f"    Operating Pressure:    {result.optimal_pressure:.4f} bar")
        print("")
        print(f"    TAC: ${result.optimal_tac:,.0f}/year")
        print("")
        print("  " + "-" * 40)
        print(f"    Converged: {'YES' if result.converged else 'NO'}")
        print(f"    Iterations: {result.convergence_iteration}")
        print(f"    Time: {result.total_time_seconds:.1f}s ({result.total_time_seconds/60:.1f} min)")
        print(f"    Evaluations: {result.total_evaluations}")
        print(f"      - Feasible: {result.feasible_evaluations}")
        print(f"      - Infeasible (T_reb>{T_REBOILER_MAX}C): {result.infeasible_evaluations}")
        print("")
        if nt_feed_sweep_results:
            feasible_sweep = len([r for r in nt_feed_sweep_results if r['tac'] < 1e10])
            print(f"  POST-ISO SWEEP (for U-curves):")
            print(f"    Points evaluated: {len(nt_feed_sweep_results)}")
            print(f"    Feasible points: {feasible_sweep}")
        print("")
        print("  OUTPUT FOLDER:")
        print(f"    {run_output_dir}")
        print("")
        print("  OUTPUT FILES:")
        print(f"    Results: {os.path.basename(results_file)}")
        print(f"    Log: {os.path.basename(log_file)}")
        print(f"    Plots: {len(plot_files)} files")
        for pf in plot_files:
            print(f"      - {os.path.basename(pf)}")
        print("")
        print("=" * 70)

        # Copy log file to run output directory
        import shutil
        log_copy = os.path.join(run_output_dir, os.path.basename(log_file))
        try:
            shutil.copy2(log_file, log_copy)
        except Exception:
            pass  # Log file copy is best-effort

        return {
            'case_name': case_name,
            'output_dir': run_output_dir,
            'optimal': {
                'nt': result.optimal_nt,
                'feed': result.optimal_feed,
                'pressure': result.optimal_pressure,
                'tac': result.optimal_tac,
            },
            'converged': result.converged,
            'iterations': result.convergence_iteration,
            'time_seconds': result.total_time_seconds,
            'evaluations': result.total_evaluations,
            'plots': plot_files,
            'results_file': results_file,
            'nt_feed_sweep': nt_feed_sweep_results,
        }
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        try:
            aspen.close()
        except:
            pass


# ════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ════════════════════════════════════════════════════════════════════════════

def print_header():
    """Print program header."""
    print("")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "  TRUE ITERATIVE SEQUENTIAL OPTIMIZATION (ISO)".center(68) + "|")
    print("|" + "  FOR DISTILLATION COLUMNS".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("|" + "  + Multiple U-Curves Generation".center(68) + "|")
    print("|" + "  PSE Lab - NTUST".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")
    print("")


def select_case():
    """Step 1: Select Case."""
    print("")
    print("=" * 60)
    print("  STEP 1: SELECT DISTILLATION SEQUENCE")
    print("=" * 60)
    print("")
    print("    1. Case 1  - Direct Sequence")
    print("    2. Case 8  - Indirect Sequence")
    print("    3. Case 9  - Mixed Sequence A")
    print("    4. Case 11 - Mixed Sequence B")
    print("    5. Case 12 - Optimal Energy Sequence")
    print("    Q. Quit")
    print("")
    
    choice = input("  Enter choice [1-5 or Q]: ").strip().upper()
    
    case_map = {'1': 'Case1', '2': 'Case8', '3': 'Case9', '4': 'Case11', '5': 'Case12'}
    
    if choice == 'Q':
        return None
    return case_map.get(choice, None) or select_case()


def select_column():
    """Step 2: Select Column."""
    print("")
    print("=" * 60)
    print("  STEP 2: SELECT COLUMN TO OPTIMIZE")
    print("=" * 60)
    print("")
    print("    2. COL2 - EB/SM Separation (Main)")
    print("    3. COL3 - Benzene/Toluene Separation")
    print("    4. COL4 - Light Ends Removal")
    print("    5. COL5 - Heavy Ends Separation")
    print("    B. Back")
    print("    Q. Quit")
    print("")
    
    choice = input("  Enter choice [2-5, B, or Q]: ").strip().upper()
    
    col_map = {'2': 'COL2', '3': 'COL3', '4': 'COL4', '5': 'COL5'}
    
    if choice == 'Q':
        return 'QUIT'
    if choice == 'B':
        return None
    return col_map.get(choice, None) or select_column()


def select_sweep_option():
    """Step 3: Select whether to run post-ISO sweep and range."""
    print("")
    print("=" * 60)
    print("  STEP 3: POST-ISO SWEEP FOR MULTIPLE U-CURVES")
    print("=" * 60)
    print("")
    print("    The post-ISO sweep generates data for multiple U-curves")
    print("    by evaluating NT-Feed combinations around the optimal point.")
    print("")
    print("    1. YES - Full sweep (NT ± 20, Feed ± 10) - Recommended")
    print("    2. YES - Focused sweep (NT ± 10, Feed ± 5) - Faster")
    print("    3. YES - Wide sweep (NT ± 30, Feed ± 15) - More data")
    print("    4. NO  - Skip sweep (ISO plots only)")
    print("")
    
    choice = input("  Select option [1-4]: ").strip()
    
    if choice == '1':
        return True, 20, 10  # run_sweep, nt_range, feed_range
    elif choice == '2':
        return True, 10, 5
    elif choice == '3':
        return True, 30, 15
    else:
        return False, 0, 0


def interactive_menu():
    """Interactive menu for case selection."""
    print_header()
    
    print("METHODOLOGY: TRUE Iterative Sequential Optimization")
    print("-" * 60)
    print("  PHASE 1 - ISO:")
    print("    Step 1: Optimize P (T_reb <= 120C constraint)")
    print("    Step 2: Optimize NT (at P*)")
    print("    Step 3: Optimize NF (at P*, NT*)")
    print("    -> Repeat until converged")
    print("")
    print("  PHASE 2 - Post-ISO Sweep:")
    print("    Sweep NT-Feed combinations AROUND optimal point")
    print("    -> Generates multiple U-curves for thesis")
    print("-" * 60)
    print("")
    
    while True:
        case_name = select_case()
        if case_name is None:
            print("\n  Exiting...")
            sys.exit(0)
        
        column_name = select_column()
        if column_name == 'QUIT':
            print("\n  Exiting...")
            sys.exit(0)
        elif column_name is None:
            continue
        
        run_sweep, nt_range, feed_range = select_sweep_option()
        
        full_case_name = f"{case_name}_{column_name}"
        
        print("")
        print(f"  Selected: {full_case_name}")
        if run_sweep:
            print(f"  Post-ISO sweep: YES (NT ± {nt_range}, Feed ± {feed_range})")
        else:
            print(f"  Post-ISO sweep: NO")
        confirm = input("  Proceed? [Y/n]: ").strip().upper()
        
        if confirm in ['', 'Y', 'YES']:
            run_iso_optimization(
                full_case_name,
                run_post_sweep=run_sweep,
                sweep_nt_step=2,
                sweep_feed_step=2,
                nt_range_around_opt=nt_range,
                feed_range_around_opt=feed_range
            )
            
            another = input("\n  Run another? [y/N]: ").strip().upper()
            if another not in ['Y', 'YES']:
                break


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    # Debug: Print immediately to confirm script started
    print("=" * 60, flush=True)
    print("ISO OPTIMIZER STARTING...", flush=True)
    print(f"Arguments: {sys.argv}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print("=" * 60, flush=True)

    if len(sys.argv) > 1:
        case_name = sys.argv[1]
        print(f"Case name: {case_name}", flush=True)
        # Check for --no-sweep flag
        run_sweep = '--no-sweep' not in sys.argv
        print_header()
        result = run_iso_optimization(case_name, run_post_sweep=run_sweep)

        # Exit with proper code
        if result is None:
            logger.error("Optimization failed!")
            sys.exit(1)
        else:
            logger.info("Optimization completed successfully!")
            sys.exit(0)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()