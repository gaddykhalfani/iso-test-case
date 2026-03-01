# tac_evaluator_seider.py
"""
TAC Evaluator with Seider Vacuum System Model
==============================================
Updated evaluator to use TACCalculator v3.1 with Seider correlations.

Changes from previous version:
-----------------------------
1. Removed reflux drum sizing (not needed)
2. Uses Seider air leakage correlation (Eq. 22.73)
3. Uses Table 22.32 vacuum equipment costs
4. Returns vacuum_operating_cost instead of vacuum_steam_cost

Author: PSE Lab, NTUST
Date: December 2024
"""

import logging

# Import Seider-based calculator
from tac_calculator import TACCalculator

logger = logging.getLogger(__name__)


class TACEvaluator:
    """
    Evaluates TAC for column configurations with intelligent caching.
    
    Uses Seider vacuum system correlations for accurate costing.
    """
    
    def __init__(self, aspen_interface, tac_calculator, block_name, feed_stream):
        """
        Initialize TAC evaluator.
        
        Parameters:
        -----------
        aspen_interface : AspenEnergyOptimizer
            Interface to Aspen Plus COM
        tac_calculator : TACCalculator
            Calculator for TAC computation (v3.1 Seider version)
        block_name : str
            Name of RadFrac block (e.g., 'COL2')
        feed_stream : str
            Name of feed stream (e.g., 'LIQPROD1')
        """
        self.aspen = aspen_interface
        self.tac_calc = tac_calculator
        self.block_name = block_name
        self.feed_stream = feed_stream
        
        # Cache: key = (NT, Feed, P_rounded), value = result dict
        self.cache = {}
        self.cache_hits = 0
        self.eval_count = 0
        
        # Pressure rounding precision for cache key
        self.p_precision = 3
        
        # Track failed evaluations
        self.failed_count = 0

        # Store original BASIS_RR for restoration after RR recovery
        # This prevents contamination of subsequent evaluations
        self.original_basis_rr = self.aspen.get_basis_rr(block_name)
        if self.original_basis_rr is not None:
            logger.info("  Stored original BASIS_RR = {:.3f}".format(self.original_basis_rr))
        else:
            logger.warning("  Could not read original BASIS_RR - will use default (2.0) for restoration")
            self.original_basis_rr = 2.0  # Reasonable default

        self.column_diag = None

    def _make_key(self, nt, feed, pressure):
        """Create cache key from parameters, including block name for isolation."""
        p_rounded = round(pressure, self.p_precision)
        return (self.block_name, int(nt), int(feed), p_rounded)
    
    def _check_aspen_convergence(self):
        """Check if Aspen simulation actually converged."""
        try:
            # Check block status
            try:
                block_path = r"\Data\Blocks\{}\Output\BLKSTAT".format(self.block_name)
                node = self.aspen.aspen.Tree.FindNode(block_path)
                if node:
                    status = node.Value
                    if status == 0 or (isinstance(status, str) and 'OK' in status.upper()):
                        return True, None
                    else:
                        return False, "Block status: {}".format(status)
            except:
                pass
            
            # Check run status
            try:
                conv_path = r"\Data\Results Summary\Run-Status\Output\UTEFLAG"
                node = self.aspen.aspen.Tree.FindNode(conv_path)
                if node:
                    if node.Value == 0:
                        return True, None
                    else:
                        return False, "Run status error flag: {}".format(node.Value)
            except:
                pass
            
            # Try to read results
            try:
                Q_reb, Q_cond = self.aspen.get_energy_results(self.block_name)
                if Q_reb is not None and Q_cond is not None and Q_reb > 0 and Q_cond > 0:
                    return True, None
                else:
                    return False, "Cannot read valid energy results"
            except Exception as e:
                return False, "Error reading results: {}".format(e)
            
        except Exception as e:
            return False, "Convergence check error: {}".format(e)
    
    def _get_temperatures(self, nt):
        """Extract condenser and reboiler temperatures from Aspen."""
        try:
            base_path = r"\Data\Blocks\{}\Output".format(self.block_name)

            # 1. Get actual number of stages first (source of truth)
            actual_nt = nt
            try:
                nstage_path = base_path + r"\NSTAGE"
                node = self.aspen.aspen.Tree.FindNode(nstage_path)
                if node and node.Value:
                    actual_nt = int(node.Value)
                    if actual_nt != nt:
                        logger.warning(f"  [Temp] NSTAGE mismatch: requested NT={nt}, Aspen output NSTAGE={actual_nt}")
            except:
                logger.warning(f"  [Temp] Could not read NSTAGE from output, using NT={nt}")

            # 2. Get Condenser Temperature (Stage 1)
            T_cond = None
            cond_path_used = None
            cond_paths = [
                base_path + r"\STAGE_TEMP\1",
                base_path + r"\B_TEMP\1",
                base_path + r"\COND_TEMP",
            ]
            for path in cond_paths:
                try:
                    node = self.aspen.aspen.Tree.FindNode(path)
                    if node and node.Value is not None:
                        T_cond = node.Value
                        cond_path_used = path
                        break
                except:
                    pass

            # 3. Get Reboiler Temperature (Stage = actual_nt)
            T_reb = None
            reb_path_used = None
            reb_paths = [
                base_path + r"\STAGE_TEMP\{}".format(actual_nt),
                base_path + r"\B_TEMP\{}".format(actual_nt),
                base_path + r"\REB_TEMP",
                base_path + r"\BOTTOM_TEMP",
            ]
            for path in reb_paths:
                try:
                    node = self.aspen.aspen.Tree.FindNode(path)
                    if node and node.Value is not None:
                        T_reb = node.Value
                        reb_path_used = path
                        break
                except:
                    pass

            # 4. Validation & Sanity Checks
            if T_cond is not None and T_reb is not None:
                # Basic physics check: Reboiler must be hotter than condenser
                if T_reb < T_cond:
                    logger.warning(f"  [Temp] INVALID: T_reb ({T_reb:.1f}C) < T_cond ({T_cond:.1f}C) - Physical impossibility!")
                    logger.warning(f"  [Temp]   T_cond path: {cond_path_used}")
                    logger.warning(f"  [Temp]   T_reb path: {reb_path_used}")
                    logger.warning(f"  [Temp]   NSTAGE={actual_nt} (requested NT={nt})")
                    return None, None

                # Check for "default" or "stale" 0.0 values often returned by failed runs
                if T_reb < 0.1 or T_cond < 0.1:
                    logger.warning(f"  [Temp] INVALID: Zero temperatures detected: T_cond={T_cond}, T_reb={T_reb}")
                    return None, None

                logger.info(f"  [Temp] T_cond={T_cond:.1f}C, T_reb={T_reb:.1f}C (NSTAGE={actual_nt})")
                return T_cond, T_reb
            else:
                logger.warning(f"  [Temp] Extraction FAILED: T_cond={T_cond}, T_reb={T_reb} (NSTAGE={actual_nt})")
                if T_cond is None:
                    logger.warning(f"  [Temp]   Could not find condenser temp in any path")
                if T_reb is None:
                    logger.warning(f"  [Temp]   Could not find reboiler temp in any path")
                return None, None

        except Exception as e:
            logger.warning("  [Temp] Exception extracting temperatures: {}".format(e))
            return None, None
    
    def evaluate(self, nt, feed, pressure, run_diagnostic_on_fail=False,
                 rr_sweep_on_fail=False, rr_sweep_range=(0.5, 4.0), rr_sweep_points=10,
                 purity_spec=None, rr_hint=None):
        """
        Evaluate TAC for given configuration.

        Parameters:
        -----------
        nt : int
            Number of stages
        feed : int
            Feed stage location
        pressure : float
            Column pressure (bar)
        run_diagnostic_on_fail : bool
            If True, run diagnostic when simulation fails to determine
            if the failure is due to unachievable purity spec (default: False)
        rr_sweep_on_fail : bool
            If True, run full RR sweep when simulation fails to generate
            RR vs Purity curve showing achievability (default: False)
            Note: This is slower (10+ simulations per failed point)
        rr_sweep_range : tuple
            (min_rr, max_rr) range for RR sweep (default: 0.5 to 4.0)
        rr_sweep_points : int
            Number of points in RR sweep (default: 10)
        purity_spec : dict, optional
            Purity specification from config.PURITY_SPECS containing:
            - 'stream': Product stream name (e.g., 'TOLL', 'EBB')
            - 'component': Key component (e.g., 'STYRE-01', 'TOLUE-01')
            - 'fraction_type': 'MASSFRAC' or 'MOLEFRAC'
            - 'target': Target purity value

        Returns:
        --------
        dict : Result dictionary with TAC and all cost components
               If run_diagnostic_on_fail=True and simulation fails, includes
               'diagnostic' key with diagnostic results.
               If rr_sweep_on_fail=True, includes 'rr_sweep' key with sweep data.
        """
        self.eval_count += 1
        
        # Check cache first
        key = self._make_key(nt, feed, pressure)
        if key in self.cache:
            self.cache_hits += 1
            cached = self.cache[key]
            if cached.get('converged', False):
                logger.info("  Cache hit! TAC = ${:,.0f}".format(cached['TAC']))
            else:
                logger.info("  Cache hit (failed simulation)")
            return cached
        
        # Set column parameters in Aspen
        success = self.aspen.set_parameters(
            self.block_name, nt, feed, pressure, self.feed_stream
        )
        
        if not success:
            logger.warning("  Failed to set parameters")
            self.failed_count += 1
            result = self._failed_result(reason="Failed to set parameters")
            self.cache[key] = result
            return result
        
        # Apply RR warm-start hint if provided (for sweep continuity)
        if rr_hint is not None:
            self.aspen.set_reflux_ratio(self.block_name, rr_hint)
            logger.debug(f"  RR warm-start: BASIS_RR={rr_hint:.4f}")

        # Run Aspen simulation (with retry on COM crash)
        max_retries = 2
        success = False
        for attempt in range(max_retries + 1):
            success = self.aspen.run_simulation()
            if success:
                break
            # Check if it was a COM crash (connection lost)
            if not self.aspen.is_connected():
                if attempt < max_retries:
                    logger.warning(f"  COM crash on attempt {attempt+1}, reconnecting...")
                    if not self.aspen.reconnect():
                        logger.error("  Reconnection failed")
                        break
                    # Re-set parameters after reconnection
                    self.aspen.set_parameters(self.block_name, nt, feed, pressure, self.feed_stream)
                else:
                    logger.error(f"  COM crash on attempt {attempt+1}, max retries reached")
            else:
                break  # Normal simulation failure, no retry

        if not success:
            logger.warning("  Simulation failed to run")
            self.failed_count += 1
            if rr_hint is not None:
                self.aspen.set_reflux_ratio(self.block_name, self.original_basis_rr)
            result = self._failed_result(reason="Simulation failed to run")
            self.cache[key] = result
            return result
        
        # Check convergence
        converged, error_msg = self._check_aspen_convergence()
        
        if not converged:
            logger.warning("  NOT CONVERGED: {}".format(error_msg))
            self.failed_count += 1

            # Run diagnostic if enabled to determine WHY it failed
            diagnostic_result = None
            if run_diagnostic_on_fail:
                try:
                    diagnostic_result = self.aspen.run_diagnostic(
                        self.block_name, nt, feed, pressure, self.feed_stream,
                        purity_spec=purity_spec
                    )
                    if diagnostic_result and diagnostic_result.get('converged_without_spec'):
                        logger.info("  +-- DIAGNOSTIC RESULT ----------------------")
                        if diagnostic_result.get('achieved_purity') is not None:
                            logger.info("  | Purity achievable: {:.4f}".format(
                                diagnostic_result['achieved_purity']))
                        if diagnostic_result.get('target_purity') is not None:
                            logger.info("  | Target purity: {:.4f}".format(
                                diagnostic_result['target_purity']))
                        if diagnostic_result.get('natural_rr') is not None:
                            logger.info("  | Natural RR: {:.2f}".format(
                                diagnostic_result['natural_rr']))
                        if (diagnostic_result.get('achieved_purity') is not None and
                            diagnostic_result.get('target_purity') is not None):
                            if diagnostic_result['achieved_purity'] < diagnostic_result['target_purity']:
                                logger.info("  | >>> SPEC UNACHIEVABLE at this config")
                            else:
                                logger.info("  | >>> Spec achievable (solver difficulty)")
                        logger.info("  +-------------------------------------------")
                except Exception as diag_error:
                    logger.warning("  Diagnostic failed: {}".format(diag_error))

            # Run RR sweep if enabled (generates full RR vs Purity curve)
            rr_sweep_result = None
            if rr_sweep_on_fail:
                try:
                    logger.info("  Running RR sweep diagnostic...")
                    rr_sweep_result = self.aspen.sweep_rr_purity(
                        self.block_name, nt, feed, pressure, self.feed_stream,
                        rr_range=rr_sweep_range, num_points=rr_sweep_points,
                        purity_spec=purity_spec,
                        original_rr=self.original_basis_rr,
                        column_diag=getattr(self, 'column_diag', None)
                    )
                except Exception as sweep_error:
                    logger.warning("  RR sweep failed: {}".format(sweep_error))

            # ════════════════════════════════════════════════════════════════
            # RR RECOVERY: If RR sweep found successful point, retry with that RR
            # ════════════════════════════════════════════════════════════════
            if rr_sweep_result and purity_spec:
                # Handle None for middle split columns (Case8_COL2, Case9_COL2)
                target_purity = purity_spec.get('target') or 0.99

                # Find minimum RR that achieves target purity
                successful_rr = None
                for point in rr_sweep_result:
                    if (point.get('converged') and
                        point.get('purity') is not None and
                        point['purity'] >= target_purity):
                        successful_rr = point['rr']
                        logger.info("  [RR Recovery] Found feasible point: RR={:.2f}, purity={:.4f}".format(
                            point['rr'], point['purity']))
                        break  # Take first (lowest RR) that works

                if successful_rr:
                    # Attempt recovery with the successful RR
                    recovery_result = self._evaluate_with_fixed_rr(
                        nt, feed, pressure, successful_rr, purity_spec
                    )

                    if recovery_result and recovery_result.get('converged'):
                        # Recovery successful! Return this result to optimizer
                        recovery_result['rr_sweep'] = rr_sweep_result  # Keep sweep data
                        if diagnostic_result:
                            recovery_result['diagnostic'] = diagnostic_result
                        self.aspen.set_reflux_ratio(self.block_name, self.original_basis_rr)
                        self.cache[key] = recovery_result
                        logger.info("  [RR Recovery] Point recovered successfully!")
                        return recovery_result
                    else:
                        logger.info("  [RR Recovery] Recovery attempt failed, returning as infeasible")

            # No recovery possible - return failed result
            # ALWAYS restore RR to prevent contamination of subsequent evaluations
            self.aspen.set_reflux_ratio(self.block_name, self.original_basis_rr)
            result = self._failed_result(reason=error_msg)
            if diagnostic_result:
                result['diagnostic'] = diagnostic_result
            if rr_sweep_result:
                result['rr_sweep'] = rr_sweep_result
            self.cache[key] = result
            return result
        
        # ════════════════════════════════════════════════════════════════════
        # GET RESULTS FROM ASPEN
        # ════════════════════════════════════════════════════════════════════
        
        Q_reb, Q_cond = self.aspen.get_energy_results(self.block_name)
        diameter = self.aspen.get_diameter(self.block_name)
        T_cond, T_reb = self._get_temperatures(nt)
        
        # Basic check
        if Q_reb is None or Q_cond is None:
            logger.warning("  Cannot read energy results")
            self.failed_count += 1
            self.aspen.set_reflux_ratio(self.block_name, self.original_basis_rr)
            result = self._failed_result(reason="Cannot read energy results")
            self.cache[key] = result
            return result
        
        # ════════════════════════════════════════════════════════════════════
        # CALCULATE TAC (Seider vacuum model)
        # ════════════════════════════════════════════════════════════════════
        try:
            tac_result = self.tac_calc.calculate(
                nt=nt,
                diameter=diameter,
                Q_reb=Q_reb,
                Q_cond=Q_cond,
                pressure=pressure,
                T_cond=T_cond,
                T_reb=T_reb
            )
        except Exception as e:
            logger.warning("  TAC calculation failed: {}".format(e))
            self.failed_count += 1
            self.aspen.set_reflux_ratio(self.block_name, self.original_basis_rr)
            result = self._failed_result(reason="TAC calculation error: {}".format(e))
            self.cache[key] = result
            return result
        
        # ════════════════════════════════════════════════════════════════════
        # BUILD RESULT DICTIONARY
        # ════════════════════════════════════════════════════════════════════
        
        vacuum_sys = tac_result.get('vacuum_system', {})
        
        result = {
            'TAC': tac_result['TAC'],
            'TPC': tac_result['TPC'],
            'TOC': tac_result['TOC'],
            'Q_reb': Q_reb,
            'Q_cond': Q_cond,
            'Q_total': Q_reb + Q_cond,
            'diameter': diameter,
            'NT': nt,
            'feed': feed,
            'pressure': pressure,
            'converged': True,
            
            # Cost breakdown
            'column_cost': tac_result.get('column_cost', 0),
            'tray_cost': tac_result.get('tray_cost', 0),
            'condenser_cost': tac_result.get('condenser_cost', 0),
            'reboiler_cost': tac_result.get('reboiler_cost', 0),
            'vacuum_system_cost': tac_result.get('vacuum_system_cost', 0),
            'steam_cost': tac_result.get('steam_cost', 0),
            'cw_cost': tac_result.get('cw_cost', 0),
            'vacuum_operating_cost': tac_result.get('vacuum_operating_cost', 0),
            
            # For backward compatibility
            'vacuum_steam_cost': tac_result.get('vacuum_operating_cost', 0),
            
            # Temperature and utility info
            'T_cond': T_cond,
            'T_reb': T_reb,
            'condenser_utility': tac_result.get('condenser', {}).get('utility', 'N/A'),
            'reboiler_utility': tac_result.get('reboiler', {}).get('utility', 'N/A'),
            'condenser_LMTD': tac_result.get('condenser', {}).get('LMTD', 0),
            'reboiler_LMTD': tac_result.get('reboiler', {}).get('LMTD', 0),
            'condenser_area': tac_result.get('condenser', {}).get('Area_m2', 0),
            'reboiler_area': tac_result.get('reboiler', {}).get('Area_m2', 0),
            
            # Vacuum system details
            'vacuum_system': vacuum_sys,
            'vacuum_type': vacuum_sys.get('system_type', 'N/A'),
            'air_leakage_lb_hr': vacuum_sys.get('air_leakage_lb_hr', 0),
            'air_leakage_kg_hr': vacuum_sys.get('air_leakage_kg_hr', 0),
        }

        # Read actual converged RR from Aspen output (for warm-starting)
        try:
            rr_output_path = "\\Data\\Blocks\\{}\\Output\\MOLE_RR".format(self.block_name)
            rr_node = self.aspen.aspen.Tree.FindNode(rr_output_path)
            if rr_node and rr_node.Value is not None:
                result['reflux_ratio'] = rr_node.Value
        except Exception:
            pass

        # Restore original BASIS_RR to prevent contamination of subsequent evaluations
        self.aspen.set_reflux_ratio(self.block_name, self.original_basis_rr)

        # Log result
        logger.info("  CONVERGED")
        logger.info("    Diameter: {:.3f} m".format(diameter))
        logger.info("    T_cond: {:.1f}°C -> {}".format(
            T_cond or 0, result['condenser_utility']))
        logger.info("    T_reb: {:.1f}°C -> {}".format(
            T_reb or 0, result['reboiler_utility']))
        logger.info("    Energy: {:.1f} kW (Reb: {:.1f}, Cond: {:.1f})".format(
            Q_reb + Q_cond, Q_reb, Q_cond))
        
        # Vacuum system info
        if vacuum_sys.get('capital_cost', 0) > 0:
            logger.info("    Vacuum: {} (Air: {:.1f} lb/hr)".format(
                vacuum_sys.get('system_type', 'N/A'),
                vacuum_sys.get('air_leakage_lb_hr', 0)))
            logger.info("    TPC: ${:,.0f} (Vacuum: ${:,.0f})".format(
                tac_result['TPC'], 
                tac_result.get('vacuum_system_cost', 0)))
        else:
            logger.info("    TPC: ${:,.0f}".format(tac_result['TPC']))
        
        logger.info("    TOC: ${:,.0f}/year".format(tac_result['TOC']))
        logger.info("    TAC: ${:,.0f}/year".format(tac_result['TAC']))
        
        # Cache the result
        self.cache[key] = result
        
        return result
    
    def _failed_result(self, reason="Unknown"):
        """Return a result dictionary for failed evaluation."""
        return {
            'TAC': 1e12,
            'TPC': 0,
            'TOC': 0,
            'Q_reb': 0,
            'Q_cond': 0,
            'Q_total': 0,
            'diameter': 0,
            'NT': 0,
            'feed': 0,
            'pressure': 0,
            'converged': False,
            'T_cond': None,
            'T_reb': None,
            'error': reason,
            'vacuum_system_cost': 0,
            'vacuum_operating_cost': 0,
            'vacuum_steam_cost': 0,
            'vacuum_system': {},
        }

    def _evaluate_with_fixed_rr(self, nt, feed, pressure, rr, purity_spec):
        """
        Evaluate TAC with RR recovery: forward mode first, then re-converge with Design Specs.

        Strategy:
        1. Run forward mode (Design Specs OFF, fixed RR) to verify feasibility
        2. If feasible, re-run with Design Specs ON using recovered RR as initial guess
        3. If Design Spec re-converges → return hard feasible TAC (comparable to other points)
        4. If Design Spec fails → fall back to forward-mode TAC (soft feasible)

        This prevents soft feasible points from having artificially low TAC compared to
        hard feasible points, which causes jagged U-curves.

        Parameters
        ----------
        nt : int
            Number of stages
        feed : int
            Feed stage location
        pressure : float
            Column pressure (bar)
        rr : float
            Reflux ratio to use (from successful RR sweep point)
        purity_spec : dict
            Purity specification for logging

        Returns
        -------
        dict : Result dictionary with TAC if successful, None if failed
        """
        logger.info("  [RR Recovery] Attempting evaluation with fixed RR={:.2f}".format(rr))

        # 1. Get all active Design Specs and Vary blocks (to restore later)
        active_specs = self.aspen.get_all_design_specs(self.block_name, max_specs=5)
        active_spec_nums = [s['spec_num'] for s in active_specs]
        all_vary_blocks = self.aspen.get_all_vary_blocks(self.block_name)

        try:
            # 2. Turn OFF all Design Specs and Vary blocks
            for spec_num in active_spec_nums:
                self.aspen.set_design_spec_active(self.block_name, spec_num=spec_num, active=False)
            for vb in all_vary_blocks:
                self.aspen.set_vary_active(self.block_name, vary_num=vb['vary_num'],
                                           spec_num=vb['spec_num'], active=False)

            # 3. Set column parameters
            self.aspen.set_parameters(self.block_name, nt, feed, pressure, self.feed_stream)

            # 4. Set RR directly
            self.aspen.set_reflux_ratio(self.block_name, rr)

            # 5. Run simulation (forward mode)
            success = self.aspen.run_simulation()

            if not success:
                logger.warning("  [RR Recovery] Forward-mode simulation failed to run")
                return None

            # 6. Check convergence
            converged, error_msg = self._check_aspen_convergence()

            if not converged:
                logger.warning("  [RR Recovery] Forward-mode NOT CONVERGED: {}".format(error_msg))
                return None

            # 7. Get forward-mode results (save as fallback)
            fwd_Q_reb, fwd_Q_cond = self.aspen.get_energy_results(self.block_name)
            fwd_diameter = self.aspen.get_diameter(self.block_name)
            fwd_T_cond, fwd_T_reb = self._get_temperatures(nt)

            if fwd_Q_reb is None or fwd_Q_cond is None:
                logger.warning("  [RR Recovery] Cannot read forward-mode energy results")
                return None

            logger.info("  [RR Recovery] Forward-mode OK: Q_reb={:.1f}, Q_cond={:.1f}".format(
                fwd_Q_reb, fwd_Q_cond))

            # ════════════════════════════════════════════════════════════════════
            # STEP 8: RE-CONVERGE WITH DESIGN SPECS ON
            # Use recovered RR as initial guess so Design Spec solver can converge.
            # This produces TAC comparable to hard feasible points.
            # ════════════════════════════════════════════════════════════════════

            logger.info("  [RR Recovery] Attempting Design Spec re-convergence with RR={:.2f} as initial guess...".format(rr))

            # Turn Design Specs and Vary blocks back ON
            for spec_num in active_spec_nums:
                self.aspen.set_design_spec_active(self.block_name, spec_num=spec_num, active=True)
            for vb in all_vary_blocks:
                if vb.get('active') == 'YES':
                    self.aspen.set_vary_active(self.block_name, vary_num=vb['vary_num'],
                                               spec_num=vb['spec_num'], active=True)

            # Keep BASIS_RR at recovered value - this gives the solver a good starting point
            # (don't reset it - the whole point is to use it as initial guess)

            # Re-run with Design Specs ON
            ds_success = self.aspen.run_simulation()
            ds_converged = False

            if ds_success:
                ds_converged, ds_error = self._check_aspen_convergence()

            if ds_converged:
                # Design Spec re-convergence succeeded!
                # Use these results (comparable to hard feasible points)
                Q_reb, Q_cond = self.aspen.get_energy_results(self.block_name)
                diameter = self.aspen.get_diameter(self.block_name)
                T_cond, T_reb = self._get_temperatures(nt)

                if Q_reb is not None and Q_cond is not None:
                    logger.info("  [RR Recovery] Design Spec re-converged! Q_reb={:.1f} (was {:.1f} in forward mode)".format(
                        Q_reb, fwd_Q_reb))

                    try:
                        tac_result = self.tac_calc.calculate(
                            nt=nt, diameter=diameter, Q_reb=Q_reb, Q_cond=Q_cond,
                            pressure=pressure, T_cond=T_cond, T_reb=T_reb
                        )
                    except Exception as e:
                        logger.warning("  [RR Recovery] TAC calc failed after re-convergence: {}".format(e))
                        tac_result = None

                    if tac_result:
                        vacuum_sys = tac_result.get('vacuum_system', {})
                        result = self._build_result_dict(
                            tac_result, Q_reb, Q_cond, diameter, nt, feed, pressure,
                            T_cond, T_reb, vacuum_sys
                        )
                        # This is now effectively hard feasible (Design Spec converged)
                        result['recovered_from_rr_sweep'] = False
                        result['recovery_rr'] = rr
                        result['reconverged_with_design_spec'] = True

                        logger.info("  [RR Recovery] HARD FEASIBLE via re-convergence! TAC=${:,.0f}/yr".format(
                            tac_result['TAC']))
                        return result
                else:
                    logger.warning("  [RR Recovery] Re-converged but cannot read energy results")
            else:
                logger.info("  [RR Recovery] Design Spec re-convergence failed, using forward-mode result")

            # ════════════════════════════════════════════════════════════════════
            # FALLBACK: Use forward-mode TAC (soft feasible)
            # Design Spec couldn't re-converge even with good initial guess.
            # ════════════════════════════════════════════════════════════════════

            # Need to re-run forward mode since Design Spec attempt may have changed state
            for spec_num in active_spec_nums:
                self.aspen.set_design_spec_active(self.block_name, spec_num=spec_num, active=False)
            for vb in all_vary_blocks:
                self.aspen.set_vary_active(self.block_name, vary_num=vb['vary_num'],
                                           spec_num=vb['spec_num'], active=False)
            self.aspen.set_reflux_ratio(self.block_name, rr)
            self.aspen.run_simulation()

            # Use saved forward-mode values for TAC calculation
            try:
                tac_result = self.tac_calc.calculate(
                    nt=nt, diameter=fwd_diameter, Q_reb=fwd_Q_reb, Q_cond=fwd_Q_cond,
                    pressure=pressure, T_cond=fwd_T_cond, T_reb=fwd_T_reb
                )
            except Exception as e:
                logger.warning("  [RR Recovery] TAC calculation failed: {}".format(e))
                return None

            vacuum_sys = tac_result.get('vacuum_system', {})
            result = self._build_result_dict(
                tac_result, fwd_Q_reb, fwd_Q_cond, fwd_diameter, nt, feed, pressure,
                fwd_T_cond, fwd_T_reb, vacuum_sys
            )

            # ════════════════════════════════════════════════════════════════
            # STEP 8b: VALIDATE FORWARD-MODE PURITY
            # Check if forward mode achieves target specs. Log the result
            # but keep as SOFT feasible — forward-mode Q_reb is systematically
            # lower than Design Spec mode, so mixing them causes false dips.
            # The optimizer uses soft feasible as fallback when no hard feasible
            # exists (e.g., dual MASS-RECOV columns like Case9_COL2).
            # ════════════════════════════════════════════════════════════════
            if purity_spec:
                achieved_purity = None
                is_ms = purity_spec.get('is_middle_split', False)
                p_stream = purity_spec.get('stream')
                frac_type = purity_spec.get('fraction_type', 'MASSFRAC')

                if is_ms:
                    top_comps = purity_spec.get('top_components', [])
                    if top_comps and p_stream:
                        achieved_purity = self.aspen.get_stream_multi_purity(p_stream, top_comps, frac_type)
                else:
                    key_comp = purity_spec.get('component')
                    if key_comp and p_stream:
                        achieved_purity = self.aspen.get_stream_purity(p_stream, key_comp, frac_type)

                fwd_target = purity_spec.get('target') or 0.99

                if achieved_purity is not None and achieved_purity >= fwd_target:
                    result['validated_forward_mode'] = True
                    logger.info("  [RR Recovery] Forward-mode purity validated: {:.4f} >= {:.4f}".format(
                        achieved_purity, fwd_target))
                else:
                    result['validated_forward_mode'] = False
                    logger.info("  [RR Recovery] Forward-mode purity check: {} (target={:.4f})".format(
                        "{:.4f}".format(achieved_purity) if achieved_purity is not None else "N/A", fwd_target))

            # Soft feasible - forward mode only (even if purity validated)
            result['recovered_from_rr_sweep'] = True
            result['recovery_rr'] = rr

            logger.info("  [RR Recovery] SOFT FEASIBLE (forward mode): TAC=${:,.0f}/yr at RR={:.2f}".format(
                tac_result['TAC'], rr))

            return result

        except Exception as e:
            logger.warning("  [RR Recovery] Exception: {}".format(e))
            return None

        finally:
            # ALWAYS restore Design Specs and Vary blocks
            try:
                for spec_num in active_spec_nums:
                    self.aspen.set_design_spec_active(self.block_name, spec_num=spec_num, active=True)
                for vb in all_vary_blocks:
                    if vb.get('active') == 'YES':
                        self.aspen.set_vary_active(self.block_name, vary_num=vb['vary_num'],
                                                   spec_num=vb['spec_num'], active=True)
                logger.info("  [RR Recovery] Restored Design Specs and Vary blocks")
            except Exception as restore_error:
                logger.warning("  [RR Recovery] Failed to restore specs: {}".format(restore_error))

            # CRITICAL: Restore original BASIS_RR to avoid contaminating next eval
            try:
                if self.original_basis_rr is not None:
                    self.aspen.set_reflux_ratio(self.block_name, self.original_basis_rr)
                    logger.info("  [RR Recovery] Restored BASIS_RR to {:.3f}".format(self.original_basis_rr))
            except Exception as rr_restore_error:
                logger.warning("  [RR Recovery] Failed to restore BASIS_RR: {}".format(rr_restore_error))

    def _build_result_dict(self, tac_result, Q_reb, Q_cond, diameter, nt, feed, pressure,
                           T_cond, T_reb, vacuum_sys):
        """Build standard result dictionary from TAC calculation results."""
        return {
            'TAC': tac_result['TAC'],
            'TPC': tac_result['TPC'],
            'TOC': tac_result['TOC'],
            'Q_reb': Q_reb,
            'Q_cond': Q_cond,
            'Q_total': Q_reb + Q_cond,
            'diameter': diameter,
            'NT': nt,
            'feed': feed,
            'pressure': pressure,
            'converged': True,

            # Cost breakdown
            'column_cost': tac_result.get('column_cost', 0),
            'tray_cost': tac_result.get('tray_cost', 0),
            'condenser_cost': tac_result.get('condenser_cost', 0),
            'reboiler_cost': tac_result.get('reboiler_cost', 0),
            'vacuum_system_cost': tac_result.get('vacuum_system_cost', 0),
            'steam_cost': tac_result.get('steam_cost', 0),
            'cw_cost': tac_result.get('cw_cost', 0),
            'vacuum_operating_cost': tac_result.get('vacuum_operating_cost', 0),
            'vacuum_steam_cost': tac_result.get('vacuum_operating_cost', 0),

            # Temperature and utility info
            'T_cond': T_cond,
            'T_reb': T_reb,
            'condenser_utility': tac_result.get('condenser', {}).get('utility', 'N/A'),
            'reboiler_utility': tac_result.get('reboiler', {}).get('utility', 'N/A'),
            'condenser_LMTD': tac_result.get('condenser', {}).get('LMTD', 0),
            'reboiler_LMTD': tac_result.get('reboiler', {}).get('LMTD', 0),
            'condenser_area': tac_result.get('condenser', {}).get('Area_m2', 0),
            'reboiler_area': tac_result.get('reboiler', {}).get('Area_m2', 0),

            # Vacuum system details
            'vacuum_system': vacuum_sys,
            'vacuum_type': vacuum_sys.get('system_type', 'N/A'),
            'air_leakage_lb_hr': vacuum_sys.get('air_leakage_lb_hr', 0),
            'air_leakage_kg_hr': vacuum_sys.get('air_leakage_kg_hr', 0),
        }
    
    def get_cached_result(self, nt, feed, pressure):
        """Get result from cache if available."""
        key = self._make_key(nt, feed, pressure)
        return self.cache.get(key, None)
    
    def is_cached(self, nt, feed, pressure):
        """Check if configuration is in cache."""
        key = self._make_key(nt, feed, pressure)
        return key in self.cache
    
    def clear_cache(self):
        """Clear all cached results."""
        self.cache = {}
        self.cache_hits = 0
        logger.info("Cache cleared")
    
    def stats(self):
        """Return evaluation statistics."""
        return {
            'total_evaluations': self.eval_count,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.cache),
            'hit_rate': (self.cache_hits / self.eval_count * 100) if self.eval_count > 0 else 0,
            'failed_count': self.failed_count,
            'success_rate': ((self.eval_count - self.failed_count) / self.eval_count * 100) if self.eval_count > 0 else 0
        }
    
    def print_stats(self):
        """Print formatted evaluation statistics."""
        s = self.stats()
        print("")
        print("=" * 50)
        print("EVALUATION STATISTICS")
        print("=" * 50)
        print("  Total evaluations: {}".format(s['total_evaluations']))
        print("  Cache hits: {}".format(s['cache_hits']))
        print("  Cache size: {}".format(s['cache_size']))
        print("  Hit rate: {:.1f}%".format(s['hit_rate']))
        print("  Failed evaluations: {}".format(s['failed_count']))
        print("  Success rate: {:.1f}%".format(s['success_rate']))
        print("=" * 50)


# ════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("TAC EVALUATOR with SEIDER VACUUM MODEL")
    print("=" * 70)
    print("""

USAGE:
------
In your iso_loop.py or iso_loop_metaheuristic.py:

    # Replace these imports:
    # from tac_calculator import TACCalculator
    # from tac_evaluator import TACEvaluator
    
    # With:
    from tac_calculator_seider import TACCalculator
    from tac_evaluator_seider import TACEvaluator

    # Then use as normal:
    tac_calc = TACCalculator(material='SS', cepci=800)
    evaluator = TACEvaluator(aspen, tac_calc, 'COL2', 'FEED')
    
    result = evaluator.evaluate(nt=45, feed=22, pressure=0.2)
    print(f"TAC: ${result['TAC']:,.0f}")
    print(f"Vacuum system: {result['vacuum_type']}")


NEW RESULT FIELDS:
------------------
- vacuum_system_cost: Capital cost of vacuum equipment ($)
- vacuum_operating_cost: Annual operating cost (steam or electricity) ($/yr)
- vacuum_type: Equipment type selected
- air_leakage_lb_hr: Air leakage rate (lb/hr) from Seider Eq. 22.73
- air_leakage_kg_hr: Air leakage rate (kg/hr)
- vacuum_system: Full dict with all vacuum system details


VACUUM EQUIPMENT AUTO-SELECTION:
--------------------------------
Based on pressure and economics (Table 22.32):

| Pressure Range | Typical Selection | Reason |
|----------------|-------------------|--------|
| > 0.9 bar      | None              | Near-atmospheric |
| 0.1-0.9 bar    | Steam Ejector     | Low capital cost |
| 0.03-0.1 bar   | Mechanical pump   | Lower operating cost |
| < 0.03 bar     | 3-Stage pump      | Deep vacuum capable |

""")
    print("=" * 70)