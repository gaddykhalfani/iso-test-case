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
    
    def _make_key(self, nt, feed, pressure):
        """Create cache key from parameters."""
        p_rounded = round(pressure, self.p_precision)
        return (int(nt), int(feed), p_rounded)
    
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
            
            T_cond = None
            try:
                path_cond = base_path + r"\STAGE_TEMP\1"
                node = self.aspen.aspen.Tree.FindNode(path_cond)
                if node and node.Value:
                    T_cond = node.Value
            except:
                pass
            
            if T_cond is None:
                try:
                    path_cond = base_path + r"\B_TEMP\1"
                    node = self.aspen.aspen.Tree.FindNode(path_cond)
                    if node and node.Value:
                        T_cond = node.Value
                except:
                    pass
            
            T_reb = None
            try:
                path_reb = base_path + r"\STAGE_TEMP\{}".format(nt)
                node = self.aspen.aspen.Tree.FindNode(path_reb)
                if node and node.Value:
                    T_reb = node.Value
            except:
                pass
            
            if T_reb is None:
                try:
                    path_reb = base_path + r"\B_TEMP\{}".format(nt)
                    node = self.aspen.aspen.Tree.FindNode(path_reb)
                    if node and node.Value:
                        T_reb = node.Value
                except:
                    pass
            
            if T_cond is not None and T_reb is not None:
                logger.debug(f"  Temps: T_cond={T_cond:.1f}°C, T_reb={T_reb:.1f}°C")
                return T_cond, T_reb
            else:
                return None, None
                
        except Exception as e:
            logger.warning("  Error extracting temperatures: {}".format(e))
            return None, None
    
    def evaluate(self, nt, feed, pressure):
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
            
        Returns:
        --------
        dict : Result dictionary with TAC and all cost components
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
        
        # Run Aspen simulation
        success = self.aspen.run_simulation()
        
        if not success:
            logger.warning("  Simulation failed to run")
            self.failed_count += 1
            result = self._failed_result(reason="Simulation failed to run")
            self.cache[key] = result
            return result
        
        # Check convergence
        converged, error_msg = self._check_aspen_convergence()
        
        if not converged:
            logger.warning("  NOT CONVERGED: {}".format(error_msg))
            self.failed_count += 1
            result = self._failed_result(reason=error_msg)
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