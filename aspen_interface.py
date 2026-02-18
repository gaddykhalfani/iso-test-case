# aspen_interface.py
"""
Aspen Plus COM interface for RadFrac optimization
Diameter calculation using Fair correlation (Seader et al., 2016)

References:
- Fair, J.R. (1961). Petro/Chem Engineer, 33(10), 45-52.
- Seader, Henley & Roper, "Separation Process Principles", 4th Ed.
- Kister, "Distillation Design", McGraw-Hill, 1992
- Sinnott & Towler, "Chemical Engineering Design", 6th Ed.
"""

import logging
import time
import math

logger = logging.getLogger(__name__)

try:
    import win32com.client as win32
except ImportError:
    logger.error("win32com not installed. Run: pip install pywin32")
    raise


class AspenEnergyOptimizer:
    """Interface to Aspen Plus via COM"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.aspen = None
        self.connected = False
    
    def connect_and_open(self, visible=True):
        """Connect to Aspen and open file"""
        for attempt in range(2):
            try:
                logger.info("Connecting to Aspen Plus...")
                if attempt == 0:
                    self.aspen = win32.gencache.EnsureDispatch("Apwn.Document")
                else:
                    # Late binding — bypasses gen_py cache entirely
                    logger.info("  Using late-bound Dispatch (no cache)...")
                    self.aspen = win32.Dispatch("Apwn.Document")

                self.aspen.InitFromArchive2(self.file_path)
                self.aspen.Visible = visible

                try:
                    self.aspen.SuppressDialogs = 1
                    logger.info("  Dialogs suppressed for automation")
                except:
                    pass

                self.connected = True
                logger.info("Connected successfully!")
                return True
            except Exception as e:
                if attempt == 0 and "gen_py" in str(e):
                    logger.warning("COM cache corrupted, cleaning up and falling back to late binding...")
                    self._cleanup_com_cache()
                    continue
                logger.error("Failed to connect: {}".format(e))
                return False

    def _cleanup_com_cache(self):
        """Clear corrupted win32com gen_py cache from memory and disk."""
        import sys, os, shutil
        try:
            # 1. Remove gen_py modules from Python's in-memory cache
            mods_to_remove = [m for m in sys.modules if m.startswith('win32com.gen_py.')]
            for mod in mods_to_remove:
                del sys.modules[mod]
            if mods_to_remove:
                logger.info("  Cleared {} gen_py modules from sys.modules".format(len(mods_to_remove)))

            # 2. Delete disk cache
            gen_py_path = os.path.join(
                os.environ.get('LOCALAPPDATA', ''), 'Temp', 'gen_py')
            if os.path.exists(gen_py_path):
                shutil.rmtree(gen_py_path, ignore_errors=True)
                logger.info("  Cleared gen_py cache at: {}".format(gen_py_path))
        except Exception as err:
            logger.warning("  Cache cleanup failed: {}".format(err))
    
    def reconnect(self):
        """Reconnect to Aspen if connection was lost"""
        logger.info("Attempting to reconnect to Aspen...")
        try:
            if self.aspen:
                try:
                    self.aspen.Close()
                except:
                    pass
            self.aspen = None
            self.connected = False
            time.sleep(2)
            return self.connect_and_open(visible=True)
        except Exception as e:
            logger.error("Reconnection failed: {}".format(e))
            return False
    
    def is_connected(self):
        """Check if Aspen is still connected"""
        try:
            node = self.aspen.Tree.FindNode("\\Data\\Blocks")
            return node is not None
        except:
            return False
    
    def get_column_info(self, block_name):
        """Get current column configuration"""
        try:
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Input\\NSTAGE".format(block_name))
            nt = node.Value if node else "N/A"
            
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Input\\PRES1".format(block_name))
            pres = node.Value if node else "N/A"
            
            logger.info("  Current NT: {}, Pressure: {} bar".format(nt, pres))
            return nt, pres
        except Exception as e:
            logger.error("Error getting column info: {}".format(e))
            return None, None
    
    def set_parameters(self, block_name, nt, feed_stage, pressure, feed_stream):
        """Set column parameters"""
        try:
            if not self.is_connected():
                logger.warning("Connection lost, attempting reconnect...")
                if not self.reconnect():
                    return False
            
            # Step 1: Set NSTAGE
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Input\\NSTAGE".format(block_name))
            node.Value = nt
            logger.info("  {}.NSTAGE = {}".format(block_name, nt))
            
            # Step 2: Set Feed Stage
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Input\\FEED_STAGE\\{}".format(block_name, feed_stream))
            node.Value = feed_stage
            logger.info("  {}.FEED_STAGE[{}] = {}".format(block_name, feed_stream, feed_stage))
            
            # Step 3: Set Pressure
            pres_node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Input\\PRES1".format(block_name))
            pres_node.Value = pressure
            
            # Verify pressure setting
            check_p = pres_node.Value
            if abs(check_p - pressure) > 0.001:
                logger.warning(f"  [Warning] Pressure set mismatch! Target={pressure}, Read={check_p}")
                # Try setting again
                pres_node.Value = pressure
                check_p_retry = pres_node.Value
                if abs(check_p_retry - pressure) > 0.001:
                    logger.error(f"  [Error] Failed to set pressure! Target={pressure}, Read={check_p_retry}")
                    return False
            
            logger.info("  {}.PRES1 = {} bar".format(block_name, pressure))
            
            return True
            
        except Exception as e:
            logger.error("Error setting parameters: {}".format(e))
            return False
    
    def run_simulation(self, reinit=True):
        """
        Run Aspen simulation.

        Parameters
        ----------
        reinit : bool
            If True, call Reinit() before Run2() to force all blocks to
            reconverge. This prevents stale results when parameters are
            changed via COM but Aspen doesn't detect the change properly.
            Default: True (always reinit for correctness).
        """
        try:
            if not self.is_connected():
                logger.warning("Connection lost before simulation")
                return False

            logger.info("Running simulation...")
            start_time = time.time()

            if reinit:
                self.aspen.Reinit()

            self.aspen.Engine.Run2()

            elapsed = time.time() - start_time
            logger.info(f"Simulation completed in {elapsed:.1f}s")

            return True

        except Exception as e:
            error_str = str(e)
            if any(kw in error_str.upper() for kw in ["RPC", "OLE", "0XE0434352", "REMOTE PROCEDURE"]):
                logger.error("Aspen crashed! Error: {}".format(e))
                self.connected = False
            else:
                logger.error("Simulation failed: {}".format(e))
            return False
    
    def get_energy_results(self, block_name):
        """
        Get reboiler and condenser duties.
        
        IMPORTANT: Aspen Plus returns energy in cal/sec!
        Must convert to kW for TAC calculations.
        
        Conversion: 1 cal/sec = 4.184 W = 0.004184 kW
        
        Reference: Unit conversion
        - 1 calorie = 4.184 Joules
        - 1 cal/sec = 4.184 J/sec = 4.184 W = 0.004184 kW
        """
        try:
            # ════════════════════════════════════════════════════════════════
            # UNIT CONVERSION: cal/sec → kW
            # ════════════════════════════════════════════════════════════════
            CAL_SEC_TO_KW = 0.004184
            
            # Get reboiler duty
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\REB_DUTY".format(block_name))
            Q_reb_raw = abs(node.Value) if node and node.Value else 0.0
            Q_reb = Q_reb_raw * CAL_SEC_TO_KW  # Convert to kW
            
            # Get condenser duty  
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\COND_DUTY".format(block_name))
            Q_cond_raw = abs(node.Value) if node and node.Value else 0.0
            Q_cond = Q_cond_raw * CAL_SEC_TO_KW  # Convert to kW
            
            logger.info("  Reboiler: {:.1f} kW ({:.0f} cal/sec)".format(Q_reb, Q_reb_raw))
            logger.info("  Condenser: {:.1f} kW ({:.0f} cal/sec)".format(Q_cond, Q_cond_raw))
            logger.info("  Total: {:.1f} kW".format(Q_reb + Q_cond))
            
            return Q_reb, Q_cond
            
        except Exception as e:
            logger.error("Error getting energy results: {}".format(e))
            return 0.0, 0.0
    
    def get_diameter(self, block_name):
        """
        Calculate column diameter using Fair correlation.
        
        This method uses Aspen's rigorous vapor and liquid flows to calculate
        the flow parameter (FLV) and apply the Fair flooding correlation.
        
        The Fair correlation accounts for BOTH vapor and liquid traffic,
        making it suitable for thesis-quality design calculations.
        
        References:
        -----------
        1. Fair, J.R. (1961). "How to Predict Sieve Tray Entrainment 
           and Flooding", Petro/Chem Engineer, 33(10), 45-52.
        2. Seader, Henley & Roper (2016). "Separation Process Principles", 
           4th Ed., Wiley, Eq. 6.44-6.51, Fig. 6.36
        3. Kister, H.Z. (1992). "Distillation Design", McGraw-Hill, Ch. 6
        
        Method:
        -------
        1. Get vapor and liquid flows from Aspen (VAP_FLOW, LIQ_FLOW)
        2. Calculate flow parameter FLV = (L/V)_mass × sqrt(rhoV/rhoL)
        3. Calculate capacity parameter C_SB from Fair correlation
        4. Calculate flooding velocity and design at 75% flood
        5. Calculate diameter from cross-sectional area
        
        Returns:
        --------
        float : Column diameter in meters
        """
        try:
            # ════════════════════════════════════════════════════════════════
            # UNIT CONVERSION
            # Aspen METCBAR: molar flow in kmol/hr
            # ════════════════════════════════════════════════════════════════
            KMOL_HR_TO_KMOL_S = 1.0 / 3600.0
            
            # ════════════════════════════════════════════════════════════════
            # DESIGN PARAMETERS
            # ════════════════════════════════════════════════════════════════
            tray_spacing_m = 0.6    # m (24 inches, standard)
            flooding_frac = 0.75    # Design at 75% of flooding
            downcomer_frac = 0.12   # Downcomer area fraction
            
            # ════════════════════════════════════════════════════════════════
            # GET VAPOR FLOW AT STAGE 2 (Top tray for total condenser)
            # 
            # CORRECT PATH: VAP_FLOW\N (1-based stage number)
            # Unit: kmol/hr
            # ════════════════════════════════════════════════════════════════
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\VAP_FLOW\\2".format(block_name))
            V_kmol_hr = node.Value if node and node.Value else 0.0
            
            if V_kmol_hr <= 0:
                logger.warning("No vapor flow data (VAP_FLOW), using default D=1.0m")
                return 1.0
            
            V_kmol_s = V_kmol_hr * KMOL_HR_TO_KMOL_S
            
            # ════════════════════════════════════════════════════════════════
            # GET LIQUID FLOW AT STAGE 2
            # 
            # CORRECT PATH: LIQ_FLOW\N (1-based stage number)
            # Unit: kmol/hr
            # ════════════════════════════════════════════════════════════════
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\LIQ_FLOW\\2".format(block_name))
            L_kmol_hr = node.Value if node and node.Value else 0.0
            
            if L_kmol_hr <= 0:
                # Estimate from reflux ratio: L = V × R/(R+1)
                node = self.aspen.Tree.FindNode(
                    "\\Data\\Blocks\\{}\\Output\\MOLE_RR".format(block_name))
                RR = node.Value if node and node.Value else 1.0
                L_kmol_hr = V_kmol_hr * RR / (RR + 1)
                logger.info("  L estimated from RR: {:.2f} kmol/hr".format(L_kmol_hr))
            
            L_kmol_s = L_kmol_hr * KMOL_HR_TO_KMOL_S
            
            # ════════════════════════════════════════════════════════════════
            # GET MOLECULAR WEIGHTS
            # Default to 104 kg/kmol (EB/SM average)
            # ════════════════════════════════════════════════════════════════
            MW_V = 104.0
            MW_L = 104.0
            
            # Try to get from Aspen
            try:
                node = self.aspen.Tree.FindNode(
                    "\\Data\\Blocks\\{}\\Output\\VAP_MW\\2".format(block_name))
                if node and node.Value and node.Value > 0:
                    MW_V = node.Value
            except:
                pass
            
            try:
                node = self.aspen.Tree.FindNode(
                    "\\Data\\Blocks\\{}\\Output\\LIQ_MW\\2".format(block_name))
                if node and node.Value and node.Value > 0:
                    MW_L = node.Value
            except:
                pass
            
            # ════════════════════════════════════════════════════════════════
            # GET TEMPERATURE AND PRESSURE AT STAGE 2
            # 
            # CORRECT PATH: B_TEMP\(N-1), B_PRES\(N-1) (0-based indexing!)
            # Stage 2 → index 1
            # ════════════════════════════════════════════════════════════════
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\B_TEMP\\1".format(block_name))
            T_C = node.Value if node and node.Value else 80.0
            T_K = T_C + 273.15
            
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\B_PRES\\1".format(block_name))
            P_bar = node.Value if node and node.Value else 0.2
            P_Pa = P_bar * 100000
            
            # ════════════════════════════════════════════════════════════════
            # GET OR CALCULATE DENSITIES
            # ════════════════════════════════════════════════════════════════
            
            # Vapor density - try Aspen first, then ideal gas
            rho_V = None
            try:
                node = self.aspen.Tree.FindNode(
                    "\\Data\\Blocks\\{}\\Output\\VAP_RHO\\2".format(block_name))
                if node and node.Value and node.Value > 0:
                    rho_V = node.Value
            except:
                pass
            
            if rho_V is None or rho_V <= 0:
                # Ideal gas law: rho = PM/(RT)
                R_gas = 8314  # J/(kmol·K)
                rho_V = (P_Pa * MW_V) / (R_gas * T_K)
            
            # Liquid density - try Aspen first, then default
            rho_L = None
            try:
                node = self.aspen.Tree.FindNode(
                    "\\Data\\Blocks\\{}\\Output\\LIQ_RHO\\2".format(block_name))
                if node and node.Value and node.Value > 0:
                    rho_L = node.Value
            except:
                pass
            
            if rho_L is None or rho_L <= 0:
                rho_L = 820.0  # Default for EB/SM mixture
            
            # ════════════════════════════════════════════════════════════════
            # CALCULATE MASS FLOWS
            # ════════════════════════════════════════════════════════════════
            V_kg_s = V_kmol_s * MW_V  # kg/s
            L_kg_s = L_kmol_s * MW_L  # kg/s
            
            # ════════════════════════════════════════════════════════════════
            # FLOW PARAMETER FLV (Seader Eq. 6.44, Fair 1961)
            # 
            # FLV = (L/V)_mass × sqrt(rhoV/rhoL)
            # 
            # This parameter accounts for liquid traffic!
            # Valid range: 0.01 - 1.0
            # ════════════════════════════════════════════════════════════════
            FLV = (L_kg_s / V_kg_s) * math.sqrt(rho_V / rho_L)
            
            # Limit to valid range
            FLV_calc = max(0.01, min(FLV, 1.0))
            
            # ════════════════════════════════════════════════════════════════
            # CAPACITY PARAMETER C_SB - FAIR CORRELATION (Seader Eq. 6.46)
            # 
            # C_SB = 0.0105 + 8.127e-4 × Ht^0.755 × exp(-1.463 × FLV^0.842)
            # 
            # where Ht = tray spacing in mm
            # 
            # This correlation is from Fair (1961) and reproduced in:
            # - Seader et al. (2016), Eq. 6.46, Fig. 6.36
            # - Kister (1992), Chapter 6
            # ════════════════════════════════════════════════════════════════
            H_t_mm = tray_spacing_m * 1000  # Convert m to mm
            
            C_SB = 0.0105 + 8.127e-4 * (H_t_mm ** 0.755) * math.exp(-1.463 * (FLV_calc ** 0.842))
            
            # ════════════════════════════════════════════════════════════════
            # FLOODING VELOCITY (Seader Eq. 6.44)
            # 
            # u_flood = C_SB × sqrt[(rhoL - rhoV) / rhoV]
            # ════════════════════════════════════════════════════════════════
            u_flood = C_SB * math.sqrt((rho_L - rho_V) / rho_V)
            
            # Design velocity at specified flooding fraction
            u_design = flooding_frac * u_flood
            
            # ════════════════════════════════════════════════════════════════
            # CALCULATE DIAMETER
            # 
            # Q_V = V_kg_s / rhoV  (actual volumetric vapor flow)
            # A_net = Q_V / u_design  (net area for vapor flow)
            # A_total = A_net / (1 - downcomer_frac)
            # D = sqrt(4 × A_total / pi)
            # ════════════════════════════════════════════════════════════════
            Q_V = V_kg_s / rho_V  # m³/s
            A_net = Q_V / u_design  # m² (net area for vapor flow)
            A_total = A_net / (1 - downcomer_frac)  # m² (total column area)
            
            D_c = math.sqrt(4 * A_total / math.pi)  # m
            
            # Apply reasonable limits
            D_c = max(0.3, min(D_c, 12.0))
            
            # ════════════════════════════════════════════════════════════════
            # LOG RESULTS
            # ════════════════════════════════════════════════════════════════
            logger.info("  V = {:.2f} kmol/hr, L = {:.2f} kmol/hr".format(V_kmol_hr, L_kmol_hr))
            logger.info("  rhoV = {:.4f} kg/m3, rhoL = {:.1f} kg/m3".format(rho_V, rho_L))
            logger.info("  FLV = {:.4f}".format(FLV))
            logger.info("  C_SB = {:.4f} m/s (Fair correlation)".format(C_SB))
            logger.info("  u_flood = {:.3f} m/s, u_design = {:.3f} m/s".format(u_flood, u_design))
            logger.info("  Diameter = {:.3f} m".format(D_c))
            
            return D_c
            
        except Exception as e:
            logger.warning("Diameter calculation failed: {}, using default 1.0m".format(e))
            return 1.0
    
    def _calculate_diameter_from_reflux(self, block_name):
        """
        Calculate diameter from distillate and reflux ratio (backup method).
        
        For total condenser (Seader Eq. 6.3):
        V_2 = D * (R + 1)
        """
        try:
            # Get distillate rate
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\MOLE_DFR".format(block_name))
            D_molar = node.Value if node and node.Value else 0.0
            
            if D_molar <= 0:
                return 0.0
            
            # Get reflux ratio
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\MOLE_RR".format(block_name))
            R = node.Value if node and node.Value else 1.0
            
            # Calculate vapor flow at top (Seader Eq. 6.3)
            V_molar = D_molar * (R + 1)  # kmol/hr
            
            # Get top conditions (0-based indexing for B_TEMP, B_PRES)
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\B_TEMP\\1".format(block_name))
            T_C = node.Value if node and node.Value else 80
            T_K = T_C + 273.15
            
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\B_PRES\\1".format(block_name))
            P_bar = node.Value if node and node.Value else 0.2
            P_Pa = P_bar * 100000
            
            # Molecular weights and densities
            MW_V = 104  # EB/SM average
            rho_L = 820  # kg/m3
            
            # Vapor density (ideal gas)
            R_gas = 8314
            rho_V = (P_Pa * MW_V) / (R_gas * T_K)
            
            # Convert molar to mass flow
            V_kmol_s = V_molar / 3600  # kmol/hr to kmol/s
            V_kg_s = V_kmol_s * MW_V
            
            # Fair correlation (simplified)
            C_SB = 0.07  # For 0.6m tray spacing, low FLV
            u_flood = C_SB * math.sqrt((rho_L - rho_V) / rho_V)
            u_design = 0.75 * u_flood
            
            # Volumetric flow and diameter
            Q_V = V_kg_s / rho_V
            A_c = Q_V / u_design
            D_c = math.sqrt(4 * A_c / math.pi)
            
            D_c = max(0.3, min(D_c, 12.0))
            
            logger.info("  Diameter: {:.3f} m (from D*(R+1))".format(D_c))
            
            return D_c
        
        except Exception as e:
            logger.debug("  Reflux calculation failed: {}".format(e))
            return 0.0
    
    def _calculate_diameter_from_duty(self, block_name):
        """
        Estimate diameter from reboiler duty (fallback method).
        
        Note: This is a rough estimate only. Use get_diameter() for accurate results.
        """
        try:
            # Get reboiler duty (in cal/sec from Aspen)
            CAL_SEC_TO_KW = 0.004184
            
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\REB_DUTY".format(block_name))
            Q_reb_raw = abs(node.Value) if node and node.Value else 0.0
            Q_reb = Q_reb_raw * CAL_SEC_TO_KW  # Convert to kW
            
            if Q_reb < 1:
                logger.warning("  Q_reb is 0, using default diameter 1.0 m")
                return 1.0
            
            # Empirical correlation for vacuum EB/SM columns
            # Calibrated: 250 kW → ~1.0 m diameter
            D_c = 0.064 * (Q_reb ** 0.50)
            
            D_c = max(0.3, min(D_c, 12.0))
            
            logger.info("  Diameter: {:.3f} m (from Q_reb={:.1f} kW)".format(D_c, Q_reb))
            
            return D_c
            
        except Exception as e:
            logger.warning("  Duty calculation failed, using default: 1.0 m")
            return 1.0
    
    def get_output_rr(self, block_name, vary_num=2):
        """
        Get actual reflux ratio from Aspen OUTPUT.

        Tries multiple paths in order of preference:
        1. VAR_VAL\2 output (calculated RR from Vary block when Design Spec is active)
        2. MOLE_RR output (standard reflux ratio output)

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block
        vary_num : int, optional
            Specific Vary block number to read from (default: 2)
            VAR_VAL\2 typically contains the RR from Vary block

        Returns
        -------
        float : Reflux ratio value, or 0.0 if not found
        """
        try:
            base_path = "\\Data\\Blocks\\{}\\Output".format(block_name)
            
            # Sanity check bounds for reflux ratio
            # High RR (up to 100) is possible for difficult separations
            RR_MIN = 0.1
            RR_MAX = 100.0
            
            # Try VAR_VAL\2 first (calculated RR from Vary block)
            try:
                path = base_path + "\\VAR_VAL\\{}".format(vary_num)
                node = self.aspen.Tree.FindNode(path)
                if node and node.Value is not None:
                    val = node.Value
                    if RR_MIN <= val <= RR_MAX:
                        logger.debug("  RR from VAR_VAL\\{}: {:.3f}".format(vary_num, val))
                        return val
                    else:
                        logger.debug("  VAR_VAL\\{} = {:.1f} (out of range, skipping)".format(vary_num, val))
            except:
                pass
            
            # Fallback to MOLE_RR
            try:
                path = base_path + "\\MOLE_RR"
                node = self.aspen.Tree.FindNode(path)
                if node and node.Value is not None and RR_MIN <= node.Value <= RR_MAX:
                    logger.debug("  RR from MOLE_RR: {:.3f}".format(node.Value))
                    return node.Value
            except:
                pass
            
            return 0.0
        except:
            return 0.0
    
    def get_distillate_rate(self, block_name):
        """Get distillate molar flow rate"""
        try:
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\MOLE_DFR".format(block_name))
            return node.Value if node and node.Value else 0.0
        except:
            return 0.0
    
    # ════════════════════════════════════════════════════════════════════════════
    # DESIGN SPEC DIAGNOSTIC METHODS
    # ════════════════════════════════════════════════════════════════════════════

    def get_design_spec_info(self, block_name, spec_num=1):
        """
        Get Design Spec configuration and status.

        For RadFrac blocks, Design Specs use:
        - Active status: \\Data\\Blocks\\{block}\\Input\\SPEC_ACTIVE\\{spec_num}
        - Target value: \\Data\\Blocks\\{block}\\Input\\VALUE\\{spec_num}

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block (e.g., 'COL2')
        spec_num : int
            Design Spec number (1, 2, 3, etc.)

        Returns
        -------
        dict : {'active': str, 'target': float, 'spec_num': int} or None if error
        """
        try:
            base = "\\Data\\Blocks\\{}\\Input".format(block_name)
            active_node = self.aspen.Tree.FindNode(base + "\\SPEC_ACTIVE\\{}".format(spec_num))
            target_node = self.aspen.Tree.FindNode(base + "\\VALUE\\{}".format(spec_num))

            return {
                'spec_num': spec_num,
                'active': active_node.Value if active_node else None,
                'target': target_node.Value if target_node else None,
            }
        except Exception as e:
            logger.warning("Error getting Design Spec {} info: {}".format(spec_num, e))
            return None

    def get_design_spec_type(self, block_name, spec_num=1):
        """
        Get the type of a Design Spec (e.g., MASS-FRAC, MASS-REC).

        Reads from: \\Data\\Blocks\\{block}\\Input\\SPEC_TYPE\\{spec_num}

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block
        spec_num : int
            Design Spec number

        Returns
        -------
        str or None : Spec type string (e.g., 'MASS-FRAC', 'MOLE-FRAC', 'MASS-REC')
        """
        try:
            path = "\\Data\\Blocks\\{}\\Input\\SPEC_TYPE\\{}".format(block_name, spec_num)
            node = self.aspen.Tree.FindNode(path)
            if node and node.Value:
                return str(node.Value)
        except Exception as e:
            logger.debug("Could not read SPEC_TYPE for spec {}: {}".format(spec_num, e))
        return None

    def auto_detect_purity_target(self, block_name):
        """
        Auto-detect the purity Design Spec target from the Aspen block.

        Reads all active Design Specs and identifies the purity spec
        (MASS-FRAC or MOLE-FRAC type) vs recovery spec (MASS-REC type).

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block

        Returns
        -------
        dict or None : {'spec_num': int, 'target': float, 'spec_type': str}
                       Returns None if no purity spec found
        """
        specs = self.get_all_design_specs(block_name)
        if not specs:
            logger.info("  No active Design Specs found for {}".format(block_name))
            return None

        logger.info("  Auto-detecting purity target from {} active Design Specs...".format(len(specs)))

        # Try to identify purity spec by type
        purity_specs = []
        recovery_specs = []
        unknown_specs = []

        for spec in specs:
            spec_type = self.get_design_spec_type(block_name, spec['spec_num'])
            spec['spec_type'] = spec_type

            if spec_type:
                logger.info("    Spec {}: target={}, type={}".format(
                    spec['spec_num'], spec['target'], spec_type))
                type_upper = spec_type.upper()
                if 'FRAC' in type_upper and 'REC' not in type_upper:
                    purity_specs.append(spec)
                elif 'REC' in type_upper:
                    recovery_specs.append(spec)
                else:
                    unknown_specs.append(spec)
            else:
                logger.info("    Spec {}: target={}, type=unknown".format(
                    spec['spec_num'], spec['target']))
                unknown_specs.append(spec)

        # Case 1: Found purity spec by type
        if purity_specs:
            chosen = purity_specs[0]
            logger.info("  -> Auto-detected purity spec: Spec {} (target={}, type={})".format(
                chosen['spec_num'], chosen['target'], chosen.get('spec_type')))
            return chosen

        # Case 2: Could not identify by type - use heuristic (lower target = purity)
        if len(specs) >= 2:
            sorted_specs = sorted(specs, key=lambda s: s['target'] if s['target'] else 1.0)
            chosen = sorted_specs[0]
            logger.warning("  -> Could not identify spec types, using heuristic (lowest target)")
            logger.warning("     Selected Spec {} (target={}) as purity spec".format(
                chosen['spec_num'], chosen['target']))
            return chosen

        # Case 3: Only one spec - assume it's purity
        if len(specs) == 1:
            chosen = specs[0]
            logger.info("  -> Single spec found: Spec {} (target={})".format(
                chosen['spec_num'], chosen['target']))
            return chosen

        return None

    def get_all_design_specs(self, block_name, max_specs=5):
        """
        Get all Design Specs for a block.

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block
        max_specs : int
            Maximum number of specs to check (default: 5)

        Returns
        -------
        list : List of active Design Spec info dicts
        """
        specs = []
        for i in range(1, max_specs + 1):
            info = self.get_design_spec_info(block_name, i)
            if info and info.get('active') == 'YES':
                specs.append(info)
        return specs

    def set_design_spec_active(self, block_name, spec_num=1, active=True):
        """
        Enable or disable a Design Spec.

        For RadFrac blocks: \\Data\\Blocks\\{block}\\Input\\SPEC_ACTIVE\\{spec_num}

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block (e.g., 'COL2')
        spec_num : int
            Design Spec number (1, 2, 3, etc.)
        active : bool
            True to enable, False to disable

        Returns
        -------
        bool : True if successful
        """
        try:
            path = "\\Data\\Blocks\\{}\\Input\\SPEC_ACTIVE\\{}".format(block_name, spec_num)
            node = self.aspen.Tree.FindNode(path)
            if node:
                node.Value = "YES" if active else "NO"
                logger.info("  Design Spec {} set to {}".format(spec_num, 'ACTIVE' if active else 'INACTIVE'))
                return True
            else:
                logger.warning("  Design Spec {} node not found".format(spec_num))
            return False
        except Exception as e:
            logger.warning("Error setting Design Spec {}: {}".format(spec_num, e))
            return False

    def set_vary_active(self, block_name, vary_num=1, spec_num=1, active=True):
        """
        Enable or disable a Vary block for a specific spec.

        Path: \\Data\\Blocks\\{block}\\Subobjects\\Vary\\{vary_num}\\Input\\VARY_ACTIVE\\{spec_num}

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block
        vary_num : int
            Vary block number (default: 1)
        spec_num : int
            Design Spec number this Vary is associated with (default: 1)
        active : bool
            True to enable, False to disable

        Returns
        -------
        bool : True if successful
        """
        try:
            path = "\\Data\\Blocks\\{}\\Subobjects\\Vary\\{}\\Input\\VARY_ACTIVE\\{}".format(
                block_name, vary_num, spec_num)
            node = self.aspen.Tree.FindNode(path)
            if node:
                node.Value = "YES" if active else "NO"
                logger.info("  Vary {} for Spec {} set to {}".format(
                    vary_num, spec_num, 'ACTIVE' if active else 'INACTIVE'))
                return True
            return False
        except Exception as e:
            logger.warning("Error setting Vary {}: {}".format(vary_num, e))
            return False

    def get_vary_info(self, block_name, vary_num=1, spec_num=1):
        """
        Get Vary block bounds and current value.

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block (e.g., 'COL2')
        vary_num : int
            Vary block number (default: 1)
        spec_num : int
            Design Spec number (default: 1)

        Returns
        -------
        dict : {'lower': float, 'upper': float, 'value': float, 'at_bound': bool, 'active': str}
        """
        try:
            base = "\\Data\\Blocks\\{}\\Subobjects\\Vary\\{}".format(block_name, vary_num)

            # Get active status for this spec
            active_node = self.aspen.Tree.FindNode(base + "\\Input\\VARY_ACTIVE\\{}".format(spec_num))

            # Get bounds and value
            lower_node = self.aspen.Tree.FindNode(base + "\\Input\\LOWER")
            upper_node = self.aspen.Tree.FindNode(base + "\\Input\\UPPER")
            value_node = self.aspen.Tree.FindNode(base + "\\Output\\VALUE")

            active = active_node.Value if active_node else None
            lower = lower_node.Value if lower_node else None
            upper = upper_node.Value if upper_node else None
            value = value_node.Value if value_node else None

            at_bound = False
            if value is not None and lower is not None and upper is not None:
                at_bound = (abs(value - lower) < 0.01) or (abs(value - upper) < 0.01)

            return {
                'active': active,
                'lower': lower,
                'upper': upper,
                'value': value,
                'at_bound': at_bound
            }
        except Exception as e:
            logger.warning("Error getting Vary {} info: {}".format(vary_num, e))
            return None

    def get_all_vary_blocks(self, block_name, max_vary=5):
        """
        Get all existing Vary blocks for a RadFrac block.

        This discovers the actual Vary-to-Spec associations which may not
        follow the pattern vary_num == spec_num.

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block
        max_vary : int
            Maximum number of Vary blocks to check (default: 5)

        Returns
        -------
        list : List of dicts with keys: 'vary_num', 'spec_num', 'active'
        """
        vary_blocks = []
        for vary_num in range(1, max_vary + 1):
            base = "\\Data\\Blocks\\{}\\Subobjects\\Vary\\{}".format(block_name, vary_num)
            # Check if this Vary block exists
            try:
                node = self.aspen.Tree.FindNode(base)
                if node is None:
                    continue

                # Check which spec this Vary is associated with (check specs 1-5)
                for spec_num in range(1, 6):
                    active_path = base + "\\Input\\VARY_ACTIVE\\{}".format(spec_num)
                    active_node = self.aspen.Tree.FindNode(active_path)
                    if active_node and active_node.Value:
                        vary_blocks.append({
                            'vary_num': vary_num,
                            'spec_num': spec_num,
                            'active': active_node.Value
                        })
            except:
                continue
        return vary_blocks

    def get_stream_purity(self, stream_name, component, fraction_type='MASSFRAC'):
        """
        Get component fraction in a stream.

        Parameters
        ----------
        stream_name : str
            Name of the stream (e.g., 'TOLL', 'EBB', 'LIQPROD2')
        component : str
            Component name (e.g., 'STYRE-01', 'TOLUE-01', 'ETHYL-01')
        fraction_type : str
            'MASSFRAC' or 'MOLEFRAC' (default: 'MASSFRAC')

        Returns
        -------
        float : Fraction value or None if error
        """
        try:
            path = "\\Data\\Streams\\{}\\Output\\{}\\MIXED\\{}".format(
                stream_name, fraction_type, component)
            node = self.aspen.Tree.FindNode(path)
            return node.Value if node else None
        except:
            return None

    def get_stream_multi_purity(self, stream_name, components, fraction_type='MASSFRAC'):
        """
        Get combined purity by summing multiple component fractions.

        Used for middle split columns where purity = sum of key components.
        For example, for Case8_COL2 top stream EBTOL, the combined purity is
        the sum of TOLUE-01 + ETHYL-01 + STYRE-01 mass fractions.

        Parameters
        ----------
        stream_name : str
            Name of the stream (e.g., 'EBTOL', 'STYAMS')
        components : list
            List of component names to sum (e.g., ['TOLUE-01', 'ETHYL-01', 'STYRE-01'])
        fraction_type : str
            'MASSFRAC' or 'MOLEFRAC' (default: 'MASSFRAC')

        Returns
        -------
        float : Sum of mass/mole fractions for all components in list
        """
        total = 0.0
        for comp in components:
            frac = self.get_stream_purity(stream_name, comp, fraction_type)
            if frac is not None:
                total += frac
        return total

    def run_diagnostic(self, block_name, nt, feed, pressure, feed_stream,
                       purity_spec=None):
        """
        Run diagnostic simulation with ALL Design Specs off to check feasibility.

        This helps determine if convergence failures are due to:
        1. Impossible purity specification (not achievable at this NT/NF/P)
        2. Reflux ratio hitting bounds
        3. Solver difficulty (achievable but hard to converge)

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block
        nt : int
            Number of stages
        feed : int
            Feed stage location
        pressure : float
            Column pressure (bar)
        feed_stream : str
            Feed stream name
        purity_spec : dict
            From config.PURITY_SPECS - must contain:
            - 'stream': Product stream name
            - 'component': Key component name
            - 'fraction_type': 'MASSFRAC' or 'MOLEFRAC'
            - 'target': Target purity value (optional)

        Returns
        -------
        dict : Diagnostic results including:
            - converged_without_spec: bool
            - natural_rr: float (reflux ratio without spec)
            - achieved_purity: float (purity achieved without spec)
            - target_purity: float (original target from first spec)
            - active_specs: list of originally active spec numbers
            - vary_info: dict (Vary block status)
        """
        # Extract purity spec parameters
        if purity_spec is None:
            logger.warning("  WARNING: purity_spec not provided, using defaults")
            purity_spec = {'stream': 'DISTIL', 'component': 'STYRENE',
                           'fraction_type': 'MASSFRAC', 'target': 0.99}

        product_stream = purity_spec.get('stream')
        key_component = purity_spec.get('component')
        fraction_type = purity_spec.get('fraction_type', 'MASSFRAC')
        spec_target = purity_spec.get('target')
        is_middle_split = purity_spec.get('is_middle_split', False)
        top_components = purity_spec.get('top_components', [])

        logger.info("=" * 50)
        logger.info("RUNNING DIAGNOSTIC (All Design Specs OFF)")
        logger.info("=" * 50)
        if is_middle_split:
            logger.info("  Middle split column - checking combined purity of: {}".format(top_components))

        # 1. Get ALL active Design Specs (check up to 5)
        active_specs = self.get_all_design_specs(block_name, max_specs=5)
        active_spec_nums = [s['spec_num'] for s in active_specs]

        if active_specs:
            logger.info("  Found {} active Design Specs: {}".format(
                len(active_specs), active_spec_nums))
            for spec in active_specs:
                logger.info("    Spec {}: target = {}".format(
                    spec['spec_num'], spec.get('target')))
        else:
            logger.info("  No active Design Specs found")

        # Get target: prefer spec_target from purity_spec, fallback to Design Spec
        first_target = spec_target if spec_target else (
            active_specs[0].get('target') if active_specs else None)

        # Get Vary info before diagnostic
        vary_before = self.get_vary_info(block_name)

        # 2. Turn off ALL active Design Specs
        for spec_num in active_spec_nums:
            self.set_design_spec_active(block_name, spec_num=spec_num, active=False)

        # Turn off ALL Vary blocks (discover actual Vary-Spec associations)
        all_vary_blocks = self.get_all_vary_blocks(block_name)
        if all_vary_blocks:
            logger.info("  Found {} Vary blocks: {}".format(
                len(all_vary_blocks),
                [(vb['vary_num'], vb['spec_num']) for vb in all_vary_blocks]))
        for vb in all_vary_blocks:
            self.set_vary_active(block_name, vary_num=vb['vary_num'],
                                 spec_num=vb['spec_num'], active=False)

        # 3. Set parameters (they should already be set, but ensure)
        self.set_parameters(block_name, nt, feed, pressure, feed_stream)

        # 4. Run simulation
        success = self.run_simulation()

        result = {
            'converged_without_spec': False,
            'natural_rr': None,
            'achieved_purity': None,
            'target_purity': first_target,
            'active_specs': active_spec_nums,
            'vary_info': vary_before,
        }

        if success:
            # Check convergence
            try:
                block_path = "\\Data\\Blocks\\{}\\Output\\BLKSTAT".format(block_name)
                node = self.aspen.Tree.FindNode(block_path)
                status = node.Value if node else None

                if status == 0 or (isinstance(status, str) and 'OK' in str(status).upper()):
                    result['converged_without_spec'] = True
                else:
                    logger.info("  BLKSTAT={} (not 0/OK, but will still read stream results)".format(status))

                # Always read stream results after successful simulation
                # In forward mode, BLKSTAT may be non-zero (warnings) but streams are valid
                result['natural_rr'] = self.get_output_rr(block_name)

                if is_middle_split:
                    result['achieved_purity'] = self.get_stream_multi_purity(
                        product_stream, top_components, fraction_type)
                else:
                    result['achieved_purity'] = self.get_stream_purity(
                        product_stream, key_component, fraction_type)

                # If we got valid results, mark as converged
                if result['achieved_purity'] is not None and result['achieved_purity'] > 0:
                    result['converged_without_spec'] = True

                if result['converged_without_spec']:
                    logger.info("  Converged without Design Specs!")
                    if result['natural_rr'] is not None:
                        logger.info("  Natural RR: {:.2f}".format(result['natural_rr']))
                    if result['achieved_purity'] is not None:
                        logger.info("  Achieved purity: {:.4f}".format(result['achieved_purity']))
                    if result['target_purity'] is not None:
                        logger.info("  Target purity: {:.4f}".format(result['target_purity']))
                        if result['achieved_purity'] is not None:
                            if result['achieved_purity'] < result['target_purity']:
                                logger.info("  >>> PURITY SPEC UNACHIEVABLE at this config!")
                            else:
                                logger.info("  >>> Purity achievable (solver difficulty)")
                else:
                    logger.info("  Still did not converge without Design Specs (status={})".format(status))

            except Exception as e:
                logger.warning("Error reading diagnostic results: {}".format(e))
        else:
            logger.info("  Simulation failed to run even without Design Specs")

        # 5. Restore ALL Design Specs AND Vary blocks that were originally active
        for spec_num in active_spec_nums:
            self.set_design_spec_active(block_name, spec_num=spec_num, active=True)
        # Restore all Vary blocks that were originally active
        for vb in all_vary_blocks:
            if vb['active'] == 'YES':
                self.set_vary_active(block_name, vary_num=vb['vary_num'],
                                     spec_num=vb['spec_num'], active=True)
        if active_spec_nums or all_vary_blocks:
            logger.info("  Restored {} Design Specs and {} Vary blocks to ACTIVE".format(
                len(active_spec_nums), len([vb for vb in all_vary_blocks if vb['active'] == 'YES'])))

        logger.info("=" * 50)

        return result

    # ════════════════════════════════════════════════════════════════════════════
    # RR SWEEP DIAGNOSTIC METHODS
    # ════════════════════════════════════════════════════════════════════════════

    def set_reflux_ratio(self, block_name, rr_value):
        """
        Set reflux ratio directly (use when Design Specs are OFF).

        In forward mode operation, RR is the independent variable.
        The column runs with this fixed RR and purity becomes the dependent result.

        Path: \\Data\\Blocks\\{block}\\Input\\BASIS_RR

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block
        rr_value : float
            Molar reflux ratio to set

        Returns
        -------
        bool : True if successful
        """
        try:
            path = "\\Data\\Blocks\\{}\\Input\\BASIS_RR".format(block_name)
            node = self.aspen.Tree.FindNode(path)
            if node:
                node.Value = rr_value
                logger.info("  Set RR = {:.3f}".format(rr_value))
                return True
            else:
                logger.warning("  RR input node not found at {}".format(path))
            return False
        except Exception as e:
            logger.warning("Error setting RR: {}".format(e))
            return False

    def get_basis_rr(self, block_name):
        """
        Get current BASIS_RR value from Aspen INPUT.

        Used to read the original reflux ratio before RR recovery
        so it can be restored afterwards.

        Path: \\Data\\Blocks\\{block}\\Input\\BASIS_RR

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block

        Returns
        -------
        float or None : Current BASIS_RR value, or None if not found
        """
        try:
            path = "\\Data\\Blocks\\{}\\Input\\BASIS_RR".format(block_name)
            node = self.aspen.Tree.FindNode(path)
            if node:
                return node.Value
            else:
                logger.warning("  RR input node not found at {}".format(path))
                return None
        except Exception as e:
            logger.warning("Error getting RR: {}".format(e))
            return None

    def diagnose_column_config(self, block_name, max_vary=5, max_specs=5):
        """
        Introspect a RadFrac column's actual operating specification from Aspen Plus.

        Reads ALL possible configuration paths to determine what operating spec
        the column actually uses. Critical because sweep_rr_purity writes to
        BASIS_RR, but if the column uses a different spec (e.g. D:F), BASIS_RR
        is silently ignored.

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block (e.g., 'COL2')
        max_vary : int
            Maximum number of Vary blocks to check (default: 5)
        max_specs : int
            Maximum number of Design Specs to check (default: 5)

        Returns
        -------
        dict : Diagnostic results with keys:
            - 'block_name', 'operating_spec', 'input_values', 'output_values'
            - 'vary_blocks', 'design_specs', 'vary_tree_children'
            - 'active_spec_type': str or None
            - 'basis_rr_is_active': bool or None
            - 'warnings': list of str
        """
        logger.info("=" * 60)
        logger.info("COLUMN CONFIGURATION DIAGNOSTIC: {}".format(block_name))
        logger.info("=" * 60)

        result = {
            'block_name': block_name,
            'operating_spec': {},
            'input_values': {},
            'output_values': {},
            'vary_blocks': [],
            'design_specs': [],
            'vary_tree_children': {},
            'active_spec_type': None,
            'basis_rr_is_active': None,
            'warnings': [],
        }

        base_input = "\\Data\\Blocks\\{}\\Input".format(block_name)
        base_output = "\\Data\\Blocks\\{}\\Output".format(block_name)

        # ────────────────────────────────────────────────────────────
        # SECTION 1: Operating Specification Type
        # ────────────────────────────────────────────────────────────
        logger.info("")
        logger.info("--- SECTION 1: Operating Specification Type ---")

        spec_type_paths = {
            'OPT_SPEC\\1':  base_input + "\\OPT_SPEC\\1",
            'OPT_SPEC\\2':  base_input + "\\OPT_SPEC\\2",
            'SPEC\\1':      base_input + "\\SPEC\\1",
            'SPEC\\2':      base_input + "\\SPEC\\2",
            'CON_SPEC':     base_input + "\\CON_SPEC",
            'CON_BASIS':    base_input + "\\CON_BASIS",
            'CALC_MODE':    base_input + "\\CALC_MODE",
            'COND_TYPE':    base_input + "\\CONDENSER",   # TOTAL, PARTIAL, NONE
            'REB_TYPE':     base_input + "\\REBOILER",    # KETTLE, THERMOSIPHON, NONE
            'OPT_RR':       base_input + "\\OPT_RR",
            'OPT_DF':       base_input + "\\OPT_DF",
            'OPT_BR':       base_input + "\\OPT_BR",
            'OPT_D':        base_input + "\\OPT_D",
            'OPT_B':        base_input + "\\OPT_B",
            'NSTAGE':       base_input + "\\NSTAGE",
            'FEED_STAGE\\1': base_input + "\\FEED_STAGE\\1",
        }

        for label, path in spec_type_paths.items():
            try:
                node = self.aspen.Tree.FindNode(path)
                if node is not None:
                    val = node.Value
                    result['operating_spec'][label] = val
                    if val is not None:
                        logger.info("  {} = {}".format(label, val))
                else:
                    result['operating_spec'][label] = None
            except Exception as e:
                result['operating_spec'][label] = "ERROR: {}".format(e)
                logger.debug("  {} -> error: {}".format(label, e))

        # ────────────────────────────────────────────────────────────
        # SECTION 2: All Operating Variable Input Values
        # ────────────────────────────────────────────────────────────
        logger.info("")
        logger.info("--- SECTION 2: All Operating Variable Inputs ---")

        input_paths = {
            'BASIS_RR':  base_input + "\\BASIS_RR",
            'BASIS_D':   base_input + "\\BASIS_D",
            'BASIS_B':   base_input + "\\BASIS_B",
            'BASIS_BR':  base_input + "\\BASIS_BR",
            'D:F':       base_input + "\\D:F",
            'B:F':       base_input + "\\B:F",
            'MASS_D':    base_input + "\\MASS_D",
            'MASS_B':    base_input + "\\MASS_B",
            'MASS_RR':   base_input + "\\MASS_RR",
            'MASS_BR':   base_input + "\\MASS_BR",
        }

        for label, path in input_paths.items():
            try:
                node = self.aspen.Tree.FindNode(path)
                if node is not None:
                    val = node.Value
                    result['input_values'][label] = val
                    marker = " <-- CURRENT" if val is not None and val != 0 else ""
                    logger.info("  {} = {}{}".format(label, val, marker))
                else:
                    result['input_values'][label] = None
            except Exception as e:
                result['input_values'][label] = "ERROR: {}".format(e)
                logger.debug("  {} -> error: {}".format(label, e))

        # ────────────────────────────────────────────────────────────
        # SECTION 3: Output Values (Actual solved values)
        # ────────────────────────────────────────────────────────────
        logger.info("")
        logger.info("--- SECTION 3: Output Values (Solved) ---")

        output_paths = {
            'MOLE_RR':   base_output + "\\MOLE_RR",
            'MOLE_DFR':  base_output + "\\MOLE_DFR",
            'MOLE_BFR':  base_output + "\\MOLE_BFR",
            'MOLE_BR':   base_output + "\\MOLE_BR",
            'MASS_RR':   base_output + "\\MASS_RR",
        }

        # Also check VAR_VAL outputs (Vary block solved values)
        for i in range(1, 6):
            output_paths['VAR_VAL\\{}'.format(i)] = base_output + "\\VAR_VAL\\{}".format(i)

        for label, path in output_paths.items():
            try:
                node = self.aspen.Tree.FindNode(path)
                if node is not None:
                    val = node.Value
                    result['output_values'][label] = val
                    if val is not None:
                        logger.info("  {} = {}".format(label, val))
                else:
                    result['output_values'][label] = None
            except Exception as e:
                result['output_values'][label] = "ERROR: {}".format(e)
                logger.debug("  {} -> error: {}".format(label, e))

        # ────────────────────────────────────────────────────────────
        # SECTION 4: Vary Block Variables (Tree Walk)
        # ────────────────────────────────────────────────────────────
        logger.info("")
        logger.info("--- SECTION 4: Vary Block Variables ---")

        for vary_num in range(1, max_vary + 1):
            vary_base = "\\Data\\Blocks\\{}\\Subobjects\\Vary\\{}".format(
                block_name, vary_num)

            try:
                vary_node = self.aspen.Tree.FindNode(vary_base)
                if vary_node is None:
                    continue
            except:
                continue

            vary_info = {
                'vary_num': vary_num,
                'variable_paths': {},
                'top_level_children': [],
                'input_children_limited': [],
            }

            logger.info("  Vary Block {}:".format(vary_num))

            # Step A: Walk Vary\N directly to see its top-level children
            try:
                if hasattr(vary_node, 'Elements') and vary_node.Elements.Count > 0:
                    count = vary_node.Elements.Count
                    logger.info("    Top-level children of Vary\\{} ({} nodes):".format(
                        vary_num, count))
                    for i in range(count):
                        try:
                            child = vary_node.Elements.Item(i)
                            child_name = child.Name if hasattr(child, 'Name') else "item_{}".format(i)
                            child_val = None
                            try:
                                child_val = child.Value
                            except:
                                child_val = "<no value>"
                            # Also count sub-children
                            sub_count = 0
                            try:
                                if hasattr(child, 'Elements'):
                                    sub_count = child.Elements.Count
                            except:
                                pass
                            vary_info['top_level_children'].append({
                                'name': child_name,
                                'value': child_val,
                                'sub_count': sub_count
                            })
                            logger.info("      [{}] {} = {} (sub-children: {})".format(
                                i, child_name, child_val, sub_count))
                        except Exception as e:
                            logger.debug("      [{}] error: {}".format(i, e))
            except Exception as e:
                logger.debug("    Could not walk Vary node: {}".format(e))

            # Step B: Try known variable path patterns
            variable_paths = {
                'SENTENCE\\VARIABLE':    vary_base + "\\Input\\SENTENCE\\VARIABLE",
                'SENTENCE\\VARIABLE\\1': vary_base + "\\Input\\SENTENCE\\VARIABLE\\1",
                'OPT_VAR':               vary_base + "\\Input\\OPT_VAR",
                'VARY_PARAM':            vary_base + "\\Input\\VARY_PARAM",
                'VARIABLE':              vary_base + "\\Input\\VARIABLE",
                'SENTENCE':              vary_base + "\\Input\\SENTENCE",
            }

            for label, path in variable_paths.items():
                try:
                    node = self.aspen.Tree.FindNode(path)
                    if node is not None:
                        val = node.Value
                        vary_info['variable_paths'][label] = val
                        if val is not None:
                            logger.info("    {} = {}".format(label, val))
                except Exception as e:
                    vary_info['variable_paths'][label] = "ERROR: {}".format(e)

            # Step C: Walk Vary\N\Input but LIMIT to first 30 children
            # (Previous bug: this returned 5000+ items = entire block Input)
            vary_input_path = vary_base + "\\Input"
            try:
                input_node = self.aspen.Tree.FindNode(vary_input_path)
                if input_node and hasattr(input_node, 'Elements'):
                    count = input_node.Elements.Count
                    logger.info("    Vary\\{}\\Input has {} children".format(
                        vary_num, count))
                    if count > 100:
                        logger.warning("    !!! {} children = likely resolves to Block Input (BUG) !!!".format(count))
                        vary_info['input_note'] = "SKIPPED: {} children detected (block Input alias)".format(count)
                    else:
                        for i in range(min(count, 30)):
                            try:
                                child = input_node.Elements.Item(i)
                                child_name = child.Name if hasattr(child, 'Name') else "item_{}".format(i)
                                child_val = None
                                try:
                                    child_val = child.Value
                                except:
                                    child_val = "<no value>"
                                vary_info['input_children_limited'].append({
                                    'name': child_name,
                                    'value': child_val
                                })
                                logger.info("      [{}] {} = {}".format(i, child_name, child_val))
                            except Exception as e:
                                logger.debug("      [{}] error: {}".format(i, e))
            except Exception as e:
                logger.debug("    Could not walk Vary Input tree: {}".format(e))

            # Step D: Try Design-Spec-nested Vary paths
            # In some Aspen versions, Vary config is under Design Spec subobjects
            ds_vary_paths = {}
            for ds_num in range(1, max_specs + 1):
                for suffix in ['SENTENCE\\VARIABLE', 'VARIABLE', 'OPT_VAR',
                               'LOWER', 'UPPER', 'MANIPULATED']:
                    key = "DS{}\\Vary{}\\{}".format(ds_num, vary_num, suffix)
                    path = "\\Data\\Blocks\\{}\\Subobjects\\Design Spec\\{}\\Subobjects\\Vary\\{}\\Input\\{}".format(
                        block_name, ds_num, vary_num, suffix)
                    ds_vary_paths[key] = path

            found_ds_vary = {}
            for label, path in ds_vary_paths.items():
                try:
                    node = self.aspen.Tree.FindNode(path)
                    if node is not None:
                        val = node.Value
                        if val is not None:
                            found_ds_vary[label] = val
                            logger.info("    {} = {}".format(label, val))
                except:
                    pass
            if found_ds_vary:
                vary_info['design_spec_vary_paths'] = found_ds_vary

            # Step E: Get bounds (try both Vary-level and block-level)
            for extra_label in ['LOWER', 'UPPER']:
                for suffix in ["\\Input\\{}".format(extra_label),
                                "\\{}".format(extra_label)]:
                    try:
                        node = self.aspen.Tree.FindNode(vary_base + suffix)
                        if node is not None and node.Value is not None:
                            vary_info[extra_label.lower()] = node.Value
                            logger.info("    {} = {} (from {})".format(
                                extra_label, node.Value, suffix))
                            break
                    except:
                        pass

            # Step F: Check VARY_ACTIVE from block Input (not Vary\Input)
            for spec_num in range(1, max_specs + 1):
                active_path = base_input + "\\VARY_ACTIVE\\{}".format(spec_num)
                try:
                    node = self.aspen.Tree.FindNode(active_path)
                    if node is not None and node.Value is not None:
                        vary_info['active_for_spec_{}'.format(spec_num)] = node.Value
                        logger.info("    VARY_ACTIVE\\{} = {}".format(spec_num, node.Value))
                except:
                    pass

            result['vary_blocks'].append(vary_info)

        # SECTION 4b: Walk top-level Subobjects to discover all subobject types
        logger.info("")
        logger.info("--- SECTION 4b: All Block Subobjects ---")
        subobj_path = "\\Data\\Blocks\\{}\\Subobjects".format(block_name)
        try:
            subobj_node = self.aspen.Tree.FindNode(subobj_path)
            if subobj_node and hasattr(subobj_node, 'Elements'):
                count = subobj_node.Elements.Count
                logger.info("  Subobjects ({} types):".format(count))
                result['subobject_types'] = []
                for i in range(count):
                    try:
                        child = subobj_node.Elements.Item(i)
                        child_name = child.Name if hasattr(child, 'Name') else "type_{}".format(i)
                        sub_count = 0
                        try:
                            if hasattr(child, 'Elements'):
                                sub_count = child.Elements.Count
                        except:
                            pass
                        result['subobject_types'].append({
                            'name': child_name,
                            'count': sub_count
                        })
                        logger.info("    [{}] {} ({} items)".format(i, child_name, sub_count))
                    except Exception as e:
                        logger.debug("    [{}] error: {}".format(i, e))
        except Exception as e:
            logger.debug("  Could not walk Subobjects: {}".format(e))

        # ────────────────────────────────────────────────────────────
        # SECTION 5: Design Spec Summary
        # ────────────────────────────────────────────────────────────
        logger.info("")
        logger.info("--- SECTION 5: Design Specs ---")

        for spec_num in range(1, max_specs + 1):
            spec_info = self.get_design_spec_info(block_name, spec_num)
            if spec_info:
                spec_type = self.get_design_spec_type(block_name, spec_num)
                spec_info['spec_type'] = spec_type
                result['design_specs'].append(spec_info)
                logger.info("  Spec {}: active={}, target={}, type={}".format(
                    spec_num, spec_info.get('active'), spec_info.get('target'),
                    spec_type))

        # ────────────────────────────────────────────────────────────
        # SECTION 6: Analysis & Warnings
        # ────────────────────────────────────────────────────────────
        logger.info("")
        logger.info("--- SECTION 6: Analysis ---")

        spec_indicators = result['operating_spec']

        # Step 1: Determine condenser operating spec from CON_SPEC
        # (CALC_MODE is calculation mode, NOT operating spec — exclude it)
        con_spec = spec_indicators.get('CON_SPEC')
        opt_spec_1 = spec_indicators.get('OPT_SPEC\\1')
        spec_1 = spec_indicators.get('SPEC\\1')
        calc_mode = spec_indicators.get('CALC_MODE')

        active_spec = None
        active_spec_source = None

        # Priority: CON_SPEC > OPT_SPEC\1 > SPEC\1  (NOT CALC_MODE)
        for key, val in [('CON_SPEC', con_spec), ('OPT_SPEC\\1', opt_spec_1),
                         ('SPEC\\1', spec_1)]:
            if val is not None and val != '' and not str(val).startswith('ERROR'):
                active_spec = str(val)
                active_spec_source = key
                break

        result['active_spec_type'] = active_spec
        result['active_spec_source'] = active_spec_source
        result['calc_mode'] = calc_mode

        if active_spec:
            logger.info("  Condenser spec: {} (from {})".format(active_spec, active_spec_source))
        else:
            logger.info("  CON_SPEC is null/empty — Aspen uses default condenser spec")
            logger.info("  Default for RadFrac with condenser = Reflux Ratio (RR)")

        logger.info("  Calculation mode: {}".format(calc_mode))

        # Step 2: Determine if BASIS_RR is the active variable
        # Use multiple evidence sources, not just spec name matching
        rr_keywords = ['RR', 'MOLE-RR', 'REFLUX', 'MOLRR']
        non_rr_keywords = ['D:F', 'D/F', 'DIST', 'MOLE-D', 'MOLE-B',
                           'BOILUP', 'BR', 'MOLE-BR', 'BOTTOMS']

        basis_rr_active = None

        if active_spec:
            spec_upper = active_spec.upper()
            if any(kw in spec_upper for kw in rr_keywords):
                basis_rr_active = True
            elif any(kw in spec_upper for kw in non_rr_keywords):
                basis_rr_active = False

        # Step 3: Cross-check with RR input/output mismatch
        basis_rr_input = result['input_values'].get('BASIS_RR')
        mole_rr_output = result['output_values'].get('MOLE_RR')
        rr_matches = None

        if (basis_rr_input is not None and mole_rr_output is not None
                and basis_rr_input != 0 and mole_rr_output != 0):
            try:
                rr_ratio = mole_rr_output / basis_rr_input
                rr_matches = abs(rr_ratio - 1.0) <= 0.1
                result['rr_input_output_ratio'] = round(rr_ratio, 4)

                if rr_matches:
                    logger.info("  RR consistent: Input={:.3f}, Output={:.3f} (ratio={:.2f})".format(
                        basis_rr_input, mole_rr_output, rr_ratio))
                else:
                    logger.warning("  RR MISMATCH: Input BASIS_RR={:.3f}, Output MOLE_RR={:.3f} (ratio={:.2f})".format(
                        basis_rr_input, mole_rr_output, rr_ratio))
            except (TypeError, ZeroDivisionError):
                pass

        # Step 4: Final determination using combined evidence
        # If CON_SPEC is null AND RR matches → default spec = RR → BASIS_RR works
        # If CON_SPEC is null AND RR mismatches → Design Spec Vary block changed it
        # If CON_SPEC is explicit non-RR → BASIS_RR won't work
        if basis_rr_active is None and active_spec is None:
            # CON_SPEC is null — check if Design Specs are active with Vary blocks
            active_design_specs = [ds for ds in result['design_specs']
                                   if ds.get('active') == 'YES']
            active_vary_blocks = [vb for vb in result['vary_blocks']
                                  if vb.get('active_for_spec_1') == 'YES'
                                  or vb.get('active_for_spec_2') == 'YES']

            if active_design_specs and active_vary_blocks:
                # Design Specs + Vary blocks are active
                # The column's default spec is RR (since CON_SPEC is null)
                # BUT: Design Spec Vary blocks override BASIS_RR during convergence
                # In forward mode (DS OFF), BASIS_RR SHOULD work
                basis_rr_active = True
                result['warnings'].append(
                    "CON_SPEC is null (default=RR). {} active Design Specs with {} Vary blocks. "
                    "In Design Spec mode, Vary blocks override BASIS_RR. "
                    "In forward mode (DS OFF), BASIS_RR should be the active spec.".format(
                        len(active_design_specs), len(active_vary_blocks)))
                logger.info("  Default spec is RR. Design Specs ({}) with Vary blocks ({}) override it.".format(
                    len(active_design_specs), len(active_vary_blocks)))
                if rr_matches is False:
                    logger.info("  RR mismatch confirms Vary block adjusted RR from {:.3f} to {:.3f}".format(
                        basis_rr_input, mole_rr_output))
            elif rr_matches is True:
                basis_rr_active = True
                logger.info("  RR input/output match confirms BASIS_RR is the active spec")
            elif rr_matches is False:
                basis_rr_active = False
                logger.warning("  RR input/output mismatch AND no clear spec type — BASIS_RR may be ignored")

        result['basis_rr_is_active'] = basis_rr_active

        # Step 5: Summary
        if basis_rr_active is True:
            logger.info("  CONCLUSION: BASIS_RR IS the active operating spec")
            logger.info("  set_reflux_ratio() and sweep_rr_purity() should work in forward mode")
        elif basis_rr_active is False:
            spec_name = active_spec if active_spec else "UNKNOWN"
            logger.warning("  CONCLUSION: BASIS_RR is NOT the active spec (spec='{}')".format(spec_name))
            logger.warning("  Writing to BASIS_RR will have NO EFFECT")
            logger.warning("  sweep_rr_purity and _evaluate_with_fixed_rr WILL FAIL")
            result['warnings'].append(
                "BASIS_RR is NOT the active operating spec. "
                "Active spec is '{}'. Writing to BASIS_RR will have no effect.".format(spec_name))
        else:
            logger.warning("  CONCLUSION: Cannot determine if BASIS_RR is active")
            result['warnings'].append(
                "Cannot determine if BASIS_RR is the active spec. "
                "Check Aspen flowsheet manually.")

        logger.info("")
        logger.info("=" * 60)
        logger.info("END DIAGNOSTIC")
        logger.info("=" * 60)

        # Save diagnostic to JSON file for post-mortem analysis
        try:
            import json, os
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            diag_path = os.path.join(results_dir, 'column_diag_{}.json'.format(block_name))
            with open(diag_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info("Diagnostic saved to: {}".format(diag_path))
        except Exception as e:
            logger.warning("Could not save diagnostic to file: {}".format(e))

        return result

    def sweep_rr_purity(self, block_name, nt, feed, pressure, feed_stream,
                        rr_range=(0.5, 4.0), num_points=15,
                        purity_spec=None, original_rr=None,
                        column_diag=None):
        """
        Sweep RR values and record achieved purity at each point.

        This runs the column in "forward mode":
        - Turn off ALL Design Specs
        - RR = independent variable (we set it)
        - Purity = dependent result (we read it)
        - Column structure fixed: NT, NF, Pressure

        The resulting RR-Purity curve reveals:
        1. Minimum reflux / pinch region (steep behavior at low RR)
        2. Feasible operating window (smooth, converged region)
        3. Diminishing returns region (purity asymptotes at high RR)

        Parameters
        ----------
        block_name : str
            Name of the RadFrac block
        nt : int
            Number of stages (fixed)
        feed : int
            Feed stage location (fixed)
        pressure : float
            Column pressure in bar (fixed)
        feed_stream : str
            Feed stream name
        rr_range : tuple
            (min_rr, max_rr) range to sweep (default: 0.5 to 4.0)
        num_points : int
            Number of points in sweep (default: 15)
        purity_spec : dict
            From config.PURITY_SPECS - must contain:
            - 'stream': Product stream name (e.g., 'TOLL', 'EBB')
            - 'component': Key component name (e.g., 'STYRE-01', 'TOLUE-01')
            - 'fraction_type': 'MASSFRAC' or 'MOLEFRAC'
            - 'target': Target purity value (optional)

        Returns
        -------
        list : List of dicts with keys: 'rr', 'purity', 'converged'
        """
        import numpy as np

        # Extract purity spec parameters
        if purity_spec is None:
            logger.error("  ERROR: purity_spec is required for RR sweep")
            logger.error("  Use config.get_purity_spec(case_name) to get the spec")
            return []

        # Guard: check if BASIS_RR is actually the active operating spec
        if column_diag and column_diag.get('basis_rr_is_active') is False:
            logger.error("  !!! ABORTING RR SWEEP: Column uses '{}', not reflux ratio !!!".format(
                column_diag.get('active_spec_type')))
            logger.error("  Writing to BASIS_RR will have no effect.")
            logger.error("  Check Aspen flowsheet: RadFrac > Design Specs > Vary to see what variable is used.")
            return []

        product_stream = purity_spec.get('stream')
        key_component = purity_spec.get('component')
        fraction_type = purity_spec.get('fraction_type', 'MASSFRAC')
        spec_target = purity_spec.get('target')
        is_middle_split = purity_spec.get('is_middle_split', False)
        top_components = purity_spec.get('top_components', [])

        # Check for middle split columns (no single key component)
        if is_middle_split:
            if top_components:
                logger.info("  Middle split column - checking combined purity of: {}".format(top_components))
            else:
                logger.warning("  Middle split column with no top_components defined")
                return []
        else:
            if not product_stream or not key_component:
                logger.error("  ERROR: purity_spec missing 'stream' or 'component'")
                return []

        logger.info("=" * 50)
        logger.info("RR SWEEP DIAGNOSTIC (Forward Mode)")
        logger.info("=" * 50)
        logger.info("  Config: NT={}, NF={}, P={:.3f} bar".format(nt, feed, pressure))
        if is_middle_split:
            logger.info("  Stream: {}, Components: {}, Type: {}".format(
                product_stream, top_components, fraction_type))
        else:
            logger.info("  Stream: {}, Component: {}, Type: {}".format(
                product_stream, key_component, fraction_type))

        # 1. Get calculated RR from VAR_VAL BEFORE turning off Design Specs
        calculated_rr = self.get_output_rr(block_name)
        MAX_SANE_RR = 200  # No real distillation column operates at RR > 200
        if calculated_rr and 0 < calculated_rr < MAX_SANE_RR:
            logger.info("  Calculated RR from Vary block: {:.3f}".format(calculated_rr))
            # Use calculated RR as center point for smart sweep range
            # Sweep ±50% around calculated RR
            smart_rr_min = max(0.3, calculated_rr * 0.5)
            smart_rr_max = calculated_rr * 1.5
            rr_range = (smart_rr_min, smart_rr_max)
            logger.info("  Smart sweep range: {:.2f} to {:.2f} (±50% of calculated RR)".format(
                smart_rr_min, smart_rr_max))
        else:
            if calculated_rr and calculated_rr >= MAX_SANE_RR:
                logger.warning("  Calculated RR={:.1f} is garbage (>{}), ignoring. Using default range: {:.2f} to {:.2f}".format(
                    calculated_rr, MAX_SANE_RR, rr_range[0], rr_range[1]))
            else:
                logger.info("  No calculated RR found, using default range: {:.2f} to {:.2f}".format(
                    rr_range[0], rr_range[1]))

        # 2. Get ALL active Design Specs and turn them OFF
        active_specs = self.get_all_design_specs(block_name, max_specs=5)
        active_spec_nums = [s['spec_num'] for s in active_specs]
        # Use purity spec target if provided, otherwise try to get from Design Spec
        target_purity = spec_target if spec_target else (
            active_specs[0].get('target') if active_specs else None)

        if active_specs:
            logger.info("  Found {} active Design Specs: {}".format(
                len(active_specs), active_spec_nums))
            logger.info("  Turning OFF all Design Specs AND Vary blocks for forward mode...")

        # Turn off all Design Specs
        for spec_num in active_spec_nums:
            self.set_design_spec_active(block_name, spec_num=spec_num, active=False)

        # Turn off ALL Vary blocks (discover actual Vary-Spec associations)
        all_vary_blocks = self.get_all_vary_blocks(block_name)
        if all_vary_blocks:
            logger.info("  Found {} Vary blocks: {}".format(
                len(all_vary_blocks),
                [(vb['vary_num'], vb['spec_num']) for vb in all_vary_blocks]))
        for vb in all_vary_blocks:
            self.set_vary_active(block_name, vary_num=vb['vary_num'],
                                 spec_num=vb['spec_num'], active=False)

        # 3. Set column parameters (NT, NF, P)
        self.set_parameters(block_name, nt, feed, pressure, feed_stream)

        # 4. Generate RR values to sweep
        rr_values = np.linspace(rr_range[0], rr_range[1], num_points)
        results = []

        logger.info("")
        logger.info("  Sweeping RR from {:.2f} to {:.2f} ({} points)".format(
            rr_range[0], rr_range[1], num_points))
        logger.info("  " + "-" * 40)

        for i, rr in enumerate(rr_values):
            # Set RR
            self.set_reflux_ratio(block_name, rr)

            # Run simulation
            success = self.run_simulation()

            purity = None
            converged = False

            if success:
                try:
                    # Check BLKSTAT for convergence info (but don't gate purity reading on it)
                    block_path = "\\Data\\Blocks\\{}\\Output\\BLKSTAT".format(block_name)
                    node = self.aspen.Tree.FindNode(block_path)
                    status = node.Value if node else None

                    if status == 0 or (isinstance(status, str) and 'OK' in str(status).upper()):
                        converged = True
                    else:
                        logger.debug("  BLKSTAT={} (not 0/OK, but will still read stream purity)".format(status))

                    # Always read purity from stream output after successful simulation
                    # In forward mode (Design Specs OFF), BLKSTAT may return non-zero
                    # (warnings) but the simulation still produces valid stream results
                    if is_middle_split:
                        purity = self.get_stream_multi_purity(product_stream, top_components, fraction_type)
                    else:
                        purity = self.get_stream_purity(product_stream, key_component, fraction_type)

                    # If we got a valid purity, mark as converged regardless of BLKSTAT
                    if purity is not None and purity > 0:
                        converged = True
                except Exception as e:
                    logger.warning("  Error reading results: {}".format(e))

            results.append({
                'rr': rr,
                'purity': purity,
                'converged': converged
            })

            # Log result
            status_str = "OK" if converged else "FAIL"
            purity_str = "{:.4f}".format(purity) if purity else "N/A"
            logger.info("  [{:2d}/{:2d}] RR={:.2f} -> Purity={} [{}]".format(
                i+1, num_points, rr, purity_str, status_str))

        # 4. Restore Design Specs AND Vary blocks
        logger.info("")
        for spec_num in active_spec_nums:
            self.set_design_spec_active(block_name, spec_num=spec_num, active=True)
        # Restore all Vary blocks that were originally active
        for vb in all_vary_blocks:
            if vb['active'] == 'YES':
                self.set_vary_active(block_name, vary_num=vb['vary_num'],
                                     spec_num=vb['spec_num'], active=True)
        if active_spec_nums or all_vary_blocks:
            logger.info("  Restored {} Design Specs and {} Vary blocks to ACTIVE".format(
                len(active_spec_nums), len([vb for vb in all_vary_blocks if vb['active'] == 'YES'])))

        # 4b. Restore original BASIS_RR to prevent contamination of next evaluation
        if original_rr is not None:
            self.set_reflux_ratio(block_name, original_rr)
            logger.info("  Restored BASIS_RR to {:.3f}".format(original_rr))

        # 5. Analyze results
        converged_points = [r for r in results if r['converged']]
        if converged_points:
            purities = [r['purity'] for r in converged_points]
            max_purity = max(purities)
            logger.info("")
            logger.info("  SUMMARY:")
            logger.info("  " + "-" * 40)
            logger.info("  Converged points: {}/{}".format(len(converged_points), num_points))
            logger.info("  Max achievable purity: {:.4f}".format(max_purity))

            if target_purity:
                logger.info("  Target purity: {:.4f}".format(target_purity))
                if max_purity >= target_purity:
                    # Find minimum RR to achieve target
                    for r in converged_points:
                        if r['purity'] and r['purity'] >= target_purity:
                            logger.info("  Min RR for target: {:.2f}".format(r['rr']))
                            logger.info("  >>> TARGET IS ACHIEVABLE")
                            break
                else:
                    logger.info("  >>> TARGET NOT ACHIEVABLE at this config!")
                    logger.info("  >>> Need more stages or different feed location")

        logger.info("=" * 50)

        return results

    def close(self):
        """Close Aspen connection"""
        try:
            if self.aspen:
                self.aspen.Close()
                self.aspen = None
                self.connected = False
                logger.info("Aspen closed")
        except:
            pass