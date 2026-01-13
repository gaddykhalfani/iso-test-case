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
        try:
            logger.info("Connecting to Aspen Plus...")
            self.aspen = win32.gencache.EnsureDispatch("Apwn.Document")
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
            logger.error("Failed to connect: {}".format(e))
            return False
    
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
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Input\\PRES1".format(block_name))
            node.Value = pressure
            logger.info("  {}.PRES1 = {} bar".format(block_name, pressure))
            
            return True
            
        except Exception as e:
            logger.error("Error setting parameters: {}".format(e))
            return False
    
    def run_simulation(self):
        """Run Aspen simulation"""
        try:
            if not self.is_connected():
                logger.warning("Connection lost before simulation")
                return False
            
            logger.info("Running simulation...")
            start_time = time.time()
            
            self.aspen.Engine.Run2()
            
            elapsed = time.time() - start_time
            logger.info("Simulation completed in {:.1f}s".format(elapsed))
            
            return True
            
        except Exception as e:
            error_str = str(e)
            if "RPC" in error_str or "remote procedure" in error_str.lower():
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
    
    def get_reflux_ratio(self, block_name):
        """Get actual reflux ratio"""
        try:
            node = self.aspen.Tree.FindNode(
                "\\Data\\Blocks\\{}\\Output\\MOLE_RR".format(block_name))
            return node.Value if node and node.Value else 0.0
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