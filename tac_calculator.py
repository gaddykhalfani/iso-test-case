# tac_calculator_seider.py
"""
TAC Calculator for Distillation Columns (v3.1 - Seider Vacuum Model)
====================================================================
Updated vacuum system sizing and costing using Seider et al. correlations.

Key Changes from v3.0:
----------------------
1. Air leakage: Seider Eq. 22.73 (volume-based correlation)
2. Vacuum equipment: Table 22.32 cost correlations
3. Auto-selection of vacuum system type
4. Removed reflux drum (not needed for TAC optimization)

References:
-----------
1. Seider, Seader, Lewin, Widagdo - Product and Process Design Principles
   - Eq. 22.73: Air leakage correlation
   - Table 22.32: Vacuum equipment cost correlations
2. Turton et al. (2018) - Equipment cost correlations
3. Towler & Sinnott (2021) - Chemical Engineering Design

Author: PSE Lab, NTUST
Date: December 2024
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CoolingUtility:
    """Cooling utility properties"""
    name: str
    T_in: float          # Inlet temperature (°C)
    T_out: float         # Outlet temperature (°C)
    cost_per_GJ: float   # Cost ($/GJ)
    U_typical: float     # Typical U for condensing HC (W/m²·K)
    

@dataclass
class HeatingUtility:
    """Heating utility properties"""
    name: str
    T_steam: float       # Steam/hot fluid temperature (°C)
    cost_per_GJ: float   # Cost ($/GJ)
    U_typical: float     # Typical U for boiling HC (W/m²·K)


@dataclass
class VacuumEquipment:
    """Vacuum equipment from Seider Table 22.32"""
    name: str
    size_type: str           # 'ejector' or 'volumetric'
    min_size: float          # Minimum size factor
    max_size: float          # Maximum size factor
    cost_coef: float         # Cost coefficient (a in Cp = a * S^b)
    cost_exp: float          # Cost exponent (b in Cp = a * S^b)
    steam_ratio: float       # kg steam / kg air (for ejectors)
    power_factor: float      # kW per ft³/min (for pumps)
    min_pressure_torr: float # Minimum suction pressure (torr)
    max_pressure_torr: float # Maximum suction pressure (torr)


# ════════════════════════════════════════════════════════════════════════════
# VACUUM EQUIPMENT DATABASE (Seider Table 22.32)
# ════════════════════════════════════════════════════════════════════════════

VACUUM_EQUIPMENT = {
    'one_stage_ejector': VacuumEquipment(
        name='One-Stage Steam Jet Ejector',
        size_type='ejector',           # S = (lb/hr) / (suction pressure, torr)
        min_size=0.1,
        max_size=100.0,
        cost_coef=1690,
        cost_exp=0.41,
        steam_ratio=10.0,              # ~10 lb steam per lb air (from example)
        power_factor=0.0,
        min_pressure_torr=100,         # Practical minimum for 1-stage
        max_pressure_torr=760,
    ),
    'liquid_ring_pump': VacuumEquipment(
        name='Liquid Ring Vacuum Pump',
        size_type='volumetric',        # S = ft³/min at suction
        min_size=50,
        max_size=350,
        cost_coef=8250,
        cost_exp=0.35,
        steam_ratio=0.0,
        power_factor=0.05,             # ~0.05 kW per ft³/min (from example: 12.6 kW for 272 ft³/min)
        min_pressure_torr=50,
        max_pressure_torr=400,
    ),
    'three_stage_lobe': VacuumEquipment(
        name='Three-Stage Lobe Blower',
        size_type='volumetric',
        min_size=60,
        max_size=240,
        cost_coef=7120,
        cost_exp=0.41,
        steam_ratio=0.0,
        power_factor=0.06,
        min_pressure_torr=10,
        max_pressure_torr=100,
    ),
    'three_stage_claw': VacuumEquipment(
        name='Three-Stage Claw Pump',
        size_type='volumetric',
        min_size=60,
        max_size=270,
        cost_coef=8630,
        cost_exp=0.36,
        steam_ratio=0.0,
        power_factor=0.055,
        min_pressure_torr=10,
        max_pressure_torr=100,
    ),
    'screw_compressor': VacuumEquipment(
        name='Screw Compressor',
        size_type='volumetric',
        min_size=50,
        max_size=350,
        cost_coef=9590,
        cost_exp=0.38,
        steam_ratio=0.0,
        power_factor=0.045,
        min_pressure_torr=20,
        max_pressure_torr=200,
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# UTILITY DATABASE
# ════════════════════════════════════════════════════════════════════════════

COOLING_UTILITIES = {
    'CW': CoolingUtility(
        name='Cooling Water',
        T_in=30.0,
        T_out=40.0,
        cost_per_GJ=0.354,
        U_typical=500.0
    ),
    'ChW': CoolingUtility(
        name='Chilled Water',
        T_in=5.0,
        T_out=15.0,
        cost_per_GJ=4.43,
        U_typical=400.0
    ),
    'Refrig': CoolingUtility(
        name='Refrigerant (-20°C)',
        T_in=-20.0,
        T_out=-10.0,
        cost_per_GJ=7.89,
        U_typical=350.0
    ),
    'Brine': CoolingUtility(
        name='Brine (-40°C)',
        T_in=-40.0,
        T_out=-30.0,
        cost_per_GJ=13.11,
        U_typical=300.0
    ),
}

HEATING_UTILITIES = {
    'LPS': HeatingUtility(
        name='LP Steam (4 bar)',
        T_steam=144.0,
        cost_per_GJ=7.78,
        U_typical=900.0
    ),
    'MPS': HeatingUtility(
        name='MP Steam (10 bar)',
        T_steam=184.0,
        cost_per_GJ=8.22,
        U_typical=850.0
    ),
    'HPS': HeatingUtility(
        name='HP Steam (40 bar)',
        T_steam=250.0,
        cost_per_GJ=9.83,
        U_typical=800.0
    ),
    'HotOil': HeatingUtility(
        name='Hot Oil (Therminol)',
        T_steam=300.0,
        cost_per_GJ=12.0,
        U_typical=300.0
    ),
    'Fired': HeatingUtility(
        name='Fired Heater',
        T_steam=400.0,
        cost_per_GJ=15.0,
        U_typical=100.0
    ),
}


class TACCalculator:
    """
    Calculate Total Annual Cost (TAC) for distillation columns.
    
    Version 3.1 with Seider vacuum system correlations.
    
    TAC = TPC/payback_period + TOC
    
    where:
    - TPC includes: column, trays, condenser, reboiler, vacuum system
    - TOC includes: steam, cooling water, vacuum system operating cost
    """
    
    # Minimum approach temperature (pinch)
    DELTA_T_MIN = 10.0  # °C
    
    def __init__(self, 
                 tray_spacing=0.6096,         # m (2 ft standard)
                 payback_period=3,            # years
                 ms_index=2221.5,             # Marshall & Swift Index (2025)
                 cepci=800,                   # CEPCI for equipment costs (2024)
                 cepci_base=500,              # Base CEPCI (Seider uses 500)
                 operating_hours=8000,        # hours/year
                 material='SS',               # 'CS' or 'SS'
                 steam_cost_per_1000lb=5.0,   # $/1000 lb steam
                 electricity_cost_kWh=0.05):  # $/kWh
        """
        Initialize TAC calculator with Seider correlations.
        """
        self.tray_spacing = tray_spacing
        self.tray_spacing_ft = tray_spacing / 0.3048
        self.payback_period = payback_period
        
        self.ms_index = ms_index
        self.ms_base = 280
        self.cepci = cepci
        self.cepci_base = cepci_base
        
        self.operating_hours = operating_hours
        self.material = material
        
        # Operating cost parameters (from Seider example)
        self.steam_cost_per_1000lb = steam_cost_per_1000lb
        self.electricity_cost_kWh = electricity_cost_kWh
        
        # Store selected utilities (for reporting)
        self.selected_cooling = None
        self.selected_heating = None
    
    def calculate(self, nt, diameter, Q_reb, Q_cond, pressure=0.2,
                  T_cond=None, T_reb=None, column_height=None):
        """
        Calculate TAC for given column configuration.
        
        Parameters:
        -----------
        nt : int
            Number of theoretical stages (including condenser and reboiler)
        diameter : float
            Column diameter in meters
        Q_reb : float
            Reboiler duty in kW
        Q_cond : float
            Condenser duty in kW (absolute value)
        pressure : float
            Operating pressure in bar
        T_cond : float, optional
            Condenser temperature in °C
        T_reb : float, optional
            Reboiler temperature in °C
        column_height : float, optional
            Column height in meters (if not provided, calculated from nt)
            
        Returns:
        --------
        dict : Comprehensive results including all cost components
        """
        # Validate inputs
        if nt <= 0 or diameter <= 0:
            logger.warning(f"Invalid inputs: NT={nt}, D={diameter}")
            return self._failed_result()
        
        if Q_reb <= 0 or Q_cond <= 0:
            logger.warning(f"Invalid duties: Q_reb={Q_reb:.1f}, Q_cond={Q_cond:.1f}")
            return self._failed_result()
        
        # Estimate temperatures if not provided
        if T_cond is None:
            T_cond = self._estimate_condenser_temp(pressure)
        
        if T_reb is None:
            T_reb = self._estimate_reboiler_temp(pressure)
        
        n_trays = nt - 2
        if column_height is None:
            column_height = n_trays * self.tray_spacing + 3.5  # Add top/bottom
        
        # ════════════════════════════════════════════════════════════════════
        # CAPITAL COSTS
        # ════════════════════════════════════════════════════════════════════
        
        # 1. Column + trays (Guthrie)
        column_cost = self._column_cost_guthrie(diameter, n_trays, pressure)
        
        # 2. Heat exchangers with AUTO UTILITY SELECTION
        cond_result = self._condenser_cost(Q_cond, T_cond)
        reb_result = self._reboiler_cost(Q_reb, T_reb)
        
        condenser_cost = cond_result['cost']
        reboiler_cost = reb_result['cost']
        
        # 3. Vacuum system (Seider method) - only if pressure < 0.9 atm
        # Use 0.9 bar as threshold to avoid unnecessary vacuum systems near atmospheric
        if pressure < 0.9:
            vacuum_result = self._size_and_cost_vacuum_system_seider(
                column_diameter=diameter,
                column_height=column_height,
                pressure_bar=pressure,
                T_cond=T_cond
            )
            vacuum_system_cost = vacuum_result['capital_cost']
            vacuum_operating_cost = vacuum_result['annual_operating_cost']
        else:
            vacuum_result = self._no_vacuum_system()
            vacuum_system_cost = 0
            vacuum_operating_cost = 0
        
        # Total Purchased Cost
        TPC = column_cost + condenser_cost + reboiler_cost + vacuum_system_cost
        
        # ════════════════════════════════════════════════════════════════════
        # OPERATING COSTS
        # ════════════════════════════════════════════════════════════════════
        
        # Condenser utility cost
        cooling_utility = cond_result['utility']
        Q_cond_GJ_yr = Q_cond * 3.6 * self.operating_hours / 1000
        cond_utility_cost = Q_cond_GJ_yr * cooling_utility.cost_per_GJ
        
        # Reboiler utility cost
        heating_utility = reb_result['utility']
        Q_reb_GJ_yr = Q_reb * 3.6 * self.operating_hours / 1000
        reb_utility_cost = Q_reb_GJ_yr * heating_utility.cost_per_GJ
        
        # Total Operating Cost
        TOC = cond_utility_cost + reb_utility_cost + vacuum_operating_cost
        
        # ════════════════════════════════════════════════════════════════════
        # TOTAL ANNUAL COST
        # ════════════════════════════════════════════════════════════════════
        TAC = TPC / self.payback_period + TOC
        
        # Store for reporting
        self.selected_cooling = cooling_utility
        self.selected_heating = heating_utility
        
        return {
            'TAC': TAC,
            'TPC': TPC,
            'TOC': TOC,
            
            # Capital cost breakdown
            'column_cost': column_cost,
            'tray_cost': 0,  # Included in column_cost
            'condenser_cost': condenser_cost,
            'reboiler_cost': reboiler_cost,
            'vacuum_system_cost': vacuum_system_cost,
            
            # Operating cost breakdown
            'steam_cost': reb_utility_cost,
            'cw_cost': cond_utility_cost,
            'vacuum_operating_cost': vacuum_operating_cost,
            
            # Column design
            'column_height': column_height,
            'n_trays': n_trays,
            
            # Condenser details
            'condenser': {
                'utility': cooling_utility.name,
                'T_cond': T_cond,
                'T_utility_in': cooling_utility.T_in,
                'T_utility_out': cooling_utility.T_out,
                'LMTD': cond_result['LMTD'],
                'U': cond_result['U'],
                'Area_m2': cond_result['area'],
                'cost_per_GJ': cooling_utility.cost_per_GJ,
                'annual_cost': cond_utility_cost,
            },
            
            # Reboiler details
            'reboiler': {
                'utility': heating_utility.name,
                'T_reb': T_reb,
                'T_utility': heating_utility.T_steam,
                'LMTD': reb_result['LMTD'],
                'U': reb_result['U'],
                'Area_m2': reb_result['area'],
                'cost_per_GJ': heating_utility.cost_per_GJ,
                'annual_cost': reb_utility_cost,
            },
            
            # Vacuum system details
            'vacuum_system': vacuum_result,
        }
    
    # ════════════════════════════════════════════════════════════════════════
    # VACUUM SYSTEM - SEIDER METHOD
    # ════════════════════════════════════════════════════════════════════════
    
    def _size_and_cost_vacuum_system_seider(self, column_diameter, column_height,
                                             pressure_bar, T_cond):
        """
        Size and cost vacuum system using Seider et al. correlations.
        
        Method:
        -------
        1. Calculate system volume (column + condenser + piping)
        2. Estimate air leakage using Eq. 22.73
        3. Calculate volumetric flow rate at suction conditions
        4. Auto-select appropriate vacuum equipment
        5. Calculate capital cost using Table 22.32
        6. Calculate operating cost (steam or electricity)
        
        References:
        -----------
        - Seider et al., Product and Process Design Principles
        - Eq. 22.73: W = 5 + {0.0298 + 0.03088[ln(P)] - 0.0005733[ln(P)]²}V^0.66
        - Table 22.32: Vacuum equipment cost correlations
        
        Parameters:
        -----------
        column_diameter : float
            Column diameter (m)
        column_height : float
            Column height (m)
        pressure_bar : float
            Operating pressure (bar abs)
        T_cond : float
            Condenser temperature (°C)
        
        Returns:
        --------
        dict : Vacuum system sizing, selection, and cost details
        """
        # ════════════════════════════════════════════════════════════════════
        # STEP 1: Calculate system volume
        # ════════════════════════════════════════════════════════════════════
        
        # Column volume
        V_column_m3 = math.pi * (column_diameter / 2)**2 * column_height
        
        # Total system volume (column + condenser + vapor line + drum)
        # From Seider example: these are all included
        # Typical: condenser ~10%, vapor line ~5%, drum ~10%
        V_system_m3 = V_column_m3 * 1.25
        
        # Convert to ft³ (1 m³ = 35.3147 ft³)
        V_system_ft3 = V_system_m3 * 35.3147
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 2: Calculate air leakage using Seider Eq. 22.73
        # ════════════════════════════════════════════════════════════════════
        
        # Convert pressure: bar to torr (1 bar = 750.062 torr)
        P_torr = pressure_bar * 750.062
        
        # Seider Eq. 22.73:
        # W = 5 + {0.0298 + 0.03088[ln(P)] - 0.0005733[ln(P)]²} × V^0.66
        # Units: W in lb/hr, P in torr, V in ft³
        
        if P_torr > 1:
            ln_P = math.log(P_torr)
            bracket = 0.0298 + 0.03088 * ln_P - 0.0005733 * ln_P**2
            W_air_lb_hr = 5 + bracket * (V_system_ft3 ** 0.66)
        else:
            # Very deep vacuum - use minimum correlation value
            W_air_lb_hr = 5 + 0.2 * (V_system_ft3 ** 0.66)
        
        # Apply reasonable bounds
        W_air_lb_hr = max(W_air_lb_hr, 5.0)      # Minimum from correlation
        W_air_lb_hr = min(W_air_lb_hr, 500.0)   # Practical maximum
        
        # Convert to kg/hr (1 lb = 0.4536 kg)
        W_air_kg_hr = W_air_lb_hr * 0.4536
        
        logger.info(f"  Vacuum system: V={V_system_m3:.1f} m³ ({V_system_ft3:.0f} ft³)")
        logger.info(f"  Air leakage (Seider Eq.22.73): {W_air_lb_hr:.1f} lb/hr ({W_air_kg_hr:.1f} kg/hr)")
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 3: Calculate volumetric flow rate at suction conditions
        # ════════════════════════════════════════════════════════════════════
        
        # Using ideal gas law: Q = n × R × T / P
        # For air: MW = 29 lb/lbmol
        # R = 10.73 psi·ft³/(lbmol·°R)
        
        T_suction_R = (T_cond + 273.15) * 1.8  # Convert °C to °R
        P_suction_psia = P_torr / 51.715       # Convert torr to psia
        
        n_air_lbmol_hr = W_air_lb_hr / 29.0
        
        # Q = n × R × T / P (ft³/hr)
        Q_ft3_hr = n_air_lbmol_hr * 10.73 * T_suction_R / P_suction_psia
        Q_ft3_min = Q_ft3_hr / 60
        
        logger.info(f"  Volumetric flow at suction: {Q_ft3_min:.1f} ft³/min")
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 4: Auto-select vacuum equipment
        # ════════════════════════════════════════════════════════════════════
        
        selected_equipment = self._select_vacuum_equipment(
            W_air_lb_hr, P_torr, Q_ft3_min
        )
        
        if selected_equipment is None:
            logger.warning("  No suitable vacuum equipment found!")
            return self._no_vacuum_system()
        
        logger.info(f"  Selected: {selected_equipment.name}")
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 5: Calculate capital cost using Table 22.32
        # ════════════════════════════════════════════════════════════════════
        
        # Calculate size factor
        if selected_equipment.size_type == 'ejector':
            # S = (lb/hr) / (suction pressure, torr)
            S = W_air_lb_hr / P_torr
            size_description = f"S = {W_air_lb_hr:.1f}/{P_torr:.0f} = {S:.2f} lb/hr-torr"
        else:
            # S = volumetric flow at suction (ft³/min)
            S = Q_ft3_min
            size_description = f"S = {S:.1f} ft³/min"
        
        # Ensure S is within equipment range
        S = max(S, selected_equipment.min_size)
        S = min(S, selected_equipment.max_size)
        
        # Calculate base cost: Cp = a × S^b
        Cp_base = selected_equipment.cost_coef * (S ** selected_equipment.cost_exp)
        
        # Apply CEPCI correction
        capital_cost = Cp_base * (self.cepci / self.cepci_base)
        
        # Minimum practical cost
        capital_cost = max(capital_cost, 2000)
        
        logger.info(f"  Size factor: {size_description}")
        logger.info(f"  Capital cost: ${capital_cost:,.0f}")
        
        # ════════════════════════════════════════════════════════════════════
        # STEP 6: Calculate operating cost
        # ════════════════════════════════════════════════════════════════════
        
        if selected_equipment.steam_ratio > 0:
            # Steam ejector: calculate steam consumption
            W_steam_lb_hr = W_air_lb_hr * selected_equipment.steam_ratio
            W_steam_kg_hr = W_steam_lb_hr * 0.4536
            
            # Annual steam cost: $/1000 lb × (lb/hr) × (hr/yr) / 1000
            annual_steam_cost = (self.steam_cost_per_1000lb * 
                                W_steam_lb_hr * 
                                self.operating_hours / 1000)
            
            annual_operating_cost = annual_steam_cost
            power_kW = 0
            
            logger.info(f"  Steam consumption: {W_steam_lb_hr:.0f} lb/hr ({W_steam_kg_hr:.1f} kg/hr)")
            logger.info(f"  Annual steam cost: ${annual_steam_cost:,.0f}/yr")
        else:
            # Mechanical pump: calculate power consumption
            power_kW = Q_ft3_min * selected_equipment.power_factor
            W_steam_lb_hr = 0
            W_steam_kg_hr = 0
            
            # Annual electricity cost
            annual_electricity_cost = (power_kW * 
                                       self.electricity_cost_kWh * 
                                       self.operating_hours)
            
            annual_operating_cost = annual_electricity_cost
            
            logger.info(f"  Power consumption: {power_kW:.1f} kW")
            logger.info(f"  Annual electricity cost: ${annual_electricity_cost:,.0f}/yr")
        
        return {
            'system_type': selected_equipment.name,
            'system_volume_m3': V_system_m3,
            'system_volume_ft3': V_system_ft3,
            'pressure_torr': P_torr,
            'air_leakage_lb_hr': W_air_lb_hr,
            'air_leakage_kg_hr': W_air_kg_hr,
            'volumetric_flow_ft3_min': Q_ft3_min,
            'size_factor': S,
            'steam_consumption_lb_hr': W_steam_lb_hr,
            'steam_consumption_kg_hr': W_steam_kg_hr,
            'power_kW': power_kW,
            'capital_cost': capital_cost,
            'annual_operating_cost': annual_operating_cost,
        }
    
    def _select_vacuum_equipment(self, W_air_lb_hr, P_torr, Q_ft3_min):
        """
        Auto-select appropriate vacuum equipment based on operating conditions.
        
        Selection Logic (based on Seider Table 22.30):
        ----------------------------------------------
        1. Steam ejectors: Best for low capital, high operating cost
        2. Liquid ring pumps: Good for moderate vacuum, low operating cost
        3. Mechanical pumps: Best for deep vacuum, high capital
        
        Parameters:
        -----------
        W_air_lb_hr : float
            Air leakage rate (lb/hr)
        P_torr : float
            Suction pressure (torr)
        Q_ft3_min : float
            Volumetric flow rate (ft³/min)
        
        Returns:
        --------
        VacuumEquipment
        """
        candidates = []
        
        for key, equip in VACUUM_EQUIPMENT.items():
            # Check pressure range
            if not (equip.min_pressure_torr <= P_torr <= equip.max_pressure_torr):
                continue
            
            # Calculate size factor
            if equip.size_type == 'ejector':
                S = W_air_lb_hr / P_torr
            else:
                S = Q_ft3_min
            
            # For small flows below equipment minimum, use minimum size
            # This represents buying the smallest available unit
            if S < equip.min_size:
                S_for_cost = equip.min_size
            elif S > equip.max_size:
                # Skip if too large - need multiple units
                continue
            else:
                S_for_cost = S
            
            # Calculate capital cost
            capital = equip.cost_coef * (S_for_cost ** equip.cost_exp) * (self.cepci / self.cepci_base)
            
            # Calculate operating cost
            if equip.steam_ratio > 0:
                # Steam ejector
                operating = (self.steam_cost_per_1000lb * W_air_lb_hr * 
                            equip.steam_ratio * self.operating_hours / 1000)
            else:
                # Electric pump - use actual flow for power calculation
                actual_power = max(Q_ft3_min, equip.min_size * 0.5) * equip.power_factor
                operating = actual_power * self.electricity_cost_kWh * self.operating_hours
            
            total_annual = capital / self.payback_period + operating
            
            candidates.append((total_annual, capital, operating, S_for_cost, equip))
        
        if not candidates:
            # Fallback: use single-stage ejector (always works for moderate vacuum)
            logger.warning("  No ideal equipment match - using one-stage ejector as fallback")
            return VACUUM_EQUIPMENT['one_stage_ejector']
        
        # Sort by total annual cost and select best
        candidates.sort(key=lambda x: x[0])
        best = candidates[0]
        
        logger.debug(f"  Equipment comparison:")
        for tac, cap, op, s, eq in candidates[:3]:
            logger.debug(f"    {eq.name}: TAC=${tac:,.0f} (Cap=${cap:,.0f}, Op=${op:,.0f})")
        
        return best[4]  # Return the equipment object
    
    def _no_vacuum_system(self):
        """Return empty vacuum system result for atmospheric operation."""
        return {
            'system_type': 'None (atmospheric)',
            'system_volume_m3': 0,
            'system_volume_ft3': 0,
            'pressure_torr': 760,
            'air_leakage_lb_hr': 0,
            'air_leakage_kg_hr': 0,
            'volumetric_flow_ft3_min': 0,
            'size_factor': 0,
            'steam_consumption_lb_hr': 0,
            'steam_consumption_kg_hr': 0,
            'power_kW': 0,
            'capital_cost': 0,
            'annual_operating_cost': 0,
        }
    
    # ════════════════════════════════════════════════════════════════════════
    # ORIGINAL METHODS (unchanged)
    # ════════════════════════════════════════════════════════════════════════
    
    def _select_cooling_utility(self, T_cond: float) -> CoolingUtility:
        """Auto-select cooling utility based on condenser temperature."""
        for key in ['CW', 'ChW', 'Refrig', 'Brine']:
            utility = COOLING_UTILITIES[key]
            if T_cond >= utility.T_out + self.DELTA_T_MIN:
                logger.debug(f"Selected {utility.name} for T_cond={T_cond:.1f}°C")
                return utility
        
        logger.warning(f"T_cond={T_cond:.1f}°C is very low! Using Brine.")
        return COOLING_UTILITIES['Brine']
    
    def _select_heating_utility(self, T_reb: float) -> HeatingUtility:
        """Auto-select heating utility based on reboiler temperature."""
        for key in ['LPS', 'MPS', 'HPS', 'HotOil', 'Fired']:
            utility = HEATING_UTILITIES[key]
            if utility.T_steam >= T_reb + self.DELTA_T_MIN:
                logger.debug(f"Selected {utility.name} for T_reb={T_reb:.1f}°C")
                return utility
        
        logger.warning(f"T_reb={T_reb:.1f}°C is very high! Using Fired Heater.")
        return HEATING_UTILITIES['Fired']
    
    def _condenser_cost(self, Q_cond: float, T_cond: float) -> dict:
        """Calculate condenser cost with auto utility selection."""
        utility = self._select_cooling_utility(T_cond)
        
        dT1 = T_cond - utility.T_in
        dT2 = T_cond - utility.T_out
        
        if dT1 <= 0 or dT2 <= 0:
            logger.warning(f"Condenser pinch violation! T_cond={T_cond}")
            LMTD = self.DELTA_T_MIN
        elif abs(dT1 - dT2) < 0.1:
            LMTD = dT1
        else:
            LMTD = (dT1 - dT2) / math.log(dT1 / dT2)
        
        LMTD = max(LMTD, 5.0)
        U = utility.U_typical
        A = (Q_cond * 1000) / (U * LMTD)
        A = max(A, 10)
        A = min(A, 1000)
        
        K1, K2, K3 = 4.8306, -0.8509, 0.3187
        log_A = math.log10(A)
        log_Cp = K1 + K2 * log_A + K3 * log_A**2
        base_cost = 10**log_Cp
        
        Fp, Fm, Ft = 1.0, 1.75, 1.0
        cost = base_cost * Fp * Fm * Ft * (self.cepci / self.cepci_base)
        
        return {'cost': cost, 'utility': utility, 'LMTD': LMTD, 'U': U, 'area': A}
    
    def _reboiler_cost(self, Q_reb: float, T_reb: float) -> dict:
        """Calculate reboiler cost with auto utility selection."""
        utility = self._select_heating_utility(T_reb)
        
        LMTD = utility.T_steam - T_reb
        LMTD = max(LMTD, self.DELTA_T_MIN)
        U = utility.U_typical
        A = (Q_reb * 1000) / (U * LMTD)
        A = max(A, 10)
        A = min(A, 500)
        
        K1, K2, K3 = 4.4646, -0.5277, 0.3955
        log_A = math.log10(A)
        log_Cp = K1 + K2 * log_A + K3 * log_A**2
        base_cost = 10**log_Cp
        
        Fp, Fm, Ft = 1.0, 1.75, 1.35
        cost = base_cost * Fp * Fm * Ft * (self.cepci / self.cepci_base)
        
        return {'cost': cost, 'utility': utility, 'LMTD': LMTD, 'U': U, 'area': A}
    
    def _estimate_condenser_temp(self, pressure: float) -> float:
        """Estimate condenser temperature from pressure."""
        A, B = 105.0, 25.0
        T = A + B * math.log(pressure)
        return max(T, 30.0)
    
    def _estimate_reboiler_temp(self, pressure: float) -> float:
        """Estimate reboiler temperature from pressure."""
        A, B = 115.0, 25.0
        T = A + B * math.log(pressure)
        return max(T, 40.0)
    
    def _column_cost_guthrie(self, diameter, n_trays, pressure):
        """Guthrie correlation for column + trays."""
        if n_trays <= 0:
            return 0
        
        D_ft = diameter * 3.281
        H_ft = n_trays * self.tray_spacing_ft
        
        D_ft = max(D_ft, 1.0)
        H_ft = max(H_ft, 4.0)
        
        Fs = 1.0 if self.tray_spacing_ft >= 2.0 else 1.4
        Ft = 0.0
        Fm = 1.7 if self.material == 'SS' else 0.0
        Fc = Fs + Ft + Fm
        
        if pressure < 0.1:
            Fp = 1.40
        elif pressure < 0.3:
            Fp = 1.25
        elif pressure < 0.5:
            Fp = 1.15
        elif pressure < 1.0:
            Fp = 1.05
        else:
            Fp = 1.00
        
        cost = (self.ms_index / self.ms_base) * 101.9 * \
               (D_ft ** 1.066) * (H_ft ** 0.82) * Fc * Fp
        
        return cost
    
    def _failed_result(self):
        """Return failed result dict."""
        return {'TAC': 1e12, 'TPC': 0, 'TOC': 0}


# ════════════════════════════════════════════════════════════════════════════
# TESTING
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("TAC CALCULATOR v3.1 - SEIDER VACUUM MODEL TEST")
    print("=" * 70)
    
    # ════════════════════════════════════════════════════════════════════════
    # TEST 1: Verify Seider Example (Example 22.18)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("TEST 1: Verify Seider Example 22.18")
    print("-" * 60)
    print("Given: V = 50,000 ft³, P = 25 kPa = 188 torr")
    print("Expected: W = 227 lb/hr")
    
    # Calculate manually
    V_ft3 = 50000
    P_torr = 188
    ln_P = math.log(P_torr)
    bracket = 0.0298 + 0.03088 * ln_P - 0.0005733 * ln_P**2
    W_calc = 5 + bracket * (V_ft3 ** 0.66)
    
    print(f"\nCalculation:")
    print(f"  ln(188) = {ln_P:.4f}")
    print(f"  bracket = 0.0298 + 0.03088×{ln_P:.3f} - 0.0005733×{ln_P:.3f}² = {bracket:.5f}")
    print(f"  V^0.66 = {V_ft3}^0.66 = {V_ft3**0.66:.1f}")
    print(f"  W = 5 + {bracket:.5f} × {V_ft3**0.66:.1f} = {W_calc:.1f} lb/hr")
    print(f"\n  ✓ Expected: 227 lb/hr, Calculated: {W_calc:.1f} lb/hr")
    
    # ════════════════════════════════════════════════════════════════════════
    # TEST 2: EB/SM Column at 0.2 bar
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("TEST 2: EB/SM Separation at 0.2 bar (150 torr)")
    print("-" * 60)
    
    calc = TACCalculator(
        material='SS',
        cepci=800,
        cepci_base=500
    )
    
    result = calc.calculate(
        nt=45,
        diameter=0.9,
        Q_reb=250.0,
        Q_cond=370.0,
        pressure=0.2,
        T_cond=65.0,
        T_reb=85.0
    )
    
    print(f"\nColumn: NT=45, D=0.9m, P=0.2 bar")
    print(f"\nVacuum System:")
    vac = result['vacuum_system']
    print(f"  Type: {vac['system_type']}")
    print(f"  System volume: {vac['system_volume_m3']:.1f} m³ ({vac['system_volume_ft3']:.0f} ft³)")
    print(f"  Air leakage: {vac['air_leakage_lb_hr']:.1f} lb/hr ({vac['air_leakage_kg_hr']:.1f} kg/hr)")
    print(f"  Volumetric flow: {vac['volumetric_flow_ft3_min']:.1f} ft³/min")
    print(f"  Size factor: {vac['size_factor']:.2f}")
    if vac['steam_consumption_lb_hr'] > 0:
        print(f"  Steam: {vac['steam_consumption_lb_hr']:.0f} lb/hr")
    else:
        print(f"  Power: {vac['power_kW']:.1f} kW")
    print(f"  Capital: ${vac['capital_cost']:,.0f}")
    print(f"  Operating: ${vac['annual_operating_cost']:,.0f}/yr")
    
    print(f"\nCost Summary:")
    print(f"  Column:        ${result['column_cost']:>12,.0f}")
    print(f"  Condenser:     ${result['condenser_cost']:>12,.0f}")
    print(f"  Reboiler:      ${result['reboiler_cost']:>12,.0f}")
    print(f"  Vacuum system: ${result['vacuum_system_cost']:>12,.0f}")
    print(f"  ────────────────────────────────")
    print(f"  TPC:           ${result['TPC']:>12,.0f}")
    print(f"  TOC:           ${result['TOC']:>12,.0f}/yr")
    print(f"  TAC:           ${result['TAC']:>12,.0f}/yr")
    
    # ════════════════════════════════════════════════════════════════════════
    # TEST 3: Deep Vacuum (0.05 bar = 37.5 torr)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("TEST 3: Deep Vacuum at 0.05 bar (37.5 torr)")
    print("-" * 60)
    
    result2 = calc.calculate(
        nt=35,
        diameter=0.8,
        Q_reb=180.0,
        Q_cond=250.0,
        pressure=0.05,
        T_cond=45.0,
        T_reb=75.0
    )
    
    vac2 = result2['vacuum_system']
    print(f"\nVacuum System:")
    print(f"  Type: {vac2['system_type']}")
    print(f"  Air leakage: {vac2['air_leakage_lb_hr']:.1f} lb/hr")
    print(f"  Capital: ${vac2['capital_cost']:,.0f}")
    print(f"  Operating: ${vac2['annual_operating_cost']:,.0f}/yr")
    print(f"\n  TAC: ${result2['TAC']:,.0f}/yr")
    
    # ════════════════════════════════════════════════════════════════════════
    # TEST 4: Moderate Vacuum (0.5 bar = 375 torr)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("TEST 4: Moderate Vacuum at 0.5 bar (375 torr)")
    print("-" * 60)
    
    result3 = calc.calculate(
        nt=30,
        diameter=0.7,
        Q_reb=150.0,
        Q_cond=200.0,
        pressure=0.5,
        T_cond=95.0,
        T_reb=120.0
    )
    
    vac3 = result3['vacuum_system']
    print(f"\nVacuum System:")
    print(f"  Type: {vac3['system_type']}")
    print(f"  Air leakage: {vac3['air_leakage_lb_hr']:.1f} lb/hr")
    print(f"  Capital: ${vac3['capital_cost']:,.0f}")
    print(f"  Operating: ${vac3['annual_operating_cost']:,.0f}/yr")
    print(f"\n  TAC: ${result3['TAC']:,.0f}/yr")
    
    # ════════════════════════════════════════════════════════════════════════
    # TEST 5: Atmospheric (no vacuum system needed)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("TEST 5: Atmospheric Operation (1.01 bar)")
    print("-" * 60)
    
    result4 = calc.calculate(
        nt=25,
        diameter=0.6,
        Q_reb=100.0,
        Q_cond=150.0,
        pressure=1.01,
        T_cond=110.0,
        T_reb=135.0
    )
    
    print(f"\nVacuum System: {result4['vacuum_system']['system_type']}")
    print(f"  Capital: ${result4['vacuum_system_cost']:,.0f}")
    print(f"\n  TAC: ${result4['TAC']:,.0f}/yr")
    
    # ════════════════════════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON: Effect of Operating Pressure")
    print("=" * 70)
    
    print(f"\n{'Pressure':<12} {'Vacuum System':<25} {'Capital':>12} {'Operating':>12} {'TAC':>12}")
    print("-" * 73)
    print(f"{'0.05 bar':<12} {vac2['system_type']:<25} ${vac2['capital_cost']:>10,.0f} ${vac2['annual_operating_cost']:>10,.0f} ${result2['TAC']:>10,.0f}")
    print(f"{'0.2 bar':<12} {vac['system_type']:<25} ${vac['capital_cost']:>10,.0f} ${vac['annual_operating_cost']:>10,.0f} ${result['TAC']:>10,.0f}")
    print(f"{'0.5 bar':<12} {vac3['system_type']:<25} ${vac3['capital_cost']:>10,.0f} ${vac3['annual_operating_cost']:>10,.0f} ${result3['TAC']:>10,.0f}")
    print(f"{'1.01 bar':<12} {'None':<25} ${0:>10,.0f} ${0:>10,.0f} ${result4['TAC']:>10,.0f}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)