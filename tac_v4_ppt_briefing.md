# TAC Calculator v4.0 — PPT Update Briefing

## Purpose
This document contains all details needed to update the thesis PowerPoint presentation
with the TAC v4.0 costing model changes. Use this as reference when working on home PC.

---

## 1. What Changed (Old → New)

### OLD Model (v3.1) — PROBLEMATIC
| Equipment | Correlation Source | Cost Index | Base Value | Year | Ratio (800/base) |
|---|---|---|---|---|---|
| Column shell | **Guthrie (1969)** | M&S (proxy via CEPCI) | 119 | 1969 | 6.72x |
| Condenser | Turton Table A.1 | CEPCI | 500 | 2006(wrong) | 1.60x |
| Reboiler | Turton Table A.1 | CEPCI | 500 | 2006(wrong) | 1.60x |
| Vacuum system | Seider Table 22.32 | CEPCI | 500 | 2006 | 1.60x |

**Problems with v3.1:**
1. Guthrie (1969) was calibrated with **M&S index** (Marshall & Swift), NOT CEPCI
2. M&S was **discontinued in 2012** — no current values exist
3. Using CEPCI=119 as proxy for M&S=280 introduces **~18% error**
4. Turton HX correlations used CEPCI base=500, but Turton's actual native base is **397** (year 2001)
5. Mixing two incompatible index systems in the same TAC calculation

### NEW Model (v4.0) — UNIFIED TURTON/CEPCI
| Equipment | Correlation Source | Cost Index | Base Value | Year | Ratio (800/base) |
|---|---|---|---|---|---|
| Column vessel | **Turton Table A.1** | CEPCI | **397** | 2001 | **2.015x** |
| Sieve trays | **Turton Table A.1** | CEPCI | **397** | 2001 | **2.015x** |
| Condenser | Turton Table A.1 | CEPCI | **397** | 2001 | **2.015x** |
| Reboiler | Turton Table A.1 | CEPCI | **397** | 2001 | **2.015x** |
| Vacuum system | Seider Table 22.32 | CEPCI | 500 | 2006 | 1.60x |

**Why Seider stays at 500**: Seider et al. (2017) has its own native CEPCI base year (2006, CEPCI=500).
It's a different source, so it keeps its own base. This is correct practice.

---

## 2. Equations & Coefficients (for PPT slides)

### General Turton Form (Table A.1)
```
log₁₀(Cp°) = K₁ + K₂·log₁₀(A) + K₃·[log₁₀(A)]²
```
Where Cp° = purchased equipment cost at base CEPCI (397, year 2001)

### Cost Escalation
```
Cost_current = Cp° × Fm × Fp × (CEPCI_current / CEPCI_base)
             = Cp° × Fm × Fp × (800 / 397)
```

### Column Vessel (Vertical Process Vessel)
- **Source**: Turton Table A.1
- **Size parameter**: Volume V (m³) = π/4 × D² × H
- **Valid range**: 0.3 ≤ V ≤ 520 m³
- **Coefficients**: K₁ = 3.4974, K₂ = 0.4485, K₃ = 0.1074
- **Material factor** (Table A.3): Fm = 1.7 (SS-304), 1.0 (CS)
- **Pressure factor** (Table A.2):
  - P < 0.5 bar abs → Fp = 1.25 (vacuum penalty)
  - 0.5 ≤ P ≤ 2.0 bar → Fp = 1.0 (atmospheric)
  - P > 2.0 bar → log₁₀(Fp) = 0.03881 − 0.11272·log₁₀(P_barg) + 0.08183·[log₁₀(P_barg)]²

### Sieve Trays
- **Source**: Turton Table A.1
- **Size parameter**: Tray area A (m²) = π/4 × D²
- **Valid range**: 0.07 ≤ A ≤ 12.3 m²
- **Coefficients**: K₁ = 2.9949, K₂ = 0.4465, K₃ = 0.3961
- **Material factor** (Table A.6): Fm_tray = 1.189 (SS-304), 1.0 (CS)
  - NOTE: Tray Fm ≠ Vessel Fm (different table!)
- **Quantity factor Fq**:
  - NT ≥ 20: Fq = 1.0
  - NT < 20: log₁₀(Fq) = 0.4771 + 0.08516·log₁₀(NT) − 0.3473·[log₁₀(NT)]²
- **Total tray cost**: Cp°_per_tray × NT × Fm_tray × Fq × (CEPCI/397)

### Condenser (Floating Head HX)
- **Source**: Turton Table A.1
- **Size parameter**: Area A (m²)
- **Valid range**: 10 ≤ A ≤ 1000 m²
- **Coefficients**: K₁ = 4.8306, K₂ = −0.8509, K₃ = 0.3187
- **Factors**: Fp = 1.0, Fm = 1.75 (SS), Ft = 1.0

### Reboiler (Kettle Reboiler)
- **Source**: Turton Table A.1
- **Size parameter**: Area A (m²)
- **Valid range**: 10 ≤ A ≤ 500 m²
- **Coefficients**: K₁ = 4.4646, K₂ = −0.5277, K₃ = 0.3955
- **Factors**: Fp = 1.0, Fm = 1.75 (SS), Ft = 1.35

### Vacuum System (unchanged from v3.1)
- **Source**: Seider Table 22.32, Eq. 22.73 (air leakage)
- **CEPCI base**: 500 (year 2006) — kept separate
- Equipment auto-selected by pressure & flow rate (steam ejectors or liquid-ring pump)

---

## 3. Quantity Factor Table (for PPT)

| NT (trays) | Fq |
|---|---|
| 1 | 3.00 |
| 5 | 1.74 |
| 10 | 1.35 |
| 15 | 1.13 |
| 20+ | 1.00 |

---

## 4. Column Cost = Vessel + Trays (Total Purchased Cost)

```
Column_cost = Vessel_cost + Tray_cost

Where:
  Vessel_cost = Cp°_vessel × Fm × Fp × (800/397)
  Tray_cost   = Cp°_tray × NT × Fm_tray × Fq × (800/397)
```

**We use Purchased Cost (Cp), NOT Bare Module Cost (CBM):**
- Purchased cost = cost of equipment FOB (free on board)
- Bare module = includes installation, piping, electrical, etc.
- We chose Cp because our TAC already has a payback period divisor that implicitly accounts for installation

---

## 5. References (for PPT bibliography)

1. **Turton, R., Shaeiwitz, J.A., Bhattacharyya, D., Whiting, W.B.** (2018).
   *Analysis, Synthesis, and Design of Chemical Processes*, 5th Edition.
   Prentice Hall. Appendix A (CAPCOST).

2. **Seider, W.D., Seader, J.D., Lewin, D.R., Widagdo, S.** (2017).
   *Product and Process Design Principles: Synthesis, Analysis, and Evaluation*,
   4th Edition. John Wiley & Sons. Chapter 22.

3. **Chemical Engineering Magazine** — CEPCI (Chemical Engineering Plant Cost Index).
   Annual updates. Current value used: **CEPCI = 800** (year 2024).

---

## 6. Impact on Results

- **HX costs increased ~26%**: CEPCI ratio changed from 800/500=1.60 to 800/397=2.015
- **Column shell cost changed**: Entirely new formula (Guthrie volume-based → Turton volume-based)
- **Tray cost now explicit**: Previously bundled inside Guthrie's shell cost
- **Vacuum costs unchanged**: Seider base stays at 500
- **ALL previous optimization results invalidated** — must re-run for thesis-quality numbers
- Optimal configurations (NT, NF, P) likely similar but absolute TAC values differ

---

## 7. Why Not M&S?

Marshall & Swift Equipment Cost Index was **discontinued around 2012**.
No official M&S values exist for years after 2012, so it cannot be used
for current-year cost escalation. CEPCI is the only widely accepted
equipment cost index still actively published.

---

## 8. Why Not Bare Module Cost?

Bare Module Cost (CBM = Cp° × FBM where FBM = B1 + B2·Fp·Fm) includes
installation multipliers. Our TAC formula already divides total plant cost
by a payback period (3 years), which serves a similar purpose. Using
purchased cost avoids double-counting installation factors.

---

## 9. Files Changed in Codebase

| File | Change |
|---|---|
| `tac_calculator.py` | Replaced Guthrie with Turton vessel+tray, fixed CEPCI bases |
| `config.py` | `cepci_base: 500 → 397` |
| `main_sequential_optimizer.py` | `cepci_base=500 → 397` |
| `ga_optimizer.py` | `cepci_base=500 → 397` |
| `pso_optimizer.py` | `cepci_base=500 → 397` |
