# Concurrent Optimization Report: Case1 COL2 & COL3
## Distillation Column Optimization - Multi-Algorithm Comparison

**Generated:** February 1, 2026
**Case:** Case1 - EB/SM Separation (COL2 & COL3)
**Constraint:** T_reboiler <= 120°C

---

## Executive Summary

Three optimization algorithms (ISO, GA, PSO) were run on both COL2 and COL3 to optimize distillation column designs for minimum Total Annual Cost (TAC). The results show different optimal configurations for each column due to their different separation requirements.

### Key Findings

**COL2 (Lighter Separation):**
- **PSO achieved the lowest TAC** at $118,239/year with only 15 trays
- GA and ISO converged to similar solutions around $191,500-191,850/year with 30-31 trays
- PSO found a more aggressive design operating at the temperature constraint boundary (T_reb = 120°C)

**COL3 (Heavier Separation):**
- **GA achieved the lowest TAC** at $753,572/year
- All algorithms converged to similar solutions (~$753,500-768,500/year)
- Requires significantly more trays (82-91) due to more difficult separation

---

## COL2 Results Comparison

| Metric | ISO | GA | PSO | Best |
|--------|-----|----|----|------|
| **TAC ($/year)** | 191,850 | 191,565 | **118,239** | PSO |
| **Number of Trays (NT)** | 31 | 30 | **15** | PSO |
| **Feed Stage (NF)** | 12 | 12 | 12 | All |
| **Pressure (bar)** | 0.325 | 0.301 | **0.461** | PSO |
| **T_reboiler (°C)** | - | 107.3 | **120.0** | GA (margin) |
| **Total Evaluations** | 100 | 1,000 | 340 | ISO |
| **Computation Time** | 108 min | 131 min | 994 min | ISO |
| **Feasibility Rate** | 66% | **92%** | 55% | GA |

### COL2 Cost Breakdown (PSO Best Solution)

| Cost Component | Value |
|----------------|-------|
| Total Plant Cost (TPC) | $215,672 |
| Capital Cost (annualized) | $71,891 |
| Total Operating Cost (TOC) | $46,348 |
| **Total Annual Cost (TAC)** | **$118,239** |

---

## COL3 Results Comparison

| Metric | ISO | GA | PSO | Best |
|--------|-----|----|----|------|
| **TAC ($/year)** | 768,487 | **753,572** | 753,788 | GA |
| **Number of Trays (NT)** | 91 | **84** | 82 | PSO |
| **Feed Stage (NF)** | 42 | 37 | 36 | PSO |
| **Pressure (bar)** | 0.156 | 0.151 | **0.142** | PSO |
| **T_reboiler (°C)** | - | 88.1 | **86.4** | PSO |
| **Total Evaluations** | 225 | 650 | 525 | ISO |
| **Computation Time** | 109 min | 121 min | 258 min | ISO |
| **Feasibility Rate** | 78% | **90%** | 86% | GA |

### COL3 Cost Breakdown (GA Best Solution)

| Cost Component | Value |
|----------------|-------|
| Total Plant Cost (TPC) | $1,313,004 |
| Capital Cost (annualized) | $437,668 |
| Total Operating Cost (TOC) | $315,905 |
| **Total Annual Cost (TAC)** | **$753,572** |

---

## Detailed Algorithm Analysis

### COL2 Analysis

#### ISO Convergence (COL2)
| Iteration | Pressure | NT | Feed | TAC |
|-----------|----------|----|----- |-----|
| 1 | 0.213 | 31 | 12 | $198,213 |
| 2 | 0.325 | 31 | 12 | $191,850 |
| 3 | 0.325 | 31 | 12 | $191,850 |

**Note:** ISO found a local optimum with more trays. PSO discovered a different, more economical design region with fewer trays and higher pressure.

#### GA Convergence (COL2)
```
Gen 1:  $231,956  (initial population)
Gen 10: $202,457  (-12.7%)
Gen 20: $192,198  (-5.1%)
Gen 31: $191,565  (final optimum)
```

#### PSO Convergence (COL2)
```
Iter 1:  $124,729  (initial swarm)
Iter 2:  $118,239  (final optimum - best overall!)
```
PSO converged very quickly to an aggressive design at the constraint boundary.

---

### COL3 Analysis

#### ISO Convergence (COL3)
| Iteration | Pressure | NT | Feed | TAC |
|-----------|----------|----|----- |-----|
| 1 | 0.194 | 95 | 42 | $771,540 |
| 2 | 0.156 | 91 | 42 | $768,487 |
| 3 | 0.156 | 91 | 42 | $768,487 |

#### GA Convergence (COL3)
```
Gen 1:  $770,756  (initial population)
Gen 5:  $761,124  (-1.2%)
Gen 9:  $760,176  (-0.1%)
Gen 10: $753,572  (final optimum)
```

#### PSO Convergence (COL3)
```
Iter 1:  $772,665  (initial swarm)
Iter 2:  $759,727  (-1.7%)
Iter 4:  $756,208  (-0.5%)
Iter 6:  $753,788  (final optimum)
```

---

## Optimal Design Recommendations

### COL2 Recommended Design

| Parameter | Conservative | Aggressive |
|-----------|--------------|------------|
| **Number of Trays** | 30 | 15 |
| **Feed Stage** | 12 | 12 |
| **Pressure** | 0.30 bar | 0.46 bar |
| **Expected TAC** | ~$191,500/year | ~$118,200/year |
| **T_reboiler** | ~107°C | ~120°C |
| **Risk Level** | Low | Higher (at constraint) |

**Recommendation:** The conservative design (GA solution) provides margin on the temperature constraint. The aggressive design (PSO) saves ~$73,000/year but operates at the constraint boundary.

### COL3 Recommended Design

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Number of Trays** | 82-84 | Consensus from GA & PSO |
| **Feed Stage** | 36-37 | GA/PSO agree |
| **Pressure** | 0.14-0.15 bar | Low pressure for separation |
| **Expected TAC** | ~$753,500/year | |
| **T_reboiler** | ~87°C | Well below 120°C limit |

---

## Algorithm Performance Summary

### Speed Comparison
| Column | ISO | GA | PSO |
|--------|-----|----|----|
| COL2 | 108 min | 131 min | 994 min |
| COL3 | 109 min | 121 min | 258 min |

### Solution Quality
| Column | ISO TAC | GA TAC | PSO TAC | Best |
|--------|---------|--------|---------|------|
| COL2 | $191,850 | $191,565 | $118,239 | PSO |
| COL3 | $768,487 | $753,572 | $753,788 | GA |

### Feasibility Rate
| Column | ISO | GA | PSO |
|--------|-----|----|----|
| COL2 | 66% | 92% | 55% |
| COL3 | 78% | 90% | 86% |

---

## Conclusions

1. **PSO found a significantly better COL2 design** with 38% lower TAC, though at the temperature constraint boundary
2. **GA provides consistent, reliable solutions** with the highest feasibility rates
3. **ISO is most efficient** for quick estimates (fewest evaluations)
4. **COL3 is more challenging** - requires 5-6x more trays than COL2 and has 4x higher TAC
5. **Algorithm selection matters** - different algorithms found different local optima for COL2

### Recommended Algorithm Selection

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Quick estimate / sensitivity study | ISO |
| Production optimization (conservative) | GA |
| Maximum cost reduction (accept risk) | PSO |
| Exploring new design space | GA or PSO |

---

## Files Generated

### COL2 Results (Latest Run)
- `Case1_COL2_20260127_205419/iso_result_20260127_224308.json`
- `Case1_COL2_GA_20260128_032654/ga_result_20260128_032711.json`
- `Case1_COL2_PSO_20260201_050541/pso_result_20260201_050559.json`

### COL3 Results (Latest Run)
- `Case1_COL3_20260201_044435/iso_result_20260201_063350.json`
- `Case1_COL3_GA_20260131_034640/ga_result_20260131_034654.json`
- `Case1_COL3_PSO_20260130_204815/pso_result_20260130_204843.json`

---

*Report generated by Column Optimization Dashboard*
*Last updated: February 1, 2026*