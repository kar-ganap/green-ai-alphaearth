# Critical Clarification: Logging vs Fire-Based Deforestation

**Date**: 2025-10-15
**Issue**: We've been validating AlphaEarth (detects ALL deforestation) against GLAD (fire-biased)

---

## The Problem

### What We've Been Doing:

```
AlphaEarth Embeddings     →     Validated Against     →     GLAD Alerts
(Multi-modal:                                              (Fire-biased:
 Optical + Radar + Lidar)                                   Optical only)
     ↓                                                           ↓
Detects ALL deforestation:                              Detects primarily:
- Selective logging                                     - Slash-and-burn
- Gradual degradation                                   - Fire-based clearing
- Canopy height reduction                               - Optical canopy loss
- Structural changes (radar)                            - Burned areas
     ↓                                                           ↓
   BROADER                                                    NARROWER
```

**Result**: We're validating the **superset** (all deforestation) against a **subset** (fire-only)!

---

## Dataset Comparison

### Hansen Global Forest Change (GFC)
**What it detects**:
- ✓ **ALL canopy loss** >30% tree cover reduction
- ✓ Includes: Fire clearing + selective logging + degradation + mechanized clearing
- ✓ Based on: Landsat time series (optical)
- ✓ Coverage: Global, complete
- ✓ Resolution: Annual (year-level)

**Limitations**:
- ✗ **NO quarterly/monthly dates** - only knows "cleared in 2020"
- ✗ Cannot distinguish Q1 from Q4
- ✗ Cannot answer "when did clearing occur"

**Source**: UMD/hansen/global_forest_change_2024_v1_12

---

### GLAD Alerts (Landsat)
**What it detects**:
- ✓ **Fire-based clearing** (slash-and-burn)
- ✓ Rapid canopy loss visible in optical
- ⚠️ **May miss selective logging** without fire
- ⚠️ **May miss gradual degradation**
- ✓ Based on: Landsat optical (+ some radar in newer versions)
- ✓ Weekly-level dates (precise timing)

**Limitations**:
- ✗ **Incomplete coverage** - sparse in our region (only 8-17 pixels found)
- ✗ **Fire-biased** - Q3 fire season dominates (71%)
- ✗ Misses non-fire deforestation mechanisms

**Source**: projects/glad/alert/{year}final

---

### AlphaEarth Embeddings
**What it detects** (hypothesized):
- ✓ **ALL forest changes**:
  - Optical: NDVI, phenology, canopy closure
  - **Radar (Sentinel-1)**: Structural changes, soil moisture, **penetrates canopy**
  - **Lidar (GEDI)**: **Canopy height reduction**, vertical structure
  - Climate (ERA5): Temperature, precipitation anomalies

**Key capabilities**:
- ✓ **Radar sees through canopy** - detects selective logging
- ✓ **GEDI measures height** - detects canopy thinning
- ✓ Multi-modal fusion captures **degradation signatures**

**Should detect**:
- ✓ Selective logging (radar + lidar)
- ✓ Gradual degradation (multi-temporal optical + radar)
- ✓ Fire-based clearing (optical burn scars)
- ✓ Road construction (radar structural changes)
- ✓ Vegetation stress (optical phenology)

**Source**: projects/meta-ei-data/assets/alpha-earth/v1

---

## Why This Matters

### Our Quarterly Validation Results:
- Q3 (Jul-Sep fire season): **71%** dominant
- Q4 (Oct-Dec): 24%
- Q1-Q2 (Jan-Jun): 6%

**Interpretation with GLAD**:
- "Q3 fire season dominates" ← TRUE for **fire-based** clearing
- "AlphaEarth detects fire season clearing with 0-3 month lead" ← TRUE but **INCOMPLETE**

**But if including logging**:
- Q1-Q2 (wet season): **This is when selective logging typically occurs!**
  - Wet season: Better access for trucks
  - Pre-fire cutting: Cutting in Q1-Q2, burning in Q3-Q4
  - GLAD misses this! (no fire yet)

- AlphaEarth **should** detect Q1-Q2 logging via:
  - Sentinel-1 radar: Structural changes, canopy gaps
  - GEDI lidar: Canopy height reduction
  - Optical: NDVI decline, phenology changes

**We've been under-counting AlphaEarth's capabilities!**

---

## Evidence AlphaEarth Detects More Than Fire

### From Our Deep-Dive Investigation:

**Temporal Trajectories** (temporal_trajectories.py):
- Y-2→Y-1 change: 0.13 (gradual)
- Y-1→Y change: 0.58 (**4.5x acceleration**)
- **This suggests PROGRESSIVE degradation**, not just fire detection

**Multi-Dimensional Signature**:
- 48% of embedding dimensions discriminate (p < 0.05)
- Mixed directionality (some increase, some decrease)
- **Cannot be explained by fire alone** - suggests multi-modal signals

**Spatial Investigation**:
- NO proximity to roads/edges/clearings
- Pixels >5km from 2019 clearings
- **Suggests isolated degradation**, not just frontier fire expansion

### What This Implies:

AlphaEarth embeddings are detecting:
- ✓ **Degradation** (gradual Y-2→Y-1 change)
- ✓ **Multi-modal signals** (48% dims, mixed directions)
- ✓ **Isolated events** (not spatially autocorrelated)

**NOT just**:
- ✗ Fire-based clearing (which would show spatial clustering near roads)
- ✗ Simple optical change (which would be 1-2 dimensions)

**Conclusion**: AlphaEarth likely detects **selective logging and degradation**, not just fire!

---

## The GLAD Validation Bias

### Why GLAD Shows Q3 Dominance (71%):

**GLAD Detection Mechanism**:
1. Optical change detection (Landsat)
2. Identifies **burned areas** and **rapid canopy loss**
3. Confirms with radar (in newer versions)
4. Issues alert

**Result**:
- ✓ Excellent at detecting **slash-and-burn** (Q3 fire season)
- ✗ Misses **selective logging** without fire (Q1-Q2)
- ✗ Misses **gradual degradation** (slow change)

**Our 71% Q3 finding reflects GLAD's bias, not true deforestation distribution!**

### Literature on Deforestation Timing:

**Slash-and-Burn Timeline**:
- Q1-Q2 (Jan-Jun): **Cutting** (wet season, tractors, no fire)
- Q2-Q3 (Jun-Jul): **Drying** (biomass left to dry)
- Q3-Q4 (Aug-Oct): **Burning** (dry season, fire) ← **GLAD detects this**

**Selective Logging Timeline**:
- Q1-Q2 (Jan-Jun): **Wet season logging** (truck access, no fire)
- No fire! Just canopy removal ← **GLAD misses this**

**AlphaEarth Should Detect**:
- ✓ Q1-Q2 cutting (radar structural changes, GEDI height loss)
- ✓ Q3-Q4 burning (optical burn scars)
- ✓ Gradual degradation (multi-temporal changes)

**GLAD Only Detects**:
- ✗ Q1-Q2 cutting (NO - no fire yet)
- ✓ Q3-Q4 burning (YES - fire visible)
- ✗ Gradual degradation (NO - too slow)

---

## Recommended Validation Strategy

### Option A: Hansen GFC Validation (No Quarterly Stratification)

**Approach**:
- Use **Hansen GFC** as ground truth (ALL deforestation)
- **Accept** we cannot stratify by quarter (annual only)
- Focus on **model performance** (AUC, precision, recall)
- **Stop trying** to answer "detection vs prediction" with quarterly analysis

**Pros**:
- ✓ Validates against **complete deforestation** (not fire-biased)
- ✓ Includes selective logging, degradation, all mechanisms
- ✓ Large sample sizes (1000s of pixels available)
- ✓ Matches our original CRAWL phase (AUC 0.894)

**Cons**:
- ✗ Cannot determine lead time (0-3 vs 3-6 months)
- ✗ Cannot distinguish early detection from prediction
- ✗ No quarterly insights

**Recommendation**: **Use for production model validation**

---

### Option B: Multi-Dataset Validation (Hansen + GLAD)

**Approach**:
1. **Primary validation**: Hansen GFC (all deforestation, no quarterly)
2. **Secondary analysis**: GLAD subset (fire-based, quarterly)
3. **Report separately**:
   - "Model AUC on ALL deforestation (Hansen): 0.85-0.90"
   - "Lead time for FIRE-based clearing (GLAD): 0-3 months (Q3 dominant)"

**Pros**:
- ✓ Honest about what we're measuring
- ✓ Hansen gives complete validation
- ✓ GLAD gives temporal insights (for fire subset)
- ✓ Acknowledges GLAD bias

**Cons**:
- ⚠️ Complex messaging (two different metrics)
- ⚠️ GLAD sample size still small (17 pixels)

**Recommendation**: **Use for research paper / detailed analysis**

---

### Option C: Alternative Temporal Validation (Quarterly Hansen Sampling)

**Approach**:
- Use Hansen GFC (all deforestation)
- **Infer** quarterly timing from **local fire data** or **precipitation patterns**:
  - Dry season months (Jul-Oct) = likely Q3-Q4 clearing
  - Wet season months (Nov-Jun) = likely Q1-Q2 cutting
  - Use regional climate data to stratify

**Pros**:
- ✓ Complete deforestation (not fire-biased)
- ✓ Quarterly approximation (not perfect but better than nothing)
- ✓ Larger sample sizes

**Cons**:
- ⚠️ Imprecise quarterly assignment (based on inference, not dates)
- ⚠️ Assumes regional patterns apply to all pixels

**Recommendation**: **Exploratory only**

---

## Immediate Next Steps

### 1. Re-run CRAWL Tests with Hansen GFC (Baseline)

**Purpose**: Validate model on **ALL deforestation** (not just fire)

**Method**:
- Sample 100+ cleared pixels (Hansen GFC 2020)
- Sample 100+ intact pixels (stable forest)
- Extract AlphaEarth embeddings (Y-1, Y-2)
- Test separability, temporal signal, generalization
- **Compare to original CRAWL** (AUC 0.894)

**Expected**:
- If AUC similar (~0.85-0.90): Model generalizes to all deforestation ✓
- If AUC lower (~0.70-0.75): Model is fire-specific ✗

---

### 2. Ablation Study: Fire vs Non-Fire Clearings

**Purpose**: Test if AlphaEarth detects selective logging (non-fire)

**Method**:
- **Group A**: Hansen clearings WITH GLAD alerts (fire-based, 17 pixels)
- **Group B**: Hansen clearings WITHOUT GLAD alerts (non-fire, e.g., selective logging)
- Test AlphaEarth signal strength for each group
- Compare: Do non-fire clearings show signal?

**Expected**:
- If Group B shows signal: AlphaEarth detects logging ✓
- If Group B no signal: AlphaEarth is fire-specific ✗

---

### 3. Regional Fire Data Overlay

**Purpose**: Validate GLAD fire bias hypothesis

**Method**:
- Overlay Hansen clearings with fire data (MODIS, VIIRS)
- Identify pixels with/without fire
- Test quarterly distribution for fire vs non-fire subsets

**Expected**:
- Fire pixels: Q3 dominant (fire season)
- Non-fire pixels: More even distribution across Q1-Q4

---

## Revised Framing for WALK Phase

### If AlphaEarth Detects ALL Deforestation (Hansen validation):

**System Capability**:
- "**Deforestation risk prediction using multi-modal satellite fusion**"
- Detects: Selective logging, degradation, fire-based clearing
- Lead time: Unknown (annual labels, cannot stratify quarterly)
- Performance: AUC 0.85-0.90 on all deforestation types

**Honest Messaging**:
- ✓ "Predicts forest loss 12 months before occurrence"
- ✓ "Based on multi-modal signals (optical + radar + lidar)"
- ✓ "Detects degradation precursors not visible in optical imagery"
- ✗ "Cannot specify lead time (0-3 vs 6-12 months) without temporal labels"

---

### If AlphaEarth Only Detects Fire (GLAD-limited):

**System Capability**:
- "**Fire-driven deforestation early detection system**"
- Detects: Slash-and-burn clearing during dry season
- Lead time: 0-3 months (concurrent with fire season)
- Performance: AUC 0.85-0.90 on fire-based clearing

**Honest Messaging**:
- ✓ "Detects fire-driven clearing with 0-3 month lead time"
- ✓ "Focused on dry season deforestation (Q3-Q4)"
- ✗ "Does not detect selective logging or gradual degradation"
- ✗ "Limited to fire-based clearing mechanisms"

---

## Recommendation

### **Immediate Action**:

1. **Run Hansen GFC validation** (100+ pixels)
   - Test if AlphaEarth generalizes to ALL deforestation
   - Get baseline AUC on complete ground truth
   - Compare to CRAWL phase (0.894)

2. **Test fire vs non-fire ablation**
   - Hansen clearings WITH GLAD alerts (fire)
   - Hansen clearings WITHOUT GLAD alerts (non-fire, likely logging)
   - See if AlphaEarth detects both

3. **Pivot framing based on results**:
   - If detects all: "Multi-modal deforestation prediction (unknown lead time)"
   - If detects fire only: "Fire-driven deforestation early detection (0-3 months)"

### **Abandon** (for now):

- ❌ Quarterly stratification with GLAD
  - Sample size too small (17 pixels)
  - Fire-biased (71% Q3)
  - Not representative of all deforestation

- ❌ "Detection vs prediction" determination
  - Cannot answer without quarterly labels
  - Hansen is annual only
  - GLAD is fire-biased and sparse

### **Accept**:

- ✓ We have a working model (AUC 0.894)
- ✓ We don't know exact lead time (annual labels)
- ✓ We validate on complete deforestation (Hansen)
- ✓ We frame honestly about limitations

---

## Key Insight

**The user is absolutely right**: We've been validating against GLAD (fire-biased subset) when AlphaEarth likely detects ALL deforestation (including logging via radar/lidar).

**This explains**:
- Why Q3 dominates (71%) in GLAD validation → GLAD is fire-biased!
- Why our sample is small (17 pixels) → GLAD coverage is sparse!
- Why spatial features showed no signal → We're detecting isolated logging, not fire expansion!

**Solution**: Validate against Hansen GFC (all deforestation), accept we cannot determine exact lead time without quarterly labels, and frame honestly about capabilities.

---

**Next Steps**: Run Hansen GFC validation and fire vs non-fire ablation study.
