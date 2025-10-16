# Deforestation Seasonal Patterns: Literature Review

**Date**: 2025-10-15
**Purpose**: Understand when deforestation typically occurs in Amazon and Congo Basin to contextualize our findings

---

## Amazon Region Deforestation Timing

### **Dry Season and Fire Season**

**Key Timing**:
- **Dry season**: July through October
- **Fire season**: August through October (peak: August-September)
- **Wet season**: November through June

**Source**: NASA Earth Observatory, INPE, Mongabay

### **Deforestation Process Timeline**

According to scholarly literature, the typical slash-and-burn deforestation follows this pattern:

**Phase 1: Cutting (Wet Season)**
- **Timing**: November through June
- **Activity**: Forest is cut down using bulldozers and tractors
- **Reason**: Easier access during wet season, less fire risk

**Phase 2: Drying**
- **Timing**: June through July (transition)
- **Activity**: Cut biomass left to dry
- **Duration**: Several months

**Phase 3: Burning (Dry Season)**
- **Timing**: July through October (peak: August-September)
- **Activity**: Dried biomass is burned
- **Reason**: Dry conditions enable complete burning, nutrient deposition

**Sources**:
- NASA Earth Observatory (2019, 2024)
- Mongabay fire season reports
- INPE fire monitoring

### **Regional Variations**

**Northern Amazon**:
- Fires begin: February
- Peak: March
- Dry season timing varies by latitude

**Southern Amazon** (Brazilian Amazon):
- Fires begin: July
- Peak: August-September
- Strongest dry season

**Source**: Rainforest Foundation, NASA

### **Changing Patterns**

**Historical**: Fire season lasted 3 months (August-October)

**Current**: Fire season now extends to 6+ months, approaching year-round in some areas

**Source**: Greenpeace, Rainforest Foundation (2024)

### **PRODES Monitoring Calendar**

**Important**: Brazil's official deforestation monitoring (PRODES) measures **August (year t-1) through July (year t)**, NOT calendar years!

This means:
- "2020 deforestation" in PRODES = Aug 2019 - Jul 2020
- Aligns with fire season timing (Aug-Oct is START of monitoring year)

**Source**: INPE PRODES documentation, NASA

---

## Congo Basin Deforestation Timing

### **Bimodal Seasonal Pattern**

The Congo Basin straddles the equator, creating **two dry seasons and two wet seasons**:

**Dry Season 1**: December-January-February (DJF)
**Wet Season 1**: March-April-May (MAM)
**Dry Season 2**: June-July-August (JJA)
**Wet Season 2**: September-October-November (SON)

**Source**: Climate Dynamics, NASA Earth Observatory

### **Slash-and-Burn Timing**

**Northern Congo Basin** (north of equator):
- Clearing/burning: June-August (JJA dry season)
- Farmers rely on long dry season to ensure fields burn well

**Southern Congo Basin** (south of equator):
- Clearing/burning: December-February (DJF dry season)

**Mayombe Forest** (Republic of Congo):
- Fires only occur: August-October
- After cutting and felling during earlier months

**Source**: ResearchGate, ScienceDirect, Rainforest Journalism Fund

### **Agricultural Cycle**

**Process**:
1. Forest cutting: Months before dry season
2. Vegetation drying: Transition period
3. Burning: During dry season (ensures complete burn)
4. Planting: At start of next wet season

**Critical dependency**: Strength of dry season directly impacts:
- Size of fields cleared
- Effectiveness of burning
- Food production

**Source**: ScienceDirect (Wetter isn't better: global warming and food security in Congo Basin)

### **Changing Dry Season**

**Trend**: Summer dry season (JJA) growing longer
- **Rate**: 6-10 days per decade (1988-2013)
- **Some areas**: Up to 30 days per decade
- **Implication**: Earlier start, later end of dry season → longer burning window

**Source**: NASA Earth Observatory, Climate Dynamics

---

## Quarterly Deforestation Distribution (Expected)

Based on the literature, we would expect the following **quarterly distribution** of deforestation:

### **Amazon (Southern/Brazilian)**

| Quarter | Months | Activity | Expected % |
|---------|--------|----------|------------|
| Q1 | Jan-Mar | Wet season cutting (ongoing) | 15-20% |
| Q2 | Apr-Jun | Late wet season cutting, drying starts | 20-25% |
| Q3 | Jul-Sep | **Peak burning season** | **30-35%** |
| Q4 | Oct-Dec | Late burning, wet season begins | 20-25% |

**Expected peak**: **Q3 (July-September)** - aligns with fire season

**Key insight**: If using GLAD alerts (detects burning), we'd expect:
- **Q3 dominant** (Aug-Sep peak burning)
- **Q4 significant** (Oct burning continues)
- **Q1-Q2 lower** (cutting phase, less detectable)

### **Congo Basin**

**Highly regional** due to bimodal pattern:

**Northern Basin** (north of equator):
- Peak: Q2-Q3 (June-August dry season)

**Southern Basin** (south of equator):
- Peak: Q4-Q1 (December-February dry season)

**Overall expected**: More evenly distributed than Amazon, but with dual peaks

---

## Comparison with Our Findings

### **What We Found (GLAD-S2, 2020 Amazon)**

From our Q2 vs Q4 test:

| Quarter | Sample Size | Embedding Distance | Interpretation |
|---------|-------------|-------------------|----------------|
| Q1 | 1 pixel | N/A | Insufficient data |
| Q2 | 2 pixels | **0.78 ± 0.14** | **STRONG signal** |
| Q3 | 8 pixels | N/A | Not tested separately |
| Q4 | 2 pixels | **0.38 ± 0.10** | WEAK signal |

**Combined Q2-Q3**: Assumed ~60% of clearings based on Q2 strength

### **Critical Discrepancy** ⚠️

**Literature predicts**: Q3 (Jul-Sep) should be **dominant** (30-35% or more)
- Peak burning: August-September
- GLAD detects fires and burning
- Fire season well-documented

**Our findings suggest**: Q2 (Apr-Jun) shows **strongest signal**
- Q2 embedding distance: 0.78 (2x larger than Q4)
- Q4 (Oct-Dec) shows weak signal: 0.38

### **Possible Explanations**

**1. Sample Size Too Small** ⚠️ **MOST LIKELY**
- We only had 13 pixels with GLAD dates out of 24 sampled
- Q1=1, Q2=2, Q3=8, Q4=2
- Not enough Q3 pixels tested separately
- Q2-Q3 merged in analysis → attributed to Q2

**2. Regional Differences**
- Our study region may have different patterns
- Not all Amazon follows same seasonal pattern
- Northern vs Southern Amazon vary

**3. GLAD Detection Bias**
- GLAD-S2 may detect cutting BEFORE burning
- Optical detection of canopy loss (Q2-Q3)
- Before fire-based detection (Q3-Q4)

**4. AlphaEarth Timing**
- Annual embeddings captured ~June
- Q2 clearings (Apr-Jun) are CONCURRENT with embedding
- Q3 clearings (Jul-Sep) are AFTER embedding (weaker signal makes sense!)
- This would explain Q2 > Q4 pattern

**5. Different Clearing Type**
- Our samples may be isolated clearing (no burning)
- Not slash-and-burn agriculture
- Mechanized clearing without fire
- Would not follow fire season pattern

---

## Implications for Our Temporal Framing

### **What This Means for Lead Time**

**If literature pattern holds (Q3-Q4 dominant)**:
- June embedding → Q3 burning (Jul-Sep): **1-3 months lead time**
- June embedding → Q4 burning (Oct-Dec): **4-6 months lead time**
- **Prediction** capability is more plausible

**If our finding holds (Q2-Q3 dominant)**:
- June embedding → Q2 clearing (Apr-Jun): **0-2 months overlap (concurrent)**
- June embedding → Q3 clearing (Jul-Sep): **1-3 months lead time**
- More **early detection** than prediction

**The truth likely depends on**:
1. Whether clearing follows fire season (literature) or is mechanized/isolated
2. Regional variation (northern vs southern Amazon)
3. What GLAD-S2 detects (canopy loss vs burning)

### **Recommendation: Test with Larger Sample**

**Critical next step**: Validate quarterly distribution with larger sample
- Need 100+ pixels with GLAD dates
- Test if Q3 is actually dominant (as literature suggests)
- Separate analysis for Q2 vs Q3 vs Q4
- This will clarify whether we have:
  - **Early detection system** (if Q2-Q3 dominant, concurrent)
  - **Prediction system** (if Q3-Q4 dominant, 1-6 month lead)
  - **Mixed system** (bimodal distribution)

---

## References

### Amazon

**Fire Season Timing**:
- NASA Earth Observatory: "Reflecting on a Tumultuous Amazon Fire Season" (2020)
- Mongabay: "Blazing start to Amazon's fire season" (2022)
- Rainforest Foundation: "Fires in the Amazon Shift with the Seasons" (2024)

**Deforestation Process**:
- NASA Earth Observatory: "Tracking Amazon Deforestation from Above" (2020)
- Wikipedia: "2019 Amazon rainforest wildfires"

**Monitoring Systems**:
- INPE PRODES documentation
- NASA: "DETERring Deforestation in the Amazon" (CPI, 2019)

### Congo Basin

**Seasonal Patterns**:
- NASA Earth Observatory: "A Longer Dry Season in the Congo Rainforest" (2019)
- Climate Dynamics: "Climate change in the Congo Basin" (2019)
- Eos: "Congo Rain Forest Endures a Longer Dry Season" (2019)

**Agricultural Timing**:
- ScienceDirect: "Wetter isn't better: global warming and food security in Congo Basin" (1999)
- Rainforest Journalism Fund: "Seeking Alternatives to Slash-and-Burn Agriculture" (2023)
- Global Forest Watch: "New Map Helps Distinguish Between Cyclical Farming and Deforestation" (2020)

**Carbon Dynamics**:
- Jiang et al.: "Congo Basin Rainforest Is a Net Carbon Source During the Dry Season" (Earth and Space Science, 2023)

---

## Summary

### **When Deforestation Typically Happens**

**Amazon**: **Q3 (Jul-Sep) and Q4 (Oct-Dec)** - aligned with fire season
- Peak: **August-September**
- Process: Cutting in wet season (Q1-Q2) → Burning in dry season (Q3-Q4)

**Congo Basin**: **Bimodal** - depends on location
- Northern: Q2-Q3 (Jun-Aug)
- Southern: Q4-Q1 (Dec-Feb)

### **Implications for Our System**

**If our Q2-Q3 finding holds**: More early detection than prediction
**If literature Q3-Q4 pattern holds**: More prediction capability than we realized

**Critical validation needed**: Larger sample with quarterly breakdown to resolve this discrepancy

---

**Key Takeaway**: The literature strongly suggests Q3-Q4 should dominate in Amazon (fire season). Our Q2 strong signal may be due to:
1. Small sample size (need validation)
2. AlphaEarth detecting cutting phase (before burning)
3. Regional differences in our study area
4. Different clearing mechanisms (mechanized vs slash-and-burn)
