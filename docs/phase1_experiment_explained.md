# Phase 1 Experiment: Understanding the Temporal Signal

**Goal**: Understand what AlphaEarth embeddings are actually detecting

---

## The Story So Far

### What We Built (CRAWL Phase)

We trained a model that predicts deforestation using AlphaEarth embeddings:

```
Input: AlphaEarth embedding from Year Y-1 (e.g., 2019)
Output: Will this pixel be cleared in Year Y? (e.g., 2020)
Result: 89.4% AUC - EXCELLENT!
```

### The Problem: What Does This Really Mean?

Our model says: **"This pixel will be cleared in 2020 based on 2019 data"**

But **WHEN in 2020** did the clearing happen?
- January 2020? (early in year)
- July 2020? (middle of year)
- December 2020? (late in year)

**This matters because it changes what our model means:**

---

## Scenario A: Early Detection (Not Prediction)

### Timeline
```
2019 Annual Embedding
├─ Jan-Mar: Normal forest
├─ Apr-Jun: Normal forest
├─ Jul-Sep: Normal forest
└─ Oct-Dec: Normal forest
         ↓
         January-March 2020: CLEARING HAPPENS HERE!
         ↓
2020 Annual Embedding
├─ Jan-Mar: CLEARED LAND (captured in annual aggregate!)
├─ Apr-Jun: Cleared land
├─ Jul-Sep: Cleared land
└─ Oct-Dec: Cleared land
```

### What We're Actually Detecting

**AlphaEarth annual embeddings aggregate observations throughout the year.**

If clearing happened in **Q1 2020** (Jan-Mar):
- The "2020 annual embedding" includes images from Q1-Q4 2020
- Therefore it **already includes the clearing** (from Q1)
- So when we compare "2019 embedding" vs "2020 embedding":
  - 2019 = intact forest all year
  - 2020 = cleared in Q1, then 9 months of cleared land
  - **Big difference!** → Model sees this as "predictive"

**But it's not really prediction** - the clearing already happened early in 2020, and the annual 2020 embedding captured it.

**Lead time**: ~0-3 months (we're just barely ahead of the annual label)

---

## Scenario B: True Precursor Signal (Real Prediction)

### Timeline
```
2019 Annual Embedding
├─ Jan-Mar: Normal forest
├─ Apr-Jun: Normal forest
├─ Jul-Sep: Normal forest
└─ Oct-Dec: PRECURSOR ACTIVITIES (roads, camps, selective logging)
         ↓
         [2019 embedding captures these preparations]
         ↓
2020 (all year): Still mostly forest, preparations continue
         ↓
         October-December 2020: CLEARING HAPPENS HERE!
         ↓
2021 Annual Embedding
├─ Jan-Mar: Shows post-clearing
├─ Apr-Jun: Cleared land
├─ Jul-Sep: Cleared land
└─ Oct-Dec: Cleared land
```

### What We're Actually Detecting

If clearing happened in **Q4 2020** (Oct-Dec):
- The "2019 annual embedding" captured **late-year precursor activities**:
  - Road building in Q4 2019
  - Camp establishment
  - Selective logging
  - Land clearing permits
- The "2020 annual embedding" mostly shows intact forest (Jan-Sep) + clearing (Q4)
- When we compare 2019 vs 2020:
  - 2019 = intact forest + late-year precursors
  - 2020 = intact forest most of year + late-year clearing
  - **Difference** → Model sees precursor activities

**This IS real prediction** - we're detecting preparation activities that precede clearing.

**Lead time**: 9-15 months (Q4 2019 precursors → Q4 2020 clearing)

---

## The Core Question

**Which scenario are we in?**

| Scenario | What Y-1 Captures | Clearing Quarter | Lead Time | System Type |
|----------|-------------------|------------------|-----------|-------------|
| **Early Detection** | Normal forest | Q1 (Jan-Mar) | 0-3 months | Glorified change detection |
| **True Precursor** | Preparation activities | Q4 (Oct-Dec) | 9-15 months | Early warning system |
| **Mixed** | Both | Q2 + Q4 | Variable | Useful but unclear |

**Without knowing WHEN clearing happened, we can't distinguish these scenarios!**

---

## How GLAD-S2 Helps Us

### The Problem with Hansen GFC

Hansen Global Forest Change (our current labels) only tells us:
- "This pixel was cleared in 2020"
- **Does NOT tell us**: January? March? July? December?

### What GLAD-S2 Provides

GLAD (Global Land Analysis & Discovery) alerts provide:
- **Week-level clearing dates**
- Based on Landsat (30m) or Sentinel-2 (10m)
- For our Amazon study region

**Example GLAD data**:
```python
Clearing Location: (-9.38°, -57.76°)
Hansen GFC: "Cleared in 2020"
GLAD: "Alert detected: Day 231 of 2020 = August 18, 2020"
       → Quarter 3 (July-September)
```

### Why This Matters

With GLAD dates, we can **stratify clearings by when they happened**:

```
All 2020 clearings (n=53)
├─ Q1 clearings (Jan-Mar): 1 clearing
├─ Q2 clearings (Apr-Jun): 6 clearings
├─ Q3 clearings (Jul-Sep): 19 clearings
└─ Q4 clearings (Oct-Dec): 3 clearings
```

Now we can test: **Does the 2019 embedding predict Q1 clearings differently than Q4 clearings?**

---

## The Original Test: Q1 vs Q4

### Hypothesis

**If we have a true precursor signal:**
- 2019 embedding should **strongly predict Q4 2020 clearings**
  - Why? Late 2019 precursors → Late 2020 clearing
  - Big signal: preparation → execution

- 2019 embedding should **weakly predict Q1 2020 clearings**
  - Why? Q1 2020 clearing already happened early in year
  - The 2020 annual embedding captured it (early detection, not precursor)

**If we have early detection only:**
- 2019 embedding should **strongly predict Q1 2020 clearings**
  - Why? Q1 clearing is captured in 2020 annual embedding
  - We're just detecting the change

- 2019 embedding should **weakly predict Q4 2020 clearings**
  - Why? Q4 clearing happens late, might not be well-captured in annual aggregate

### The Test

**Metric**: Embedding distance from Y-1 to Y

For each clearing with GLAD date:
1. Get 2019 embedding (Y-1)
2. Get 2020 embedding (Y)
3. Calculate distance: `d = ||emb_2020 - emb_2019||`
4. Group by quarter

**Expected results**:

| Scenario | Q1 Distance | Q4 Distance | Interpretation |
|----------|-------------|-------------|----------------|
| **True Precursor** | 0.6 (small) | 0.9 (large) | Y-1 strongly predicts Q4, weakly predicts Q1 |
| **Early Detection** | 0.9 (large) | 0.6 (small) | Y-1 strongly predicts Q1, weakly predicts Q4 |
| **Mixed Signal** | 0.75 | 0.75 | Similar for both quarters |

### Why This Test Failed

**Problem**: Q1 clearings are extremely rare (1 out of 29 = 3.4%)

**Reason**: Amazon wet season
- January-March: Heavy rains
- Difficult to clear forest, burn, and establish crops/pasture
- Most clearing happens Q2-Q3 (dry season: May-September)

**Result**: Can't compare Q1 vs Q4 (need n≥3 for statistics)

---

## The New Test: Q2 vs Q4

### Why Q2 vs Q4?

**Q2 (April-June)**:
- Early-to-mid year
- Clearing captured in annual 2020 embedding
- Still mostly intact in 2019 embedding
- **Represents "early detection" scenario**

**Q4 (October-December)**:
- Late year
- Clearing happens at END of 2020
- 2020 annual embedding is mostly intact forest (Jan-Sep) + late clearing (Oct-Dec)
- **Represents "precursor signal" scenario** (if precursors in 2019)

**We have data**: Q2 n=6, Q4 n=3 (both sufficient for t-test)

### What Q2 vs Q4 Tells Us

#### Hypothesis 1: True Precursor Signal

**Prediction**: Q4 distance > Q2 distance

**Logic**:
```
Q4 Clearing:
2019 embedding: Intact forest + late-year precursors (roads, camps)
2020 embedding: Intact forest (Jan-Sep) + clearing (Oct-Dec)
Distance: LARGE (precursors → clearing)

Q2 Clearing:
2019 embedding: Intact forest (all year)
2020 embedding: Clearing (Apr-Jun) + 6 months cleared land (Jul-Dec)
Distance: MEDIUM (already cleared, captured in annual)
```

**Result if true**: Q4 mean distance > Q2 mean distance
**Interpretation**: We're detecting preparation activities in Y-1 that precede Q4 clearing
**System type**: Early warning (9-15 month lead time)

---

#### Hypothesis 2: Early Detection

**Prediction**: Q2 distance > Q4 distance

**Logic**:
```
Q2 Clearing:
2019 embedding: Intact forest
2020 embedding: Heavily weighted toward cleared state (8 months cleared)
Distance: LARGE (strong clearing signal in annual aggregate)

Q4 Clearing:
2019 embedding: Intact forest
2020 embedding: Mostly intact (9 months) + late clearing (3 months)
Distance: SMALL (annual aggregate still looks mostly forested)
```

**Result if true**: Q2 mean distance > Q4 mean distance
**Interpretation**: Annual embeddings are temporally weighted toward mid-year
**System type**: Detection system (0-6 month lead time)

---

#### Hypothesis 3: Mixed Signal

**Prediction**: Q2 distance ≈ Q4 distance

**Logic**: Both precursor detection and early detection are happening

**Result if true**: Similar mean distances
**Interpretation**: System provides value but with variable lead time
**System type**: Risk prediction (variable 3-12 month lead time)

---

## What We Actually Observed

### Our Results

| Quarter | n | Mean Distance | Std Dev | p-value |
|---------|---|---------------|---------|---------|
| Q2 | 6 | **0.782** | 0.139 | < 0.001 |
| Q3 | 19 | **0.778** | 0.234 | < 0.001 |
| Q4 | 3 | **0.376** | 0.102 | 0.017 |

### Key Finding: Q4 < Q2/Q3

**Q4 distance (0.376) is ~50% lower than Q2/Q3 (0.78)**

### What This Means

This pattern matches **Hypothesis 2: Early Detection**

**Interpretation**:
1. **Annual embeddings are temporally weighted toward mid-year (Q2-Q3)**
   - More clear-sky observations in dry season (May-September)
   - Q2-Q3 clearings heavily influence annual aggregate
   - Q4 clearings under-represented (only 3 months of clearing in annual)

2. **Q4 clearings show weaker signal (0.376) because:**
   - 2020 annual embedding = 9 months intact (Jan-Sep) + 3 months cleared (Oct-Dec)
   - Looks more similar to 2019 (all intact) than to a full-year clearing
   - Less dramatic change → smaller distance

3. **Q2-Q3 clearings show stronger signal (0.78) because:**
   - 2020 annual embedding = few months intact + 8-10 months cleared
   - Looks very different from 2019 (all intact)
   - More dramatic change → larger distance

### Implication

**We're likely detecting mid-year clearing (Q2-Q3), not precursor activities.**

This is still valuable for:
- ✅ Annual risk modeling
- ✅ Resource allocation
- ✅ Identifying high-risk areas for monitoring

But it's NOT:
- ❌ A true "early warning system" with 9-15 month lead time
- ❌ Detecting precursor activities (roads, camps) before clearing

---

## Q2 vs Q4 Statistical Test

### What We'll Do

**Null Hypothesis (H0)**: Q2 and Q4 have the same mean distance
**Alternative (H1)**: Q2 and Q4 have different mean distances

**Test**: Two-sample t-test or Mann-Whitney U test (if non-normal)

**Current data**:
- Q2: n=6, mean=0.782, std=0.139
- Q4: n=3, mean=0.376, std=0.102

### Expected Results

**Scenario 1: Q2 > Q4 (statistically significant)**
- p < 0.05
- **Conclusion**: Annual embeddings weight mid-year → **Early detection system**
- **Lead time**: 0-6 months (detecting clearings that happen Q2-Q3)
- **Value**: Still useful for annual risk, but honest framing needed

**Scenario 2: Q2 ≈ Q4 (not significant)**
- p > 0.05
- **Conclusion**: No clear temporal pattern → **Mixed signal**
- **Lead time**: Variable 3-12 months
- **Value**: Provides risk scores, but uncertain lead time

**Scenario 3: Q4 > Q2 (unlikely given our data)**
- p < 0.05, Q4 mean > Q2 mean
- **Conclusion**: True precursor signal → **Early warning system**
- **Lead time**: 9-15 months
- **Value**: High - can predict clearings well in advance

### Why This Is Useful

Even though we don't have Q1 data, **Q2 vs Q4 still answers our core question**:

**"Does the Y-1 embedding capture precursor signals (roads, camps in late Y-1 that precede late-Y clearing)?"**

- **If Q4 > Q2**: Yes → Precursor signal
- **If Q2 > Q4**: No → Early detection of mid-year clearing
- **If Q2 ≈ Q4**: Mixed → Some of both

---

## Visual Summary

### What Happens in Each Quarter

```
                 2019 (Year Y-1)              2020 (Year Y)
                 ===============              =============

Q1 Clearing:    [Intact all year]  →  [CLEAR Q1][Cleared 9mo]
                                      ↑ Large change (0.9 expected)
                                      ↑ Captured in 2020 annual embedding
                                      ↑ EARLY DETECTION

Q2 Clearing:    [Intact all year]  →  [Intact Q1][CLEAR Q2][Cleared 6mo]
                                      ↑ Large change (0.78 observed)
                                      ↑ Heavily captured in annual
                                      ↑ EARLY-MID DETECTION

Q3 Clearing:    [Intact all year]  →  [Intact Jan-Jun][CLEAR Q3][Cleared 3mo]
                                      ↑ Large change (0.78 observed)
                                      ↑ Heavily captured in annual
                                      ↑ MID-YEAR DETECTION

Q4 Clearing:    [Intact + precursors?]  →  [Intact 9mo][CLEAR Q4]
                                           ↑ Small change (0.38 observed)
                                           ↑ Under-represented in annual
                                           ↑ If precursors: EARLY WARNING
                                           ↑ If no precursors: WEAK DETECTION
```

### The Test

```
Q2 Distance (0.78)     vs     Q4 Distance (0.38)
        ↓                              ↓
    Q2 > Q4?                      Q4 < Q2?
        ↓                              ↓
   YES (observed)                 YES (observed)
        ↓                              ↓
   Annual embeddings weight mid-year
        ↓
   EARLY DETECTION, not precursor signal
        ↓
   Lead time: 0-6 months
   Still useful for annual risk modeling
```

---

## Next Steps

### Option 1: Run Q2 vs Q4 Statistical Test ⭐ **RECOMMENDED**

**What I'll do**:
1. Perform two-sample t-test (Q2 vs Q4 distances)
2. Calculate effect size (Cohen's d)
3. Interpret statistical significance
4. Write final conclusion

**Time**: 15 minutes

**Output**:
- Statistical test results
- Definitive interpretation of temporal signal
- Updated documentation
- GO/NO-GO decision for WALK phase

### Option 2: Scale to 300 Samples (Try for Q1 Data)

**Goal**: Get 5-10 Q1 samples for original Q1 vs Q4 test

**Risk**: Q1 deforestation may be <3% → still insufficient
**Time**: 45-60 minutes compute
**Value**: Original test is cleaner, but Q2 vs Q4 already tells us the answer

### Option 3: Accept Findings, Move to WALK

**Decision**: We have enough evidence for early detection vs precursor
**Action**: Proceed with honest framing (annual risk model, not early warning)
**Focus**: Build robust spatial CV, not temporal investigation

---

## My Recommendation

**Run Option 1 (Q2 vs Q4 test) right now.**

Given what we've observed (Q4 = 0.38 << Q2 = 0.78), I'm 90% confident the test will show:
- **Q2 > Q4 with p < 0.05**
- **Interpretation**: Early detection of mid-year clearing, not precursor signal
- **Lead time**: 0-6 months (still valuable for annual risk)
- **System type**: Annual risk model, not early warning

This gives us a **definitive answer** to proceed to WALK phase with the right framing.

**Shall I run the Q2 vs Q4 statistical test?**
