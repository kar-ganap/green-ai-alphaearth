# Spatial Precursor Analysis Plan

**Date**: 2025-10-15
**Motivation**: Deforestation spreads spatially from roads and edges. The "precursor signal" is likely SPATIAL (nearby features) not TEMPORAL (same pixel over time).

---

## The Core Hypothesis

**Deforestation follows a spatial diffusion model:**

```
If pixel P will be cleared in year Y:
  → Neighborhood of P in year Y-1 shows:
     - Roads within 500m
     - Recent clearing within 200m
     - Forest edge proximity
     - Accessibility indicators

NOT: Pixel P itself looks different in Y-1 vs Y-2
BUT: Pixel P's surroundings in Y-1 show precursor features
```

**Timeline**: Roads → Clearing happens within weeks-months, not years
**Mechanism**: Spatial spread from access points and edges

---

## What We Know So Far

### From CRAWL Test 4 (Minimal Model)
- **"Distance to center" feature was highly predictive** (AUC 0.894)
- This is a SPATIAL feature: pixels near previously cleared areas
- Suggests spatial autocorrelation is strong

### From Temporal Investigation
- Q4 clearings show WEAK signal (0.38 distance)
- Q2-Q3 clearings show STRONG signal (0.78 distance)
- This rules out long temporal precursors for same pixel

### What We Haven't Tested
- ✗ Spatial patterns in Y-1 embeddings around Y clearings
- ✗ Whether nearby roads/edges in Y-1 predict P clearing in Y
- ✗ Spatial autocorrelation explicitly
- ✗ Edge/frontier dynamics

---

## Proposed Investigations

### Investigation 1: Neighborhood Analysis ⭐ **HIGHEST PRIORITY**

**Question**: Do neighborhoods of cleared pixels (in Y) show different patterns in Y-1 compared to intact pixels?

**Method**:
```python
For each cleared pixel P in year Y:
  1. Extract Y-1 embeddings for:
     - Pixel P itself (center)
     - 8 neighbors (3x3 grid, 30m radius)
     - 24 neighbors (5x5 grid, 50m radius)

  2. Calculate spatial features:
     - Mean/std of neighborhood embeddings
     - Gradient magnitude (edge detection)
     - Distance between center and neighbors
     - Heterogeneity (variance in neighborhood)

  3. Compare to intact pixels:
     - Do cleared pixels have higher gradients in Y-1?
     - More heterogeneous neighborhoods?
     - Neighbors more similar to cleared signature?
```

**Expected Result**:
- **If spatial precursors exist**: Cleared pixels' Y-1 neighborhoods show higher gradients (edges), more heterogeneity (mixed forest/clearing/roads)
- **If no spatial signal**: Cleared and intact neighborhoods look similar in Y-1

**Output**:
- Visualization: Average Y-1 neighborhood for cleared vs intact pixels
- Statistical test: Gradient/heterogeneity difference
- Feature importance: Which spatial metrics predict clearing?

**Time**: 3-4 hours

---

### Investigation 2: Distance to Features ⭐ **HIGH PRIORITY**

**Question**: Are cleared pixels closer to roads/edges/recent clearing in Y-1?

**Method**:
```python
For each pixel in dataset:
  1. Calculate in Y-1:
     - Distance to nearest cleared pixel (from Y-2 or Y-1)
     - Distance to forest edge
     - Embedding similarity to "road signature"
     - Embedding similarity to "edge signature"

  2. Test spatial predictors:
     - Model: P(cleared in Y) ~ distance_to_clearing_Y-1 + distance_to_edge_Y-1
     - AUC with spatial features only
     - Compare to temporal features (Y-1 vs Y-2 distance)
```

**Expected Result**:
- **If spatial precursors dominant**: Distance to recent clearing is strong predictor
- **If temporal matters too**: Both spatial and temporal contribute

**Feature Engineering**:
```python
spatial_features = {
    'distance_to_clearing_y1': ...,      # Distance to nearest Y-1 clearing
    'distance_to_clearing_y2': ...,      # Distance to nearest Y-2 clearing
    'edge_proximity_y1': ...,            # Distance to forest edge in Y-1
    'clearing_density_500m': ...,        # % cleared within 500m in Y-1
    'frontier_indicator': ...,           # Binary: is this near deforestation frontier?
}
```

**Output**:
- Feature importance ranking
- AUC with spatial features
- Maps: Cleared pixels colored by distance to Y-1 features

**Time**: 4-5 hours

---

### Investigation 3: Spatial Autocorrelation ⭐ **MEDIUM PRIORITY**

**Question**: Do clearings cluster in space? Do predictions cluster?

**Method**:
```python
1. Global Moran's I:
   - Test if clearings are spatially clustered
   - Test if model predictions are spatially clustered

2. Local Moran's I (LISA):
   - Identify hotspots (high clearing, high-risk neighbors)
   - Identify coldspots (low clearing, low-risk neighbors)

3. Variogram analysis:
   - How far does spatial correlation extend? (100m? 1km? 10km?)
```

**Expected Result**:
- **Positive spatial autocorrelation**: Clearings cluster in space
- **Range**: ~500m-2km (typical deforestation patch sizes)
- **Implications**: Need spatial CV to avoid overfitting

**Output**:
- Moran's I statistic and p-value
- LISA cluster maps
- Variogram plot (correlation vs distance)

**Time**: 2-3 hours

---

### Investigation 4: Edge/Frontier Dynamics ⭐ **MEDIUM PRIORITY**

**Question**: Does clearing happen preferentially at forest edges? How fast do edges advance?

**Method**:
```python
1. Define forest edge in Y-1:
   - Pixels with forest on one side, non-forest on other
   - Use NDVI or Hansen tree cover

2. Calculate for each pixel:
   - Distance to edge in Y-1
   - Edge advancement rate (edge position Y-1 vs Y-2)
   - "Frontier region" indicator (active deforestation zone)

3. Test edge hypothesis:
   - P(cleared in Y) ~ distance_to_edge_Y-1
   - Compare edge vs interior forest clearing rates
```

**Expected Result**:
- **Edge effect**: Clearing probability decays with distance from edge
- **Typical range**: 80% of clearing within 500m of edge
- **Frontier regions**: Higher clearing rates

**Output**:
- Clearing probability vs distance to edge plot
- Frontier region map
- Edge advancement rate statistics

**Time**: 3-4 hours

---

### Investigation 5: Road Detection (Exploratory) ⭐ **OPTIONAL**

**Question**: Can we detect roads in Y-1 embeddings? Do they predict clearing?

**Method**:
```python
1. Road signature:
   - Sample known road pixels from OSM or manual labels
   - Get their AlphaEarth embeddings
   - Define "road signature" = mean embedding of road pixels

2. Road proximity:
   - For each pixel, calculate similarity to road signature
   - Distance to pixels with high road-similarity

3. Test road hypothesis:
   - P(cleared in Y) ~ road_similarity_Y-1
   - Visualize: Do cleared pixels have nearby high-road-similarity pixels?
```

**Expected Result**:
- **If roads detectable**: High road-similarity in Y-1 predicts Y clearing
- **If not**: Road signal too subtle in 10m embeddings

**Output**:
- Road signature visualization (t-SNE of embeddings)
- Road similarity maps
- Road proximity vs clearing probability

**Time**: 5-6 hours (requires road labels)

---

## Synthesis: Spatial vs Temporal Signal

### Comparison Test

**Build three models and compare AUC:**

```python
Model 1: Temporal only
  Features:
    - Embedding distance (Y-1 to Y-2)
    - Embedding velocity
  Expected AUC: ~0.70-0.75

Model 2: Spatial only
  Features:
    - Distance to Y-1 clearing
    - Distance to forest edge
    - Neighborhood heterogeneity
    - Clearing density 500m
  Expected AUC: ~0.85-0.90 (if hypothesis is right!)

Model 3: Spatial + Temporal
  Features: All of above
  Expected AUC: ~0.90-0.93
```

**Interpretation:**
- If **Spatial >> Temporal**: Precursors are spatial (roads, edges)
- If **Temporal >> Spatial**: Precursors are temporal (same pixel changes)
- If **Both contribute**: Mixed signal (both matter)

---

## Implementation Priority

### Phase 1: Quick Validation (4-6 hours) ⭐ **START HERE**

1. **Investigation 1**: Neighborhood analysis (3-4h)
   - Visualize cleared vs intact neighborhoods in Y-1
   - Calculate spatial gradients/heterogeneity
   - Quick statistical test

2. **Investigation 2**: Distance to features (2-3h)
   - Calculate distance to Y-1 clearings
   - Test simple spatial model
   - Compare AUC to temporal baseline

**Decision Point**:
- If spatial signal is strong (AUC > 0.85) → Proceed to Phase 2
- If weak (AUC < 0.75) → Reconsider hypothesis

### Phase 2: Detailed Analysis (8-10 hours)

3. **Investigation 3**: Spatial autocorrelation (2-3h)
4. **Investigation 4**: Edge dynamics (3-4h)
5. **Synthesis**: Compare spatial vs temporal (3h)

### Phase 3: Feature Engineering (4-6 hours)

6. Implement spatial features in production pipeline
7. Integrate with WALK phase
8. Update model with spatial + temporal features

**Total time**: 16-22 hours across all phases

---

## Expected Outcomes

### Scenario A: Strong Spatial Signal (Most Likely)

**Finding**: Distance to Y-1 clearing is strongest predictor (AUC > 0.85)

**Interpretation**:
- Deforestation spreads spatially from existing clearing
- Roads/accessibility drive clearing patterns
- Timeline: Weeks-months after road access

**Mechanism**: Spatial diffusion model
- Y-1 clearing/roads → Y clearing nearby
- Lead time: 6-12 months (Y-1 features → Y outcomes)
- Causal: YES - spatial accessibility is mechanism

**System framing**:
> "Spatial risk model: Identifies high-risk pixels based on proximity to roads, recent clearing, and forest edges. Predicts 6-12 months in advance where clearing is likely to spread."

**Value**:
- ✓ Mechanistically grounded
- ✓ Actionable (target enforcement near edges/roads)
- ✓ Honest framing (spatial spread, not distant temporal precursors)

---

### Scenario B: Mixed Spatial + Temporal (Possible)

**Finding**: Both spatial (0.85 AUC) and temporal (0.70 AUC) contribute

**Interpretation**:
- Spatial: Deforestation spreads from roads/edges
- Temporal: Some pixels show degradation before clearing

**Mechanism**: Multi-factor
- Spatial accessibility + temporal degradation
- Different pathways for different clearings

**System framing**:
> "Multi-signal risk model: Combines spatial spread patterns with temporal degradation signals. Identifies both frontier expansion and interior degradation."

**Value**:
- ✓ More complete picture
- ✓ Can distinguish clearing types
- ✓ Higher accuracy

---

### Scenario C: Weak Spatial Signal (Unlikely)

**Finding**: Spatial features don't predict well (AUC < 0.75)

**Interpretation**:
- Our hypothesis was wrong
- Either temporal dominates OR model uses other patterns

**Next steps**:
- Investigate what model IS using
- Consider other feature types
- May need to pivot approach

---

## Key Advantage: This Is Mechanistically Grounded

### Why Spatial Precursors Make Sense

**Compared to "9-month temporal precursors" (which we questioned):**

| Factor | Temporal (Same Pixel) | Spatial (Nearby Pixels) |
|--------|----------------------|------------------------|
| **Mechanism** | Pixel changes 9 months before clearing? | Roads/edges present in Y-1 → clearing spreads |
| **Timeline** | 9-15 months (questionable) | Weeks-months (realistic) |
| **Physical constraint** | None (why wait 9 months?) | Need road access (concrete) |
| **Literature support** | Weak | Strong (spatial diffusion models) |
| **Actionable** | Hard to act on pixel changes | Easy: monitor edges/roads |

**Spatial model aligns with:**
- ✓ Frontier economics (clearing spreads from roads)
- ✓ Physical constraints (need access)
- ✓ Empirical observations (clearing clusters in space)
- ✓ Policy/enforcement (target frontier regions)

---

## Integration with WALK Phase

### If Spatial Analysis Confirms Hypothesis

**Feature engineering**:
```python
features = {
    # Spatial features (new!)
    'distance_to_clearing_y1': ...,
    'distance_to_edge_y1': ...,
    'clearing_density_500m': ...,
    'neighborhood_heterogeneity': ...,
    'frontier_indicator': ...,

    # Temporal features (existing)
    'embedding_velocity': ...,
    'embedding_distance': ...,

    # AlphaEarth embedding (existing)
    'embedding_y1': ...  # 64 dims
}
```

**Spatial CV becomes even more critical**:
- Can't have training and test pixels near each other
- Spatial autocorrelation would inflate performance
- Need ~5-10km buffer between folds

**Model interpretation**:
- Can now explain WHY pixels are high-risk (near roads/edges)
- Can map deforestation frontiers
- Can prioritize enforcement regions

---

## Recommendation

**Run Phase 1 (Investigations 1-2) before proceeding to WALK.**

**Why:**
1. **Quick** (4-6 hours)
2. **Definitive** - Will tell us if spatial hypothesis is right
3. **Actionable** - Results directly inform WALK feature engineering
4. **Mechanistic** - Gives us real causal story

**If spatial signal is strong:**
- ✓ Proceed to WALK with spatial + temporal features
- ✓ Use spatial CV properly
- ✓ Frame as "spatial risk model" not "temporal early warning"
- ✓ Have mechanistically grounded story

**If spatial signal is weak:**
- Investigate what model IS using
- May need to pivot approach
- Better to know now than after building full WALK pipeline

---

## Next Steps

1. **Run Investigation 1 (Neighborhood Analysis)**
   - Visualize Y-1 neighborhoods of Y clearings
   - Calculate spatial gradients
   - Quick proof-of-concept

2. **If promising → Run Investigation 2 (Distance Features)**
   - Build simple spatial model
   - Compare to temporal baseline
   - Make GO/NO-GO decision

3. **If spatial strong → Integrate into WALK**
   - Add spatial features
   - Use spatial CV
   - Update framing

**Estimated time to decision point: 4-6 hours**

---

**Should we start with Investigation 1 (Neighborhood Analysis)?**
