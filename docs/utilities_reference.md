# Utilities Reference

Complete reference for the utility modules.

## Module: `src/utils/config.py`

**Purpose:** Configuration management and loading.

### Key Classes

#### `Config`
- Loads and manages `config.yaml`
- Provides dot-notation access to config values
- Handles path resolution

**Example:**
```python
from src.utils.config import get_config

config = get_config()

# Access config values
n_estimators = config.get("model.xgboost.n_estimators")  # 200
embedding_dims = config.embedding_dimensions  # 64

# Get absolute paths
data_dir = config.get_path("paths.data_dir")  # /path/to/project/data
```

## Module: `src/utils/earth_engine.py`

**Purpose:** Google Earth Engine client for AlphaEarth embeddings and data.

### Key Classes

#### `EarthEngineClient`
- Fetches AlphaEarth embeddings from Google Earth Engine
- Gets deforestation labels from Global Forest Watch
- Caches API responses for faster repeated access

**Example:**
```python
from src.utils.earth_engine import EarthEngineClient

client = EarthEngineClient(use_cache=True)

# Get embedding for a location and date
embedding = client.get_embedding(
    lat=-3.5,
    lon=-62.5,
    date="2023-01-01"
)
# Returns: np.ndarray of shape (64,)

# Get time series
dates, embeddings = client.get_embedding_timeseries(
    lat=-3.5,
    lon=-62.5,
    start_date="2022-01-01",
    end_date="2023-01-01",
    interval_days=30
)
# Returns: (list of dates, np.ndarray of shape (n_timesteps, 64))

# Get deforestation labels
bounds = {
    "min_lat": -4.0,
    "max_lat": -3.0,
    "min_lon": -63.0,
    "max_lon": -62.0
}
events = client.get_deforestation_labels(bounds, year=2023)

# Get stable forest locations
stable_locs = client.get_stable_forest_locations(bounds, n_samples=100)
```

## Module: `src/utils/geo.py`

**Purpose:** Geospatial utilities for distance calculations and coordinate handling.

### Key Functions

**`haversine_distance(lat1, lon1, lat2, lon2)`**
- Calculates great circle distance between two points
- Returns distance in meters

**`distance_matrix(locations1, locations2=None)`**
- Computes pairwise distances between all locations
- Returns matrix of distances in meters

**`get_neighbors(center_lat, center_lon, distance_m, n_neighbors)`**
- Returns N neighboring locations around a center point
- Useful for spatial features

**`buffer_zone(locations, buffer_distance_m)`**
- Filters locations to maintain minimum spacing
- Critical for spatial cross-validation

**`geographic_bounds(locations, padding_degrees)`**
- Returns bounding box for a list of locations

**Example:**
```python
from src.utils.geo import haversine_distance, get_neighbors, buffer_zone

# Calculate distance
dist = haversine_distance(-3.5, -62.5, -3.6, -62.6)  # meters

# Get 8 neighbors at 1km distance
neighbors = get_neighbors(-3.5, -62.5, distance_m=1000, n_neighbors=8)

# Remove nearby locations (for spatial CV)
locations = [(-3.5, -62.5), (-3.5001, -62.5001), (-4.0, -63.0)]
filtered = buffer_zone(locations, buffer_distance_m=10000)  # 10km buffer
```

### Key Classes

#### `Location`
Wrapper for geographic locations with useful methods.

```python
from src.utils.geo import Location

loc1 = Location(lat=-3.5, lon=-62.5, metadata={"name": "Site A"})
loc2 = Location(lat=-3.6, lon=-62.6)

distance = loc1.distance_to(loc2)  # meters
coords = loc1.to_tuple()  # (-3.5, -62.5)
data = loc1.to_dict()  # {"lat": -3.5, "lon": -62.5, "name": "Site A"}
```

## Module: `src/utils/visualization.py`

**Purpose:** Visualization utilities for plots and maps.

### Key Functions

**`plot_pca_separation(embeddings_cleared, embeddings_intact, accuracy)`**
- Plots PCA visualization for CRAWL Test 1
- Shows separation between cleared and intact forest embeddings

**`plot_temporal_signal(times, distances, p_value)`**
- Plots temporal signal for CRAWL Test 2
- Shows embedding changes before clearing

**`plot_regional_generalization(region_results, cv_threshold)`**
- Plots regional consistency for CRAWL Test 3
- Shows signal strength across different regions

**`plot_minimal_model_results(X, y, feature_names, auc)`**
- Plots minimal model results for CRAWL Test 4
- 2D scatter of features with decision boundary

**`create_decision_gate_summary(test_results)`**
- Creates summary visualization for all CRAWL tests
- Shows GO/NO-GO decision

**`save_figure(fig, filename, subdir, dpi)`**
- Saves figures to results/figures/ directory
- Automatically organizes by phase (crawl/walk/run)

**Example:**
```python
from src.utils.visualization import plot_pca_separation, save_figure
import matplotlib.pyplot as plt

# Create plot
fig = plot_pca_separation(
    embeddings_cleared=cleared_embeddings,
    embeddings_intact=intact_embeddings,
    accuracy=0.89
)

# Save automatically to results/figures/crawl/
save_figure(fig, "test_1_separability.png", subdir="crawl")
```

## Usage in CRAWL Tests

All CRAWL tests will use these utilities:

```python
# Example CRAWL Test 1 structure
from src.utils import EarthEngineClient, plot_pca_separation, save_figure, get_config

config = get_config()
client = EarthEngineClient()

# 1. Get data using earth_engine
cleared_locs = client.get_deforestation_labels(...)
intact_locs = client.get_stable_forest_locations(...)

embeddings_cleared = [client.get_embedding(loc['lat'], loc['lon'], date)
                      for loc in cleared_locs]

# 2. Analyze (using sklearn, numpy, etc.)
accuracy = test_separability(embeddings_cleared, embeddings_intact)

# 3. Visualize
fig = plot_pca_separation(embeddings_cleared, embeddings_intact, accuracy)

# 4. Save
save_figure(fig, "test_1_separability.png", subdir="crawl")

# 5. Decision gate
threshold = config.crawl_thresholds["test_1_separability"]["min_accuracy"]
passed = accuracy >= threshold

print(f"✓ PASS" if passed else "✗ FAIL")
```

## Next Steps

With utilities complete, we can now build the CRAWL tests:
1. `src/crawl/test_1_separability.py`
2. `src/crawl/test_2_temporal.py`
3. `src/crawl/test_3_generalization.py`
4. `src/crawl/test_4_minimal_model.py`

Each test will follow the pattern:
1. Load config
2. Get data from Earth Engine
3. Extract features (using geo utils)
4. Run analysis
5. Visualize results
6. Make go/no-go decision
