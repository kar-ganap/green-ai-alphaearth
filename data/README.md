# Data Directory

This directory stores downloaded data and processed datasets.

## Structure

```
data/
├── raw/                    # Raw data from Google Earth Engine
│   ├── embeddings/        # AlphaEarth embeddings
│   ├── labels/            # Deforestation labels (GFW, GLAD)
│   └── context/           # Roads, protected areas, DEM
├── processed/             # Processed training data
│   ├── train_locations.pkl
│   ├── test_locations.pkl
│   └── features/
└── cache/                 # Cached API responses
```

## Data Sources

### AlphaEarth Embeddings
- **Collection**: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- **Resolution**: 10m × 10m
- **Dimensions**: 64
- **Coverage**: 2017-2024

### Deforestation Labels
- **Primary**: Global Forest Watch (Hansen et al.)
- **Secondary**: GLAD alerts, PRODES (Brazil)
- **Resolution**: 30m

### Context Data
- **Roads**: OpenStreetMap
- **Protected areas**: World Database on Protected Areas
- **Terrain**: Copernicus DEM GLO-30

## Download Data

Run the data download script:

```bash
python src/utils/download_data.py --region amazon --years 2020 2021 2022 2023
```

## Data Files

Data files are gitignored due to size. Download locally before running models.
