# Results Directory

This directory stores model outputs, figures, and experiment results.

## Structure

```
results/
├── models/              # Trained models
│   ├── production_model.pkl
│   ├── baseline_models/
│   └── checkpoints/
├── figures/             # Plots and visualizations
│   ├── crawl/          # CRAWL test results
│   ├── walk/           # WALK validation plots
│   └── run/            # Production figures
├── predictions/         # Model predictions
│   ├── alerts.json
│   └── historical/
└── metrics/             # Performance metrics
    ├── baselines.json
    ├── validation_results.json
    └── error_analysis.json
```

## Files Generated

### CRAWL Phase
- `crawl_test_1_separability.png`
- `crawl_test_2_temporal.png`
- `crawl_test_3_generalization.png`
- `crawl_test_4_minimal.png`

### WALK Phase
- `walk_spatial_splits.png`
- `baseline_comparison.json`
- `feature_ablation.json`
- `validation_protocol.json`

### RUN Phase
- `production_model.pkl`
- `error_analysis.json`
- `feature_importance.png`

Results files are gitignored - generated during model training.
