# Unit Tests

Comprehensive unit tests for the deforestation early warning system utilities.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_config.py
pytest tests/test_geo.py
pytest tests/test_visualization.py
```

### Run tests by marker
```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

### Verbose output
```bash
pytest -v
```

## Test Structure

```
tests/
├── __init__.py
├── test_config.py          # Config management tests
├── test_geo.py             # Geospatial utilities tests
├── test_visualization.py   # Plotting utilities tests
└── README.md               # This file
```

## Test Coverage

Tests cover:

### `test_config.py`
- ✅ Config file loading
- ✅ Nested value access with dot notation
- ✅ Default values
- ✅ Path resolution
- ✅ Property accessors (alphaearth_collection, embedding_dimensions, etc.)
- ✅ Error handling for missing files

### `test_geo.py`
- ✅ Haversine distance calculations
- ✅ Distance matrix computation
- ✅ Neighbor location generation (4 and 8 neighbors)
- ✅ Buffer zone filtering (spatial cross-validation)
- ✅ Geographic bounds calculation
- ✅ Point in bounds checking
- ✅ Grid sampling
- ✅ Location class methods

### `test_visualization.py`
- ✅ PCA separation plots
- ✅ Temporal signal plots
- ✅ Regional generalization plots
- ✅ Minimal model results plots
- ✅ ROC curves
- ✅ Confusion matrices
- ✅ Plot saving to files

## Integration Tests

Note: Earth Engine client tests require actual Google Earth Engine authentication and are
marked as integration tests. Run them separately:

```bash
pytest -m integration
```

These tests are not run by default in CI/CD pipelines.

## Writing New Tests

### Test Structure
```python
import pytest
from src.utils import function_to_test


class TestFeatureName:
    """Test specific feature."""

    def test_basic_functionality(self):
        """Test basic case."""
        result = function_to_test(input)
        assert result == expected

    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Best Practices
1. **One test, one assertion** (when possible)
2. **Use descriptive test names** (test_what_when_then)
3. **Use fixtures** for repeated setup
4. **Mock external dependencies** (Earth Engine API, etc.)
5. **Test both success and failure cases**

## Continuous Integration

Tests run automatically on:
- Every commit (pre-commit hook)
- Every pull request
- Before deployment

Minimum coverage requirement: 80%
