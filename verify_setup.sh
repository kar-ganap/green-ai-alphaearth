#!/bin/bash
# Verification script to test utility modules

set -e  # Exit on error

echo "================================"
echo "Verifying Project Setup"
echo "================================"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Run: source .venv/bin/activate"
    echo ""
fi

# Check Python version
echo "1. Checking Python version..."
python_version=$(python --version 2>&1)
echo "   ✓ $python_version"
echo ""

# Check if dependencies are installed
echo "2. Checking dependencies..."
if python -c "import pytest" 2>/dev/null; then
    echo "   ✓ pytest installed"
else
    echo "   ✗ pytest not found. Run: uv pip install -e \".[dev]\""
    exit 1
fi

if python -c "import numpy" 2>/dev/null; then
    echo "   ✓ numpy installed"
else
    echo "   ✗ numpy not found. Run: uv pip install -e ."
    exit 1
fi
echo ""

# Check project structure
echo "3. Checking project structure..."
required_dirs=("src" "tests" "data" "results" "docs")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir/ exists"
    else
        echo "   ✗ $dir/ missing"
    fi
done
echo ""

# Check utility modules
echo "4. Checking utility modules..."
if [ -f "src/utils/config.py" ]; then
    echo "   ✓ src/utils/config.py"
fi
if [ -f "src/utils/earth_engine.py" ]; then
    echo "   ✓ src/utils/earth_engine.py"
fi
if [ -f "src/utils/geo.py" ]; then
    echo "   ✓ src/utils/geo.py"
fi
if [ -f "src/utils/visualization.py" ]; then
    echo "   ✓ src/utils/visualization.py"
fi
echo ""

# Check test files
echo "5. Checking test files..."
if [ -f "tests/test_config.py" ]; then
    echo "   ✓ tests/test_config.py"
fi
if [ -f "tests/test_geo.py" ]; then
    echo "   ✓ tests/test_geo.py"
fi
if [ -f "tests/test_visualization.py" ]; then
    echo "   ✓ tests/test_visualization.py"
fi
echo ""

# Run tests
echo "6. Running unit tests..."
echo "   (This may take a minute...)"
echo ""

if pytest tests/ -v --tb=short -m "not integration"; then
    echo ""
    echo "   ✓ All unit tests passed!"
    echo ""
else
    echo ""
    echo "   ✗ Some tests failed. Check output above."
    exit 1
fi

# Summary
echo "================================"
echo "✓ Setup Verification Complete"
echo "================================"
echo ""
echo "Ready to proceed with CRAWL tests!"
echo ""
echo "Next steps:"
echo "  1. Authenticate with Earth Engine: earthengine authenticate"
echo "  2. Start CRAWL Test 1: python src/crawl/test_1_separability.py"
echo ""
