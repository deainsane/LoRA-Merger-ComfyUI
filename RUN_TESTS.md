# Running Tests

Quick guide to running tests for the LoRA-Merger-ComfyUI custom nodes.

## Integration Test

### Quick Start

```bash
cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 test_gradient_analyzer_integration.py
```

### What It Tests

The integration test validates the **PM LoRA Semantic Analyzer (Gradient)** pipeline:

1. ✅ Dependencies (PyTorch, diffusers, safetensors)
2. ✅ Node instantiation (all semantic merge nodes)
3. ✅ INPUT_TYPES configuration
4. ✅ Checkpoint dropdown population
5. ✅ Specification parsing

### Expected Output

```
================================================================================
PM LoRA Semantic Analyzer (Gradient) - Integration Test
================================================================================

[... file checks ...]

================================================================================
TEST SUMMARY
================================================================================
✓ PASS   Dependencies
✓ PASS   LoRA Power Stacker
✓ PASS   Semantic Analyzer Instantiation
✓ PASS   Semantic Merge Spec
✓ PASS   Semantic Merger Instantiation

Results: 5/5 tests passed

🎉 ALL TESTS PASSED!
```

### Exit Codes

- `0` - All tests passed
- `1` - One or more tests failed
- `130` - Test interrupted (Ctrl+C)

## Unit Tests

### Run All Unit Tests

```bash
cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -m pytest tests/
```

### Run Specific Test Files

```bash
# Test types system
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -m pytest tests/test_types.py -v

# Test decomposition
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -m pytest tests/test_decomposition.py -v

# Test validation
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -m pytest tests/test_validation.py -v

# Test algorithms
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -m pytest tests/test_algorithms.py -v
```

### Run with Coverage

```bash
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -m pytest --cov=src tests/
```

## Syntax Validation

### Check Python Syntax

```bash
# Single file
python3 -m py_compile src/nodes_semantic_merge.py

# All Python files
find . -name "*.py" -not -path "./venv/*" -exec python3 -m py_compile {} \;
```

### Check with Flake8

```bash
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -m flake8 src/ --max-line-length=120
```

## Test Files

### Integration Tests
- `test_gradient_analyzer_integration.py` - Validates semantic analyzer pipeline

### Unit Tests (in `tests/` directory)
- `test_types.py` - Type system and validators
- `test_algorithms.py` - Merge algorithms
- `test_decomposition.py` - SVD/QR decomposition
- `test_validation.py` - Input validation

### Test Workflows
- `/home/lars/SD/Apps/ComfyUI/user/default/workflows/PM-Gradient-Analyzer-zImage.json` - Example workflow

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run integration tests
        run: python3 test_gradient_analyzer_integration.py

      - name: Run unit tests
        run: pytest tests/ -v --cov=src
```

## Test Requirements

### For Integration Tests

**Required:**
- ComfyUI installed at `/home/lars/SD/Apps/ComfyUI`
- venv at `/home/lars/SD/Apps/ComfyUI/venv`
- Dependencies: `torch`, `diffusers>=0.30.0`, `safetensors`

**Optional (for file checks):**
- zImage checkpoint in `models/diffusion_models/`
- LoRAs in `models/loras/`

### For Unit Tests

**Required:**
- PyTorch
- Standard Python libraries

**Optional:**
- pytest
- pytest-cov (for coverage reports)

## Quick Commands Reference

```bash
# Integration test (basic validation)
./test_gradient_analyzer_integration.py

# Unit tests (comprehensive)
pytest tests/ -v

# Syntax check
python3 -m py_compile src/*.py

# Coverage report
pytest --cov=src --cov-report=html tests/

# Specific test
pytest tests/test_types.py::test_validate_lora_tensors -v
```

## Debugging Test Failures

### Enable Verbose Output

```bash
# Integration test
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 test_gradient_analyzer_integration.py 2>&1 | tee test_output.log

# Unit tests
pytest tests/ -vv --tb=long
```

### Check Imports

```bash
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -c "
import sys
sys.path.insert(0, '/home/lars/SD/Apps/ComfyUI')
sys.path.insert(0, '/home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI')
from src.nodes_semantic_merge import PMLoRASemanticAnalyzerGradient
print('✓ Import successful')
"
```

### Test Individual Components

```bash
# Test diffusers
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -c "
from diffusers.models import ZImageTransformer2DModel
print('✓ ZImageTransformer2DModel available')
"

# Test folder_paths
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -c "
import sys
sys.path.insert(0, '/home/lars/SD/Apps/ComfyUI')
import folder_paths
print('✓ folder_paths available')
"
```

## Performance Testing

### Time Individual Tests

```bash
time /home/lars/SD/Apps/ComfyUI/venv/bin/python3 test_gradient_analyzer_integration.py
```

### Profile Tests

```bash
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -m cProfile -o test_profile.prof test_gradient_analyzer_integration.py

# View profile
/home/lars/SD/Apps/ComfyUI/venv/bin/python3 -m pstats test_profile.prof
```

## Documentation

- **TEST_GRADIENT_ANALYZER.md** - Detailed test documentation
- **QUICK_START_GRADIENT_ANALYZER.md** - User guide
- **ZIMAGE_LOADER_OPTIMIZATION.md** - Technical details
- **BUGFIX_ZIMAGE_IMPORT.md** - Import fix documentation

## Support

For issues or questions:
1. Check test output for error messages
2. Review documentation files
3. Verify all dependencies are installed
4. Check that files exist in expected locations
5. Open an issue on GitHub if problems persist
