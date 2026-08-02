# Running Tests

The test suite runs **without a ComfyUI installation**. ComfyUI's modules
(`comfy`, `folder_paths`, `comfy_extras`, `nodes`) are mocked in
[tests/conftest.py](tests/conftest.py), so the only requirements are torch,
mergekit and pytest.

## Quick Start

```bash
cd /home/lars/SD/Apps/ComfyUI/custom_nodes/LoRA-Merger-ComfyUI
/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest tests/
```

Expected: **238 passed**.

Any Python with the dependencies below works; `/home/lars/SD/Apps/ComfyUI/.venv`
is simply the environment that already has them on this machine.

### Run a single file or test

```bash
PY=/home/lars/SD/Apps/ComfyUI/.venv/bin/python

$PY -m pytest tests/test_validation.py            # one file
$PY -m pytest tests/test_gta_sparsify.py -v       # verbose
$PY -m pytest tests/test_types.py::TestTypeGuards # one class
$PY -m pytest tests/ -k "sparsify or interp"      # by name
```

### Coverage

```bash
/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m pytest --cov=src tests/
```

Requires `pytest-cov` (see `requirements-dev.txt`).

## Two Kinds of Test File

Both kinds are collected by `pytest tests/`; the difference only matters when
you want to run one directly.

**Script-style** — plain asserts driven by a `run([...])` list, guarded behind
`if __name__ == "__main__":`. These also run as standalone scripts with no
pytest at all, which is useful for a quick check or when bisecting:

```bash
/home/lars/SD/Apps/ComfyUI/.venv/bin/python tests/test_gta_sparsify.py
# -> PASS magnitude ... All 12 passed
```

Files: `test_gta_behavior`, `test_gta_merge`, `test_gta_parity`,
`test_gta_sparsify`, `test_interp_delta_merge`, `test_interp_fidelity`,
`test_interp_integration`, `test_lora_save`, `test_merge_node_names`,
`test_merger_vram_offload`.

**Pytest-style** — `Test*` classes, fixtures, `pytest.raises`. Run these through
pytest only; executing them directly does not set up the import paths.

Files: `test_algorithms`, `test_blocks`, `test_decomposition`,
`test_spectral_norm`, `test_types`, `test_utility`, `test_validation`.

## How Imports Are Set Up

Two rules keep the suite importable outside ComfyUI. Both are handled by
[tests/conftest.py](tests/conftest.py); they matter when you add a test.

1. **Import project code as `src.<module>`**, never as a bare top-level module:

   ```python
   from src.validation import LoRAStackValidator   # correct
   from validation import LoRAStackValidator       # breaks
   from types import is_lora_tensors               # breaks: stdlib `types`
   ```

   Modules under `src/` use package-relative imports (`from ..types import ...`),
   which only resolve when `src` itself is the package. conftest puts the project
   root on `sys.path` for this. The same applies to `unittest.mock.patch`
   targets: `@patch('src.merge.algorithms.LinearMergeTask')`.

2. **Do not add `tests/__init__.py`.** With it, pytest walks up to the project
   root's `__init__.py` — the ComfyUI node entry point — and imports ComfyUI
   before any test runs.

`conftest.py` lives in `tests/`, not the project root, for the same reason: a
root-level conftest is imported as part of that package.

## Test Files

| File | Covers |
|------|--------|
| `test_algorithms.py` | Merge algorithm registry and dispatch |
| `test_blocks.py` | Block selection, key normalization, block weights |
| `test_decomposition.py` | SVD / randomized SVD / energy-based decomposers |
| `test_gta_behavior.py` | GTA delta-space merge behavior |
| `test_gta_merge.py` | Sign election and disjoint merge |
| `test_gta_parity.py` | Parity of local sparsify vs. mergekit |
| `test_gta_sparsify.py` | magnitude / outliers / bernoulli / della sparsify |
| `test_interp_delta_merge.py` | slerp / nuslerp / karcher / nearswap in delta space |
| `test_interp_fidelity.py` | Delta-space blend vs. reference average |
| `test_interp_integration.py` | End-to-end merge through the node |
| `test_lora_save.py` | Tensor sanitization before safetensors save |
| `test_merge_node_names.py` | Node widget names and titles |
| `test_merger_vram_offload.py` | `offload_models` widget behavior |
| `test_spectral_norm.py` | Spectral norm scaling |
| `test_types.py` | Type guards and validators |
| `test_utility.py` | SVD/QR pipeline helpers |
| `test_validation.py` | LoRA stack, shape and parameter validation |

## Requirements

```bash
pip install -r requirements-dev.txt
```

Needed to run the suite: `torch`, `mergekit`, `pytest`. Optional:
`pytest-cov` (coverage), `pytest-xdist` (`-n auto` parallel runs).

A ComfyUI installation is **not** required.

## Syntax and Lint

```bash
find src -name "*.py" -exec python3 -m py_compile {} \;
/home/lars/SD/Apps/ComfyUI/.venv/bin/python -m flake8 src/ --max-line-length=120
```

## Debugging Failures

```bash
PY=/home/lars/SD/Apps/ComfyUI/.venv/bin/python

$PY -m pytest tests/ -vv --tb=long   # full tracebacks
$PY -m pytest tests/ -x              # stop at first failure
$PY -m pytest tests/ --lf            # rerun last failures
```

If a new test fails at import with `attempted relative import beyond top-level
package`, `No module named 'src'`, or picks up the stdlib `types` module, it is
importing project code as a top-level module — see "How Imports Are Set Up".

## Continuous Integration

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest tests/ -v
```

No ComfyUI checkout is needed in CI.
