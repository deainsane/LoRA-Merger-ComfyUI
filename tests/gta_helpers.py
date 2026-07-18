"""Shared helpers for standalone gta.py script tests.

Not a `test_` module (pytest is broken in this repo anyway). Loads
src/merge/gta.py by file path so the broken package __init__ is never imported,
and re-exports mergekit's sparsify functions as the parity oracle.
"""
import importlib.util
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gta():
    path = os.path.join(REPO, "src", "merge", "gta.py")
    spec = importlib.util.spec_from_file_location("gta_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gta = load_gta()


def run(tests):
    """Run a dict/list of {name: fn}. Exit non-zero on first failure."""
    if isinstance(tests, dict):
        tests = list(tests.items())
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    if failed:
        print(f"\n{failed} FAILED")
        sys.exit(1)
    print(f"\nAll {len(tests)} passed")