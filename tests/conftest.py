"""Pytest configuration for LoRA Power-Merger.

Sets up the test environment so the whole suite runs on torch, mergekit and
pytest alone -- no ComfyUI installation required.

Deliberately lives in tests/ rather than at the project root: the root holds an
``__init__.py`` (the ComfyUI node entry point, which imports ComfyUI), so pytest
would import a root-level conftest as a submodule of that package and execute the
entry point before any test ran. tests/ is not a package, so this file imports on
its own.
"""

import os
import sys
import types
from unittest.mock import MagicMock

# Put the PROJECT ROOT (not src/) on the path, so modules under test are imported
# as `src.<module>`. Modules in src/ use package-relative imports (`from ..types
# import ...`); importing them as top-level modules makes those resolve beyond
# the top-level package and fail. src/__init__.py is empty, so this pulls in no
# ComfyUI dependencies by itself.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _stub_root_package():
    """Stop pytest from executing the project root's ``__init__.py``.

    Because the root holds an ``__init__.py``, pytest treats the project root as a
    package and imports it in ``Package.setup()`` before running any test under
    it. That file is the ComfyUI node entry point (``from .src.lora_apply import
    ...``), so importing it outside a ComfyUI process fails on the relative import
    -- and it is not test code in any case.

    ``Package.setup()`` goes through ``import_path``, which under the ``importlib``
    import mode returns an already-present ``sys.modules`` entry before touching
    the file. The module name it derives for ``<rootdir>/__init__.py`` is
    ``"__init__"``; seeding that name with an empty module makes the import a
    no-op. Registered here rather than via ``pytest_collect_directory``: that hook
    fires only while recursing into directories, not for the parent packages of a
    test path named directly on the command line.
    """
    sys.modules.setdefault("__init__", types.ModuleType("__init__"))


def pytest_ignore_collect(collection_path, config):
    """Never collect package __init__.py files or anything under src/."""
    path_str = str(collection_path)
    if path_str.endswith('__init__.py'):
        return True
    if (os.sep + 'src' + os.sep) in path_str or path_str.endswith(os.sep + 'src'):
        return True
    return False


def _mock_comfyui_modules():
    """Install stand-ins for the ComfyUI modules src/ imports at module scope."""
    # Every comfy submodule imported anywhere under src/ needs an entry: `comfy`
    # is a MagicMock rather than a real package, so `import comfy.<x>` only
    # resolves via sys.modules. Keep in sync with:
    #   grep -rhoE "(from|import) comfy(\.[a-z_]+)*" src/ --include=*.py | sort -u
    comfy_mock = MagicMock()
    sys.modules['comfy'] = comfy_mock
    for name in ('utils', 'model_management', 'lora', 'weight_adapter',
                 'model_patcher', 'sd', 'sample'):
        sys.modules[f'comfy.{name}'] = MagicMock()

    class LoRAAdapterMock:
        """Stand-in for comfy.weight_adapter.LoRAAdapter.

        Mirrors the real signature: it is a plain container over ``loaded_keys``
        and a ``weights`` 6-tuple (up, down, alpha, mid, dora_scale, reshape),
        which the merger reads back after building its output state dict.
        """

        name = "lora"

        def __init__(self, loaded_keys=None, weights=None):
            self.loaded_keys = set() if loaded_keys is None else loaded_keys
            self.weights = weights

    comfy_mock.weight_adapter.LoRAAdapter = LoRAAdapterMock
    sys.modules['comfy.weight_adapter'].LoRAAdapter = LoRAAdapterMock

    # The merger logs free VRAM around the offload step, formatting the result
    # with `:.0f` -- a bare MagicMock would raise on __format__.
    mm = sys.modules['comfy.model_management']
    mm.get_free_memory.return_value = 8 * 1024 ** 3
    comfy_mock.model_management = mm
    comfy_mock.utils = sys.modules['comfy.utils']

    sys.modules['comfy_extras'] = MagicMock()
    sys.modules['comfy_extras.nodes_custom_sampler'] = MagicMock()
    sys.modules['nodes'] = MagicMock()
    sys.modules['latent_preview'] = MagicMock()

    folder_paths_mock = MagicMock()
    folder_paths_mock.get_folder_paths.return_value = []
    folder_paths_mock.folder_names_and_paths = {}
    sys.modules['folder_paths'] = folder_paths_mock


def pytest_configure(config):
    """Mock ComfyUI before test modules (and the src/ modules they import) load."""
    _mock_comfyui_modules()
    _stub_root_package()


# Also applied at import time: conftest is imported before collection, so this
# covers anything that resolves earlier than pytest_configure.
_mock_comfyui_modules()
_stub_root_package()
