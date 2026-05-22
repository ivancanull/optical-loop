"""Test bootstrap for the local `opticalloop` package.

The repository directory is named `optical-loop`, so tests load the package
the same way the CLI does instead of requiring an editable install.
"""

import importlib.util
import sys
from pathlib import Path


def pytest_configure() -> None:
    package_root = Path(__file__).resolve().parents[1]
    if "opticalloop" in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        "opticalloop",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load opticalloop package from {package_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["opticalloop"] = module
    spec.loader.exec_module(module)
