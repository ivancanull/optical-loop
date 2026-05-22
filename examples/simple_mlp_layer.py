"""Read a cached Timeloop result for one AlexNet layer."""

import importlib.util
import sys
from pathlib import Path


def _bootstrap_local_package() -> None:
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


_bootstrap_local_package()

from opticalloop import LayerSimulator, MRRMacroConfig, TimeloopLayerRef, TimeloopResultCache


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    layer = TimeloopLayerRef(network="alexnet", layer_path="alexnet/0")
    architecture = MRRMacroConfig(n_tiles=1, n_pes=1, n_cols=100, n_rows=12)
    cache = TimeloopResultCache(results_dir=repo_root / "results")

    result = LayerSimulator(layer=layer, architecture=architecture, cache=cache).run()

    print("OpticalLoop Timeloop-backed AlexNet layer")
    print(f"Source:      {result.source}")
    print(f"Layer:       {layer.layer_path}")
    print(f"Architecture:{architecture.architecture_key}")
    print(f"Cycles:      {result.cycles}")
    print(f"Latency:     {result.latency_s:.6e} s")
    print(f"Energy:      {result.energy_j:.6e} J")
    print(f"Area:        {result.area_mm2:.6e} mm^2")
    print(f"TOPS/W:      {result.tops_per_w:.6f}")
    print("Component energy from Timeloop:")
    for component, energy_j in sorted(result.energy_breakdown.items()):
        print(f"  {component:28s} {energy_j:.6e} J")


if __name__ == "__main__":
    main()
