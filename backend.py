"""Timeloop backend adapter for OpticalLoop."""

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from opticalloop.config.workload import TimeloopLayerRef
from opticalloop.result import SimulationResult


class TimeloopArchitectureConfig(Protocol):
    macro: str
    system: str
    max_utilization: bool
    architecture_key: str

    def to_timeloop_variables(self) -> dict:
        ...


@dataclass(frozen=True)
class TimeloopRun:
    """One Timeloop mapper request owned by the backend adapter."""

    layer: TimeloopLayerRef
    architecture: TimeloopArchitectureConfig
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeloopBackend:
    """Run Timeloop and convert mapper stats to `SimulationResult`."""

    scripts_dir: Optional[Path] = None
    quick_run: Optional[Callable] = None

    def run_layer(
        self, layer: TimeloopLayerRef, architecture: TimeloopArchitectureConfig
    ) -> SimulationResult:
        quick_run = self.quick_run or self._load_quick_run()
        run = TimeloopRun(layer=layer, architecture=architecture)
        stats = quick_run(**self._quick_run_kwargs(run))
        return self._result_from_stats(stats, run)

    def run_batch(
        self, runs: Sequence[TimeloopRun], n_jobs: Optional[int] = None
    ) -> List[SimulationResult]:
        """Run a batch through the vendored Timeloop helper layer.

        `quick_run` injection is intentionally executed sequentially so unit
        tests can assert exact call order without joblib process boundaries.
        Production live runs use `utils.parallel_test(...)`.
        """

        runs = list(runs)
        if not runs:
            return []

        if self.quick_run is not None:
            stats_list = [self.quick_run(**self._quick_run_kwargs(run)) for run in runs]
        else:
            utils = self._load_utils_module()
            delayed_calls = [
                utils.delayed(utils.quick_run)(**self._quick_run_kwargs(run))
                for run in runs
            ]
            if n_jobs is None:
                stats_list = utils.parallel_test(delayed_calls)
            else:
                stats_list = utils.parallel_test(delayed_calls, n_jobs=n_jobs)

        return [
            self._result_from_stats(stats, run)
            for stats, run in zip(list(stats_list), runs)
        ]

    @staticmethod
    def _quick_run_kwargs(run: TimeloopRun) -> Dict[str, object]:
        architecture = run.architecture
        return {
            "macro": architecture.macro,
            "layer": run.layer.layer_path,
            "variables": architecture.to_timeloop_variables(),
            "system": architecture.system,
            "max_utilization": architecture.max_utilization,
        }

    @staticmethod
    def _result_from_stats(stats, run: TimeloopRun) -> SimulationResult:
        architecture = run.architecture
        metadata = {
            "network": run.layer.network,
            "layer_path": run.layer.layer_path,
            "architecture": architecture.architecture_key,
            "macro": architecture.macro,
            "system": architecture.system,
        }
        metadata.update(dict(run.metadata))
        return SimulationResult.from_timeloop_stats(
            stats,
            source="timeloop-live",
            metadata=metadata,
        )

    def _load_quick_run(self) -> Callable:
        return self._load_utils_module().quick_run

    def _load_utils_module(self):
        self._prepare_timeloop_shared_libraries()
        scripts_dir = self.scripts_dir or self._default_scripts_dir()
        scripts_dir = Path(scripts_dir)
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import utils

        return utils

    @staticmethod
    def _prepare_timeloop_shared_libraries() -> None:
        mapper = shutil.which("timeloop-mapper")
        if mapper is None:
            return
        lib_dir = Path(mapper).resolve().parents[1] / "lib"
        if not (lib_dir / "libtimeloop-mapper.so").exists():
            return
        current = os.environ.get("LD_LIBRARY_PATH", "")
        paths = [path for path in current.split(os.pathsep) if path]
        lib_dir_text = str(lib_dir)
        if lib_dir_text not in paths:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([lib_dir_text, *paths])

    @staticmethod
    def _default_scripts_dir() -> Path:
        package_scripts = Path(__file__).resolve().parent / "workspace" / "scripts"
        if package_scripts.exists():
            return package_scripts
        return Path(__file__).resolve().parents[1] / "workspace" / "scripts"
