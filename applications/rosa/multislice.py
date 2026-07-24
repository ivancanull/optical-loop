"""MB-OSA experiment analysis and Adaptive Slice-Width Mapping (ASWM)."""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
import numpy as np

from opticalloop.applications.rosa.reproduction import ExperimentManifest


ACCURACY_STATUS = "NOT_MODELED"
ASWM_SLICE_BITS = (1, 2, 4)
ENERGY_MODELS = (
    "optimistic_constant_symbol",
    "linear_bit",
    "conservative_walden",
)
PARETO_ARTIFACT_MAX_POINTS = 1_000


@dataclass(frozen=True)
class SliceChoice:
    layer: str
    slice_bits: int
    energy_j: float
    latency_s: float
    cycles: int | None = None
    stationarity: str = ""

    @property
    def choice_key(self) -> tuple[str, int]:
        return self.stationarity, self.slice_bits


@dataclass(frozen=True)
class ParetoState:
    energy_j: float
    latency_s: float
    choices: tuple[SliceChoice, ...]
    cycles: int | None = None

    @property
    def edp_j_s(self) -> float:
        return self.energy_j * self.latency_s


@dataclass(frozen=True)
class ParetoFrontier(Sequence[ParetoState]):
    """Compact, lazy view of an exact final Pareto frontier."""

    energies: np.ndarray
    latencies: np.ndarray
    cycles: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.energies)

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return tuple(self[position] for position in indices)
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        return ParetoState(
            float(self.energies[position]),
            float(self.latencies[position]),
            (),
            int(self.cycles[position]) if self.cycles is not None else None,
        )


class ASWMOptimizer:
    """Exact cumulative energy/latency Pareto optimization for one fixed core."""

    def __init__(self, frontier_limit: int = 100_000) -> None:
        if frontier_limit <= 0:
            raise ValueError("frontier_limit must be positive")
        self.frontier_limit = frontier_limit

    def optimize(
        self,
        choices_by_layer: Mapping[str, Sequence[SliceChoice]],
        expected_choices: set[tuple[str, int]] | None = None,
    ) -> tuple[ParetoState, Sequence[ParetoState]]:
        layers = sorted(choices_by_layer)
        ordered_choices = []
        for layer in layers:
            choices = tuple(sorted(choices_by_layer[layer], key=lambda choice: choice.slice_bits))
            actual = {choice.choice_key for choice in choices}
            expected = expected_choices or {
                (choices[0].stationarity, bits) for bits in ASWM_SLICE_BITS
            }
            if actual != expected:
                raise ValueError(
                    f"Layer {layer} choices {sorted(actual)} do not match {sorted(expected)}"
                )
            ordered_choices.append(choices)
        use_cycles = all(
            choice.cycles is not None
            for choices in ordered_choices
            for choice in choices
        )
        energies = np.array([0.0], dtype=np.float64)
        latencies = np.array([0.0], dtype=np.float64)
        dominance = np.array([0], dtype=np.int64) if use_cycles else latencies.copy()
        history: list[tuple[np.ndarray, np.ndarray]] = []
        for layer, choices in zip(layers, ordered_choices):
            choice_energy = np.array([choice.energy_j for choice in choices])
            choice_latency = np.array([choice.latency_s for choice in choices])
            choice_dominance = np.array(
                [choice.cycles for choice in choices], dtype=np.int64
            ) if use_cycles else choice_latency
            count = len(choices)
            expanded_energy = (energies[:, None] + choice_energy).reshape(-1)
            expanded_latency = (latencies[:, None] + choice_latency).reshape(-1)
            expanded_dominance = (
                dominance[:, None] + choice_dominance
            ).reshape(-1)
            parents = np.repeat(np.arange(len(energies), dtype=np.int64), count)
            choice_indices = np.tile(np.arange(count, dtype=np.uint8), len(energies))
            order = np.lexsort((expanded_dominance, expanded_energy))
            ordered_dominance = expanded_dominance[order]
            keep = np.ones(len(order), dtype=bool)
            if len(order) > 1:
                keep[1:] = ordered_dominance[1:] < np.minimum.accumulate(
                    ordered_dominance[:-1]
                )
            retained = order[keep]
            energies = expanded_energy[retained]
            latencies = expanded_latency[retained]
            dominance = expanded_dominance[retained]
            history.append((parents[retained], choice_indices[retained]))
            if len(energies) > self.frontier_limit:
                raise RuntimeError(
                    f"ASWM Pareto frontier exceeded {self.frontier_limit} states at {layer}"
                )
        best_index = int(np.argmin(energies * latencies))
        selected = []
        cursor = best_index
        for index in range(len(layers) - 1, -1, -1):
            parents, choice_indices = history[index]
            selected.append(ordered_choices[index][int(choice_indices[cursor])])
            cursor = int(parents[cursor])
        selected.reverse()
        best = ParetoState(
            float(energies[best_index]), float(latencies[best_index]), tuple(selected),
            int(dominance[best_index]) if use_cycles else None,
        )
        frontier = ParetoFrontier(
            energies, latencies, dominance if use_cycles else None
        )
        return best, frontier

    @staticmethod
    def prune(states: Iterable[ParetoState]) -> tuple[ParetoState, ...]:
        # Native cycles are exact integers. Using them for dominance avoids
        # treating round-off-different sums of the same latency as distinct
        # Pareto states while preserving the exact mapper objective.
        ordered = sorted(
            states,
            key=lambda state: (
                state.energy_j,
                state.cycles if state.cycles is not None else state.latency_s,
            ),
        )
        kept = []
        best_latency = math.inf
        for state in ordered:
            latency_key = state.cycles if state.cycles is not None else state.latency_s
            if latency_key < best_latency:
                kept.append(state)
                best_latency = latency_key
        return tuple(kept)


class SliceEnergyModel:
    """Apply documented DAC and optical-loss sensitivities to native results."""

    @staticmethod
    def dac_factor(slice_bits: int, model: str) -> float:
        if model == "optimistic_constant_symbol":
            return 1.0
        if model == "linear_bit":
            return float(slice_bits)
        if model == "conservative_walden":
            return float(2**slice_bits - 1)
        raise ValueError(f"Unknown energy model: {model}")

    def adjust(self, row: Mapping[str, object], model: str, loss_db_per_stage: float) -> float:
        slice_bits = int(row["front_mrr_slice_bits"])
        breakdown = _mapping(row.get("energy_breakdown", {}))
        front_dac = "input_dac" if row.get("sliced_operand") == "input" else "weight_dac"
        dac_energy = _component_sum(breakdown, front_dac)
        laser_energy = _component_sum(breakdown, "laser")
        primary_factor = self.dac_factor(slice_bits, "linear_bit")
        requested_factor = self.dac_factor(slice_bits, model)
        energy = float(row["energy_j"]) - dac_energy + dac_energy * requested_factor / primary_factor
        delay_stages = max(int(row.get("temporal_slices", 1)) - 1, 0)
        laser_compensation = 10 ** (float(loss_db_per_stage) * delay_stages / 10.0)
        return energy + laser_energy * (laser_compensation - 1.0)


class MultiSliceAnalyzer:
    """Build fixed-width, ASWM, sensitivity, plot, and validation artifacts."""

    def __init__(self, manifest: ExperimentManifest, run_dir: Path) -> None:
        self.manifest = manifest
        self.run_dir = Path(run_dir).resolve()
        self.artifacts = self.run_dir / "artifacts-multislice"
        self.energy_model = SliceEnergyModel()

    def raw_dataframe(self) -> pd.DataFrame:
        rows = []
        for path in sorted((self.run_dir / "jobs").glob("*.json")):
            payload = json.loads(path.read_text())
            if payload["status"] == "success":
                rows.append({**payload["job"], **payload["metrics"]})
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["accuracy"] = float("nan")
            frame["accuracy_delta"] = float("nan")
            frame["accuracy_constraint"] = False
            frame["accuracy_status"] = ACCURACY_STATUS
        return frame

    def analyze(self, *, execute_notebook: bool = False) -> Mapping[str, Path]:
        self.artifacts.mkdir(parents=True, exist_ok=True)
        raw = self.raw_dataframe()
        raw.to_csv(self.artifacts / "layer_results.csv", index=False)
        modeled = self._modeled_rows(raw)
        modeled.to_csv(self.artifacts / "modeled_layer_results.csv", index=False)
        fixed = self._fixed_summary(modeled)
        fixed.to_csv(self.artifacts / "fixed_width_summary.csv", index=False)
        selections, aswm, frontiers = self._aswm_results(modeled)
        selections.to_csv(self.artifacts / "aswm_layer_selections.csv", index=False)
        aswm.to_csv(self.artifacts / "aswm_summary.csv", index=False)
        frontiers.to_csv(self.artifacts / "aswm_pareto_frontiers.csv", index=False)
        rankings = self._rankings(aswm)
        rankings.to_csv(self.artifacts / "shared_core_rankings.csv", index=False)
        workload_best = (
            aswm.sort_values("edp_j_s")
            .groupby(
                ["network", "mapping", "energy_model", "optical_loss_db_per_stage"],
                as_index=False,
            )
            .first()
        )
        workload_best.to_csv(self.artifacts / "workload_best_cores.csv", index=False)
        comparison = self._scenario_comparison(fixed, aswm)
        comparison.to_csv(self.artifacts / "sensitivity_comparison.csv", index=False)
        checks = MultiSliceValidator(self.manifest, self.run_dir).validate(
            raw, modeled, fixed, selections, aswm
        )
        checks.to_csv(self.artifacts / "validation.csv", index=False)
        report = self._report(checks, fixed, aswm, rankings)
        (self.artifacts / "REPORT.md").write_text(report)
        plot_paths = self._plots(raw, fixed, selections, frontiers)
        outputs = {
            path.name: path for path in self.artifacts.iterdir() if path.is_file()
        }
        outputs.update({path.name: path for path in plot_paths})
        if execute_notebook:
            notebook = self._execute_notebook()
            outputs[notebook.name] = notebook
        return outputs

    @staticmethod
    def _scenario_comparison(fixed: pd.DataFrame, aswm: pd.DataFrame) -> pd.DataFrame:
        fixed_candidates = fixed[
            fixed.front_mrr_slice_bits.isin(ASWM_SLICE_BITS)
            & fixed.accumulation.eq("optical")
        ]
        rows = []
        for adaptive in aswm.itertuples(index=False):
            candidates = fixed_candidates[
                (fixed_candidates.network == adaptive.network)
                & (fixed_candidates.architecture == adaptive.architecture)
                & (fixed_candidates.energy_model == adaptive.energy_model)
                & (fixed_candidates.optical_loss_db_per_stage == adaptive.optical_loss_db_per_stage)
            ]
            if adaptive.mapping in {"ASWM-WS", "ASWM-IS"}:
                candidates = candidates[candidates.stationarity == adaptive.stationarity]
            elif adaptive.mapping == "Mixed-1bit":
                candidates = candidates[candidates.front_mrr_slice_bits == 1]
            best = candidates.sort_values("edp_j_s").iloc[0]
            rows.append({
                **adaptive._asdict(),
                "best_fixed_stationarity": best.stationarity,
                "best_fixed_slice_bits": int(best.front_mrr_slice_bits),
                "fixed_edp_j_s": float(best.edp_j_s),
                "aswm_reduction_vs_best_fixed": 1 - adaptive.edp_j_s / best.edp_j_s,
            })
        return pd.DataFrame(rows)

    def _modeled_rows(self, raw: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in raw.to_dict("records"):
            for model in ENERGY_MODELS:
                for loss_db in self.manifest.raw["optical_loss_db_per_stage"]:
                    energy = self.energy_model.adjust(row, model, float(loss_db))
                    rows.append({
                        **{key: value for key, value in row.items() if key not in {"energy_breakdown", "area_breakdown", "power_breakdown"}},
                        "energy_model": model,
                        "optical_loss_db_per_stage": float(loss_db),
                        "modeled_energy_j": energy,
                        "modeled_edp_j_s": energy * float(row["latency_s"]),
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def _fixed_summary(modeled: pd.DataFrame) -> pd.DataFrame:
        if modeled.empty:
            return pd.DataFrame()
        keys = [
            "network", "architecture", "variant", "stationarity", "accumulation",
            "sliced_operand", "front_mrr_slice_bits", "energy_model",
            "optical_loss_db_per_stage", "accuracy_status",
        ]
        rows = []
        for values, group in modeled.groupby(keys, sort=False, dropna=False):
            energy = group.modeled_energy_j.sum()
            latency = group.latency_s.sum()
            rows.append({**dict(zip(keys, values)), "layers": len(group),
                         "energy_j": energy, "latency_s": latency, "edp_j_s": energy * latency})
        return pd.DataFrame(rows)

    def _aswm_results(self, modeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        candidate_architectures = {
            architecture["name"] for architecture in self.manifest.raw["architectures"]
            if architecture["candidate"]
        }
        osa = modeled[
            modeled.architecture.isin(candidate_architectures)
            & modeled.front_mrr_slice_bits.isin(ASWM_SLICE_BITS)
            & modeled.accumulation.eq("optical")
        ]
        optimizer = ASWMOptimizer(self.manifest.raw["aswm"]["pareto_frontier_limit"])
        selection_rows, summary_rows, frontier_rows = [], [], []
        group_keys = ["network", "architecture", "energy_model", "optical_loss_db_per_stage"]

        def optimize_scenarios(frame: pd.DataFrame, mapping: str, expected_choices) -> None:
            for values, group in frame.groupby(group_keys, sort=False):
                choices_by_layer = {
                    layer: tuple(
                        SliceChoice(
                            layer=layer,
                            slice_bits=int(row.front_mrr_slice_bits),
                            energy_j=float(row.modeled_energy_j),
                            latency_s=float(row.latency_s),
                            cycles=int(row.cycles),
                            stationarity=str(row.stationarity),
                        )
                        for row in layer_group.itertuples(index=False)
                    )
                    for layer, layer_group in group.groupby("layer")
                }
                try:
                    best, frontier = optimizer.optimize(
                        choices_by_layer, expected_choices=expected_choices
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        f"{error}; mapping={mapping}; scenario={dict(zip(group_keys, values))}"
                    ) from error
                common = {**dict(zip(group_keys, values)), "mapping": mapping}
                for choice in best.choices:
                    selection_rows.append({
                        **common, "layer": choice.layer,
                        "stationarity": choice.stationarity,
                        "front_mrr_slice_bits": choice.slice_bits,
                        "energy_j": choice.energy_j, "latency_s": choice.latency_s,
                        "accuracy": float("nan"), "accuracy_status": ACCURACY_STATUS,
                    })
                summary_stationarity = (
                    mapping.removeprefix("ASWM-")
                    if mapping in {"ASWM-WS", "ASWM-IS"}
                    else "MIXED" if mapping == "Mixed-1bit" else "JOINT"
                )
                summary_rows.append({
                    **common, "stationarity": summary_stationarity,
                    "layers": len(best.choices), "energy_j": best.energy_j,
                    "latency_s": best.latency_s, "edp_j_s": best.edp_j_s,
                    "frontier_states": len(frontier), "accuracy": float("nan"),
                    "accuracy_status": ACCURACY_STATUS,
                })
                best_index = min(range(len(frontier)), key=lambda index: frontier[index].edp_j_s)
                indices = range(len(frontier))
                if len(frontier) > PARETO_ARTIFACT_MAX_POINTS:
                    indices = sorted(set(np.linspace(
                        0, len(frontier) - 1, PARETO_ARTIFACT_MAX_POINTS,
                        dtype=np.int64,
                    ).tolist() + [best_index]))
                for index in indices:
                    state = frontier[index]
                    frontier_rows.append({
                        **common, "frontier_index": index, "energy_j": state.energy_j,
                        "latency_s": state.latency_s, "edp_j_s": state.edp_j_s,
                        "selected": index == best_index,
                        "frontier_total_states": len(frontier),
                        "artifact_sampled": len(frontier) > PARETO_ARTIFACT_MAX_POINTS,
                    })

        for stationarity in ("WS", "IS"):
            optimize_scenarios(
                osa[osa.stationarity == stationarity],
                f"ASWM-{stationarity}",
                {(stationarity, bits) for bits in ASWM_SLICE_BITS},
            )
        optimize_scenarios(
            osa,
            "Joint-ASWM",
            {(stationarity, bits) for stationarity in ("WS", "IS") for bits in ASWM_SLICE_BITS},
        )
        optimize_scenarios(
            osa[osa.front_mrr_slice_bits == 1],
            "Mixed-1bit",
            {(stationarity, 1) for stationarity in ("WS", "IS")},
        )
        return pd.DataFrame(selection_rows), pd.DataFrame(summary_rows), pd.DataFrame(frontier_rows)

    @staticmethod
    def _rankings(aswm: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, loss, mapping), scenario in aswm.groupby(
            ["energy_model", "optical_loss_db_per_stage", "mapping"]
        ):
            for architecture, group in scenario.groupby("architecture"):
                values = group.edp_j_s.astype(float)
                rows.append({
                    "energy_model": model, "optical_loss_db_per_stage": loss,
                    "mapping": mapping, "architecture": architecture,
                    "geometric_mean_edp": math.exp(sum(map(math.log, values)) / len(values)),
                })
        ranking = pd.DataFrame(rows)
        if not ranking.empty:
            ranking["rank"] = ranking.groupby(
                ["energy_model", "optical_loss_db_per_stage", "mapping"]
            ).geometric_mean_edp.rank(method="min").astype(int)
        return ranking.sort_values(["energy_model", "optical_loss_db_per_stage", "mapping", "rank"])

    def _plots(self, raw: pd.DataFrame, fixed: pd.DataFrame, selections: pd.DataFrame, frontiers: pd.DataFrame) -> tuple[Path, ...]:
        import matplotlib.pyplot as plt

        paths = []
        primary = fixed[(fixed.energy_model == "linear_bit") & (fixed.optical_loss_db_per_stage == 0)]
        if not primary.empty:
            heat = primary.pivot_table(index=["stationarity", "architecture"], columns="front_mrr_slice_bits", values="edp_j_s", aggfunc="mean")
            axis = heat.plot.bar(logy=True, figsize=(12, 5), ylabel="Mean network EDP (J·s)")
            axis.set_title("MB-OSA EDP by core and slice width")
            plt.tight_layout(); path = self.artifacts / "edp_core_slice.png"; plt.savefig(path, dpi=160); plt.close(); paths.append(path)
        primary_frontier = frontiers[(frontiers.energy_model == "linear_bit") & (frontiers.optical_loss_db_per_stage == 0)]
        if not primary_frontier.empty:
            axis = primary_frontier.plot.scatter(x="latency_s", y="energy_j", c="edp_j_s", logx=True, logy=True, figsize=(8, 6))
            axis.set_title("ASWM energy-delay Pareto points")
            plt.tight_layout(); path = self.artifacts / "energy_delay_pareto.png"; plt.savefig(path, dpi=160); plt.close(); paths.append(path)
        primary_selection = selections[
            (selections.energy_model == "linear_bit")
            & (selections.optical_loss_db_per_stage == 0)
            & selections.mapping.eq("Joint-ASWM")
        ]
        if not primary_selection.empty:
            primary_aswm = self._aswm_results_for_plot(primary_selection)
            heat_columns = {}
            for network, network_rows in primary_aswm.groupby("network"):
                ordered = network_rows.sort_values("layer")
                heat_columns[network] = pd.Series(
                    ordered.front_mrr_slice_bits.astype(float).to_numpy()
                )
            heat = pd.DataFrame(heat_columns)
            figure, axis = plt.subplots(figsize=(10, 7))
            image = axis.imshow(heat, aspect="auto", interpolation="nearest", vmin=1, vmax=4)
            axis.set_xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right")
            axis.set_ylabel("Layer index (natural manifest order)")
            axis.set_title("ASWM slice-width selections at each workload's best core")
            colorbar = figure.colorbar(image, ax=axis, ticks=[1, 2, 4])
            colorbar.set_label("Front-MRR slice bits")
            plt.tight_layout(); path = self.artifacts / "aswm_selection_heatmap.png"; plt.savefig(path, dpi=160); plt.close(); paths.append(path)
        if not raw.empty:
            component_rows = []
            for row in raw.to_dict("records"):
                breakdown = _mapping(row.get("energy_breakdown", {}))
                front_dac = "input_dac" if row["sliced_operand"] == "input" else "weight_dac"
                component_rows.append({
                    "front_mrr_slice_bits": row["front_mrr_slice_bits"],
                    "Front DAC": _component_sum(breakdown, front_dac),
                    "Laser": _component_sum(breakdown, "laser"),
                    "ADC": _component_sum(breakdown, "adc"),
                    "OSA": _component_sum(breakdown, "delay_line"),
                    "Other": max(float(row["energy_j"]) - sum(float(v) for v in breakdown.values() if isinstance(v, (int, float))), 0.0),
                })
            components = pd.DataFrame(component_rows).groupby("front_mrr_slice_bits").sum()
            axis = components.plot.bar(stacked=True, figsize=(9, 5), ylabel="Summed layer energy (J)")
            axis.set_title("MB-OSA component-energy sensitivity")
            plt.tight_layout(); path = self.artifacts / "component_energy_breakdown.png"; plt.savefig(path, dpi=160); plt.close(); paths.append(path)
        return tuple(paths)

    @staticmethod
    def _aswm_results_for_plot(primary_selection: pd.DataFrame) -> pd.DataFrame:
        """Select one best-core layer mapping per workload for a readable heatmap."""
        totals = (
            primary_selection.groupby(["network", "architecture"], as_index=False)
            .agg(energy_j=("energy_j", "sum"), latency_s=("latency_s", "sum"))
        )
        totals["edp_j_s"] = totals.energy_j * totals.latency_s
        best_cores = (
            totals.sort_values("edp_j_s").groupby("network", as_index=False).first()
        )
        return primary_selection.merge(
            best_cores[["network", "architecture"]],
            on=["network", "architecture"],
            validate="many_to_one",
        )

    def _report(self, checks: pd.DataFrame, fixed: pd.DataFrame, aswm: pd.DataFrame, rankings: pd.DataFrame) -> str:
        metadata = json.loads((self.run_dir / "run.json").read_text())
        tier = str(metadata["tier"])
        hard_failure = bool(((checks.severity == "ERROR") & (checks.status == "FAIL")).any())
        status = "FAIL" if hard_failure else "PASS"
        primary_fixed = fixed[(fixed.energy_model == "linear_bit") & (fixed.optical_loss_db_per_stage == 0)]
        primary_aswm = aswm[(aswm.energy_model == "linear_bit") & (aswm.optical_loss_db_per_stage == 0)]
        result_heading = (
            "Smoke diagnostic aggregates" if tier == "smoke"
            else "Primary-model best results"
        )
        lines = ["# MB-OSA and ASWM Experiment Report", "", f"Overall status: **{status}**", "",
                 f"Run tier: **{tier.upper()}**.", "",
                 "Accuracy status: **NOT_MODELED**. No accuracy constraint or claim is applied.", ""]
        provenance = metadata.get("provenance", {})
        lines.extend([
            "## Environment provenance", "",
            f"- Run ID: `{metadata.get('run_id', 'unknown')}`",
            f"- Manifest SHA-256: `{metadata.get('manifest_digest', 'unknown')}`",
            f"- Source commit: `{provenance.get('git_commit', 'unknown')}`",
            f"- Timeloop mapper: {provenance.get('timeloop_mapper', 'unknown')}",
            f"- TimeloopFE: {provenance.get('timeloopfe', 'unknown')}",
            f"- Accelergy: {provenance.get('accelergy', 'unknown')}",
            f"- Python: {provenance.get('python', 'unknown')}",
            f"- Jobs: {metadata.get('successful_jobs', 0)} successful, "
            f"{metadata.get('failed_jobs', 0)} failed, "
            f"{metadata.get('remaining_jobs', 0)} remaining",
            f"- Created: {metadata.get('created_at', 'unknown')}",
            f"- Updated: {metadata.get('updated_at', 'unknown')}", "",
        ])
        reporting_tolerance = float(
            self.manifest.raw["tolerances"]["edp_relative"]
        )
        if tier == "smoke":
            lines.extend([
                "This smoke run covers one representative layer per workload. Its aggregates "
                "verify the native simulation and analysis path; they are not whole-network "
                "EDP results and must not be used for research conclusions.", "",
            ])
        lines.extend([f"## {result_heading}", ""])
        if not primary_aswm.empty:
            primary_comparison = self._scenario_comparison(primary_fixed, primary_aswm)
            strict_reductions = 0
            for (mapping, network), group in primary_aswm.groupby(["mapping", "network"]):
                best = group.sort_values("edp_j_s").iloc[0]
                comparison = primary_comparison[
                    (primary_comparison.mapping == mapping)
                    & (primary_comparison.network == network)
                    & (primary_comparison.architecture == best.architecture)
                ].iloc[0]
                analog_candidates = primary_fixed[
                    (primary_fixed.network == network)
                    & (primary_fixed.architecture == best.architecture)
                    & (primary_fixed.front_mrr_slice_bits == 8)
                ]
                if mapping in {"ASWM-WS", "ASWM-IS"}:
                    analog_candidates = analog_candidates[
                        analog_candidates.stationarity == best.stationarity
                    ]
                analog = analog_candidates.sort_values("edp_j_s").iloc[0]
                reduction = float(comparison.aswm_reduction_vs_best_fixed)
                analog_change = best.edp_j_s / analog.edp_j_s - 1
                if reduction > reporting_tolerance:
                    strict_reductions += 1
                    outcome = "strict reduction"
                elif reduction < -reporting_tolerance:
                    outcome = "worse"
                else:
                    outcome = "equal within numerical tolerance"
                lines.append(
                    f"- {mapping}/{network}: {best.architecture}, EDP={best.edp_j_s:.6g}; "
                    f"{outcome} versus same-core best fixed ({reduction:.6%}); "
                    f"EDP change versus analog reference={analog_change:+.3%}"
                )
            if not strict_reductions:
                all_scenarios = self._scenario_comparison(fixed, aswm)
                sensitivity_reductions = int(
                    (all_scenarios.aswm_reduction_vs_best_fixed > reporting_tolerance).sum()
                )
                finding_prefix = "**Smoke observation:**" if tier == "smoke" else "**Finding:**"
                qualification = (
                    " This is not a whole-network finding; run the full tier before making an "
                    "adaptive-EDP claim."
                    if tier == "smoke" else ""
                )
                lines.extend([
                    "",
                    f"{finding_prefix} none of the adaptive policies strictly improves primary-model "
                    "EDP over its eligible fixed mapping; no adaptive-EDP improvement is claimed. "
                    "Across all "
                    f"{len(all_scenarios)} core/workload/sensitivity comparisons, "
                    f"{sensitivity_reductions} show a strict adaptive reduction.{qualification}",
                ])
        lines.extend(["", "## Shared-core ranking", ""])
        primary_rank = rankings[(rankings.energy_model == "linear_bit") & (rankings.optical_loss_db_per_stage == 0)]
        for row in primary_rank.itertuples(index=False):
            lines.append(f"- {row.mapping} #{row.rank} {row.architecture}: geometric-mean EDP={row.geometric_mean_edp:.6g}")
        lines.extend(["", "## DAC and optical-loss sensitivity", ""])
        if not aswm.empty:
            sensitivity = aswm.groupby(
                ["energy_model", "optical_loss_db_per_stage"], as_index=False
            ).edp_j_s.agg(["min", "median", "max"]).reset_index()
            lines.extend([
                "| DAC model | Loss per stage (dB) | Minimum EDP | Median EDP | Maximum EDP |",
                "|---|---:|---:|---:|---:|",
            ])
            for row in sensitivity.itertuples(index=False):
                lines.append(
                    f"| {row.energy_model} | {row.optical_loss_db_per_stage:g} | "
                    f"{row.min:.6g} | {row.median:.6g} | {row.max:.6g} |"
                )
        lines.extend(["", "## Validation", "", "| Severity | Status | Check | Detail |", "|---|---|---|---|"])
        for row in checks.itertuples(index=False):
            lines.append(f"| {row.severity} | {row.status} | {row.check} | {str(row.detail).replace('|', '/')} |")
        return "\n".join(lines) + "\n"

    def _execute_notebook(self) -> Path:
        source = self.manifest.repo_root / "examples/rosa/mb_osa_aswm_experiments.ipynb"
        output = self.artifacts / "mb_osa_aswm_experiments.executed.ipynb"
        environment = os.environ.copy(); environment["OPTICALLOOP_MULTISLICE_RUN_DIR"] = str(self.run_dir)
        kernel_root = self.artifacts / ".jupyter/kernels/opticalloop-multislice"; kernel_root.mkdir(parents=True, exist_ok=True)
        (kernel_root / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "OpticalLoop multislice", "language": "python",
        }))
        environment["JUPYTER_PATH"] = str(self.artifacts / ".jupyter")
        completed = subprocess.run([
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
            str(source), "--output", output.name, "--output-dir", str(self.artifacts),
            "--ExecutePreprocessor.timeout=3600", "--ExecutePreprocessor.kernel_name=opticalloop-multislice",
        ], cwd=self.manifest.repo_root, env=environment, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"Notebook execution failed:\n{completed.stdout}\n{completed.stderr}")
        return output


class MultiSliceValidator:
    """Validate MB-OSA simulation coverage, scaling, ASWM, and empty accuracy."""

    def __init__(self, manifest: ExperimentManifest, run_dir: Path) -> None:
        self.manifest = manifest
        self.run_dir = Path(run_dir)

    def validate(self, raw: pd.DataFrame, modeled: pd.DataFrame, fixed: pd.DataFrame,
                 selections: pd.DataFrame, aswm: pd.DataFrame) -> pd.DataFrame:
        metadata = json.loads((self.run_dir / "run.json").read_text())
        expected = int(metadata["expected_jobs"])
        expected_jobs = self.manifest.jobs(
            metadata["tier"], manifest_digest=metadata["manifest_digest"]
        )
        expected_ids = {job.job_id for job in expected_jobs}
        actual_ids = set(raw.job_id) if not raw.empty else set()
        result_paths = tuple((self.run_dir / "jobs").glob("*.json"))
        result_payloads = [json.loads(path.read_text()) for path in result_paths]
        failed_results = [payload for payload in result_payloads if payload.get("status") != "success"]
        rows = [
            self._check("job_coverage", len(raw) == expected, f"{len(raw)}/{expected}"),
            self._check("no_duplicate_jobs", raw.empty or raw.job_id.is_unique, f"unique={raw.job_id.nunique() if not raw.empty else 0}"),
            self._check(
                "exact_expected_jobs",
                actual_ids == expected_ids,
                f"missing={len(expected_ids - actual_ids)}, unexpected={len(actual_ids - expected_ids)}",
            ),
            self._check(
                "no_failed_job_results",
                not failed_results and len(result_paths) == expected,
                f"result_files={len(result_paths)}, failed={len(failed_results)}",
            ),
            self._check("supported_slice_widths", raw.empty or set(raw.front_mrr_slice_bits) == {1, 2, 4, 8}, str(sorted(raw.front_mrr_slice_bits.unique()) if not raw.empty else [])),
            self._check("temporal_slice_counts", raw.empty or bool((raw.temporal_slices == (8 // raw.front_mrr_slice_bits)).all()), "expected 8/4/2/1"),
            self._check(
                "temporal_accumulation_counts",
                raw.empty or bool(
                    (raw.temporal_accumulations == raw.temporal_slices - 1).all()
                    and raw.loc[raw.accumulation == "none", "temporal_accumulations"].eq(0).all()
                ),
                "expected 7/3/1/0",
            ),
            self._check(
                "accuracy_not_modeled",
                raw.empty or bool(
                    raw.accuracy.isna().all()
                    and raw.accuracy_delta.isna().all()
                    and ~raw.accuracy_constraint.astype(bool).any()
                    and raw.accuracy_status.eq(ACCURACY_STATUS).all()
                ),
                ACCURACY_STATUS,
            ),
        ]
        if not raw.empty:
            component_presence_ok = True
            for row in raw.itertuples(index=False):
                components = set(_mapping(row.energy_breakdown))
                has_delay = any(name == "delay_line" or name.endswith(".delay_line") for name in components)
                has_digital = any(name == "digital_shift_add" or name.endswith(".digital_shift_add") for name in components)
                expected_delay = row.accumulation == "optical"
                expected_digital = row.accumulation == "digital"
                component_presence_ok &= has_delay == expected_delay and has_digital == expected_digital
            rows.append(self._check(
                "accumulation_component_selection",
                component_presence_ok,
                "optical=delay_line, digital=digital_shift_add, 8bit=bypass",
            ))
            if "mapping_text" in raw and raw.mapping_text.notna().all():
                stationarity_mapping_ok = True
                for row in raw[raw.front_mrr_slice_bits < 8].itertuples(index=False):
                    accumulator = "delay_line" if row.accumulation == "optical" else "digital_shift_add"
                    section = str(row.mapping_text).split(f"{accumulator} [", 1)
                    if len(section) != 2:
                        stationarity_mapping_ok = False
                        continue
                    section = section[1].split("laser [", 1)[0]
                    expected_dimension = "for X in" if row.stationarity == "WS" else "for Y in"
                    stationarity_mapping_ok &= expected_dimension in section
                    if row.stationarity == "IS" and row.pes > 1:
                        pe_parts = str(row.mapping_text).split("inter_photonic_pe_spatial [", 1)
                        if len(pe_parts) != 2:
                            stationarity_mapping_ok = False
                            continue
                        pe_section = pe_parts[1].split("input_dac [", 1)[0]
                        spatial_lines = [line for line in pe_section.splitlines() if "(Spatial-X)" in line]
                        # Depthwise/grouped layers may have M=1, so Timeloop emits no
                        # PE spatial loop. Empty is valid; any emitted loop must still be M.
                        stationarity_mapping_ok &= all("for M in" in line for line in spatial_lines)
                rows.append(self._check(
                    "native_mapping_stationarity",
                    stationarity_mapping_ok,
                    "WS accumulator traverses X; IS accumulator traverses Y and PE spatial loops only traverse M",
                ))
            expected_layers = {
                network: {job.layer for job in expected_jobs if job.network == network}
                for network in self.manifest.networks
            }
            layer_mismatches = {
                network: {
                    "missing": len(layers - set(raw.loc[raw.network == network, "layer"])),
                    "unexpected": len(set(raw.loc[raw.network == network, "layer"]) - layers),
                }
                for network, layers in expected_layers.items()
            }
            rows.append(self._check(
                "workload_layer_completeness",
                all(not detail["missing"] and not detail["unexpected"] for detail in layer_mismatches.values()),
                json.dumps(layer_mismatches, sort_keys=True),
            ))
            architectures = {
                architecture["name"]: architecture
                for architecture in self.manifest.raw["architectures"]
            }
            constraints = self.manifest.raw["constraints"]
            invalid_candidates = [
                name for name, architecture in architectures.items()
                if architecture["candidate"] and (
                    architecture["cols"] > constraints["max_candidate_cols"]
                    or architecture["pes"] * architecture["cols"] * architecture["rows"]
                    > constraints["max_weight_mrrs"]
                )
            ]
            raw_shape_mismatches = raw.apply(
                lambda row: any(
                    int(row[key]) != int(architectures[row.architecture][key])
                    for key in ("tiles", "pes", "cols", "rows")
                ),
                axis=1,
            )
            rows.append(self._check(
                "architecture_constraints",
                not invalid_candidates and not bool(raw_shape_mismatches.any()),
                f"invalid_candidates={invalid_candidates}, shape_mismatches={int(raw_shape_mismatches.sum())}",
            ))
            positive_units = (
                raw.energy_j.gt(0) & raw.latency_s.gt(0) & raw.cycles.gt(0)
                & raw.cycle_seconds.gt(0)
            )
            latency_relative = (
                raw.latency_s - raw.cycles * raw.cycle_seconds
            ).abs() / raw.latency_s.abs().clip(lower=1e-30)
            rows.append(self._check(
                "unit_consistency",
                bool(positive_units.all() and (latency_relative <= 1e-9).all()),
                f"positive={bool(positive_units.all())}, max_latency_relative={latency_relative.max():.3e}",
            ))
            expected_cycle_seconds = 1.0 / float(self.manifest.raw["frequency_hz"])
            frequency_relative = (
                raw.cycle_seconds - expected_cycle_seconds
            ).abs() / expected_cycle_seconds
            rows.append(self._check(
                "frequency_consistency",
                bool((frequency_relative <= 1e-12).all()),
                f"expected_hz={self.manifest.raw['frequency_hz']}, max_relative={frequency_relative.max():.3e}",
            ))
            osa = raw[
                raw.front_mrr_slice_bits.isin(ASWM_SLICE_BITS)
                & raw.accumulation.eq("optical")
            ].copy()
            osa["front_dac_energy_j"] = osa.apply(
                lambda row: _component_sum(
                    _mapping(row.energy_breakdown),
                    "input_dac" if row.sliced_operand == "input" else "weight_dac",
                ),
                axis=1,
            )
            dac = osa.pivot(
                index=["network", "layer", "architecture", "stationarity"],
                columns="front_mrr_slice_bits",
                values="front_dac_energy_j",
            )
            dac_distinct = (dac[2] > dac[1]) & (dac[4] > dac[2])
            rows.append(self._check(
                "native_dac_resolution_distinct",
                bool(
                    dac.notna().all().all()
                    and (dac > 0).all().all()
                    and dac_distinct.all()
                ),
                f"strictly_increasing={bool(dac_distinct.all())}",
            ))
            osa_area = raw[raw.macro.isin(["mrr_ws_osa", "mrr_is_osa"])].copy()
            mrr_area_rows = []
            for row in osa_area.itertuples(index=False):
                breakdown = _mapping(row.area_breakdown)
                mrr_area_rows.append({
                    "network": row.network, "layer": row.layer,
                    "architecture": row.architecture, "stationarity": row.stationarity,
                    "front_mrr_slice_bits": row.front_mrr_slice_bits,
                    "input_mrr": _component_sum(breakdown, "input_mrr"),
                    "weight_mrr": _component_sum(breakdown, "weight_mrr"),
                })
            mrr_areas = pd.DataFrame(mrr_area_rows)
            area_spread = mrr_areas.groupby(
                ["network", "layer", "architecture", "stationarity"]
            )[["input_mrr", "weight_mrr"]].agg(lambda values: values.max() - values.min())
            rows.append(self._check(
                "physical_mrr_area_invariant",
                bool((area_spread.abs() <= 1e-18).all().all()),
                f"max_spread_mm2={area_spread.abs().max().max():.3e}",
            ))
            analog = raw[raw.front_mrr_slice_bits == 8]
            analog_pivot = analog.pivot(
                index=["network", "layer", "architecture", "stationarity"],
                columns="macro", values=["cycles", "energy_j"],
            )
            analog_cycle_equal = []
            analog_energy_relative = []
            for stationarity, no_osa_macro, osa_macro in (
                ("WS", "mrr_ws_no_osa", "mrr_ws_osa"),
                ("IS", "mrr_is_no_osa", "mrr_is_osa"),
            ):
                stationarity_rows = analog_pivot.xs(stationarity, level="stationarity")
                analog_cycle_equal.append(
                    stationarity_rows["cycles"][no_osa_macro].eq(
                        stationarity_rows["cycles"][osa_macro]
                    ).all()
                )
                analog_energy_relative.extend((
                    stationarity_rows["energy_j"][no_osa_macro]
                    - stationarity_rows["energy_j"][osa_macro]
                ).abs() / stationarity_rows["energy_j"][no_osa_macro].clip(lower=1e-30))
            rows.append(self._check(
                "analog_8bit_bypass_equivalence",
                bool(all(analog_cycle_equal) and max(analog_energy_relative) <= 1e-12),
                f"cycles_equal={all(analog_cycle_equal)}, max_energy_relative={max(analog_energy_relative):.3e}",
            ))
            relative = (raw.edp_j_s - raw.energy_j * raw.latency_s).abs() / raw.edp_j_s.abs().clip(lower=1e-30)
            rows.append(self._check("edp_formula", bool((relative <= self.manifest.raw["tolerances"]["edp_relative"]).all()), f"max={relative.max():.3e}"))
        primary_fixed = fixed[(fixed.energy_model == "linear_bit") & (fixed.optical_loss_db_per_stage == 0)]
        primary_aswm = aswm[(aswm.energy_model == "linear_bit") & (aswm.optical_loss_db_per_stage == 0)]
        if not primary_aswm.empty:
            worse = []
            for row in primary_aswm.itertuples(index=False):
                candidates = primary_fixed[
                    (primary_fixed.network == row.network)
                    & (primary_fixed.architecture == row.architecture)
                    & primary_fixed.accumulation.eq("optical")
                    & primary_fixed.front_mrr_slice_bits.isin(ASWM_SLICE_BITS)
                ]
                if row.mapping in {"ASWM-WS", "ASWM-IS"}:
                    candidates = candidates[candidates.stationarity == row.stationarity]
                elif row.mapping == "Mixed-1bit":
                    candidates = candidates[candidates.front_mrr_slice_bits == 1]
                if row.edp_j_s > candidates.edp_j_s.min() * (1 + 1e-12):
                    worse.append(f"{row.network}/{row.architecture}")
            rows.append(self._check("aswm_no_worse_than_fixed", not worse, f"worse={worse[:3]}"))
        if metadata["tier"] == "smoke" and not raw.empty:
            representative = raw[
                (raw.network == "alexnet")
                & (raw.stationarity == "WS")
                & raw.accumulation.isin(["optical", "none"])
            ].drop_duplicates("front_mrr_slice_bits").set_index("front_mrr_slice_bits")
            if set(representative.index) == {1, 2, 4, 8}:
                base = float(representative.loc[1, "cycles"])
                ratios = {bits: float(representative.loc[bits, "cycles"]) / base for bits in (1, 2, 4)}
                tolerance = self.manifest.raw["tolerances"]["cycle_ratio_relative"]
                passed = all(abs(ratios[bits] - 1 / bits) <= tolerance / bits for bits in ratios)
                rows.append(self._check("native_cycle_ratios", passed, str(ratios)))
                dac = {bits: _component_sum(_mapping(representative.loc[bits, "energy_breakdown"]), "input_dac") for bits in (1, 2, 4)}
                dac_tolerance = self.manifest.raw["tolerances"]["dac_scaling_relative"]
                dac_ratios = {bits: dac[bits] / dac[1] for bits in (1, 2, 4)}
                dac_passed = all(
                    abs(dac_ratios[bits] - bits) <= dac_tolerance * bits
                    for bits in dac_ratios
                )
                rows.append(self._check("native_dac_energy_scaling", dac_passed, str(dac_ratios)))
        return pd.DataFrame(rows)

    @staticmethod
    def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
        return {"check": name, "severity": "ERROR", "status": "PASS" if passed else "FAIL", "detail": detail}


def _mapping(value: object) -> Mapping[str, float]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _component_sum(values: Mapping[str, object], component: str) -> float:
    return sum(float(value) for name, value in values.items() if name == component or name.endswith(f".{component}"))
