import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


DEAP_MACRO_DIR = Path("workspace") / "models" / "arch" / "1_macro" / "deap_cnns"
DEAP_WORKLOAD_DIR = Path("workspace") / "models" / "workloads" / "deap_deepbench"
DEAP_NOTEBOOK = Path("examples") / "deap_cnns" / "deap_cnns_reproduction.ipynb"


def test_deap_cli_interface_removed() -> None:
    result = subprocess.run(
        [sys.executable, "optical_loop.py", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "deap-cnns" not in result.stdout
    assert not Path("applications/deap_cnns/workflow.py").exists()
    assert not Path("applications/deap_cnns/validation.py").exists()


def test_deap_deepbench_keeps_only_notebook_workloads() -> None:
    workloads = sorted(path.name for path in DEAP_WORKLOAD_DIR.glob("*.yaml"))

    assert workloads == ["bench0.yaml", "bench1.yaml"]
    assert "instance: {N: 4, C: 1, M: 32, P: 72, Q: 349, R: 20, S: 5" in (
        DEAP_WORKLOAD_DIR / "bench0.yaml"
    ).read_text()
    assert "instance: {N: 8, C: 64, M: 128, P: 110, Q: 110, R: 3, S: 3" in (
        DEAP_WORKLOAD_DIR / "bench1.yaml"
    ).read_text()


def test_deap_macro_variables_are_generic_timeloop_inputs() -> None:
    variables = yaml.safe_load((DEAP_MACRO_DIR / "variables_free.yaml").read_text())[
        "variables"
    ]

    assert variables["N_COLUMNS"] == 100
    assert variables["N_ROWS"] == 12
    assert variables["N_Conv"] == 1
    assert variables["N_TILES"] == 1
    assert variables["DAC_UNIT_RESISTANCE"] == 5000
    assert "N_PES" not in variables


def test_deap_canonical_macro_dataflow_and_readout() -> None:
    text = (DEAP_MACRO_DIR / "arch.yaml").read_text()

    conv_unit = _yaml_item_block(text, "conv_unit")
    wavelength_column = _yaml_item_block(text, "wavelength_column")
    channel_weight_row = _yaml_item_block(text, "channel_weight_row")

    assert "spatial: {meshX: N_Conv}" in conv_unit
    assert "maximize_dims: [[P, Q], [N]]" in conv_unit
    assert "spatial: {meshX: N_COLUMNS}" in wavelength_column
    assert "*spatial_must_reuse_outputs" in wavelength_column
    assert "maximize_dims: [[R, S]]" in wavelength_column
    assert "spatial: {meshY: N_ROWS}" in channel_weight_row
    assert "no_reuse: [Outputs, Weights]" in channel_weight_row
    assert "factors: [N=1, M=1, P=1, Q=1, R=1, S=1, X=1]" in channel_weight_row
    assert "maximize_dims: [[C]]" in channel_weight_row
    for block in (conv_unit, wavelength_column):
        assert not re.search(r"\bM\b", block)
        assert not re.search(r"\bK\b", block)
    assert "maximize_dims: [[M]]" not in channel_weight_row
    assert "maximize_dims: [[K]]" not in channel_weight_row

    expected_order = [
        "conv_unit",
        "TIA",
        "photodiode_output_readout",
        "adc",
        "laser",
        "dac",
        "input_mrr",
        "channel_weight_row",
        "wavelength_column",
        "weight_mrr",
    ]
    positions = [text.index(f"    name: {name}\n") for name in expected_order]
    assert positions == sorted(positions)

    assert "n_instances: N_ROWS" in _yaml_item_block(text, "TIA")
    assert "subclass: deap_adc" in _yaml_item_block(text, "adc")
    assert "n_instances: N_COLUMNS" in _yaml_item_block(text, "laser")
    assert "n_instances: N_COLUMNS" in _yaml_item_block(text, "input_mrr")
    dac = _yaml_item_block(text, "dac")
    assert "subclass: dac_r2r_ladder_compound" in dac
    assert "n_instances: N_COLUMNS" in dac
    assert text.count("subclass: dac_r2r_ladder_compound") == 1
    assert len(re.findall(r"^\s+name: .*dac\s*$", text, re.MULTILINE)) == 1
    assert "*virtualized_mac" in text


def test_deap_notebook_uses_generic_backend_and_fixed_cases() -> None:
    notebook = json.loads(DEAP_NOTEBOOK.read_text())
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )

    assert "TimeloopBackend" in source
    assert "TimeloopRun" in source
    assert "TimeloopLayerRef" in source
    assert "TimeloopMacroConfig" in source
    assert "deap_deepbench/bench0" in source
    assert "deap_deepbench/bench1" in source
    assert '"N_COLUMNS": 100' in source
    assert '"N_ROWS": 12' in source
    assert '"N_COLUMNS": 9' in source
    assert '"N_ROWS": 113' in source
    assert "mapping_text" in source


def _yaml_item_block(text: str, name: str) -> str:
    marker = f"    name: {name}\n"
    marker_start = text.find(marker)
    assert marker_start >= 0, name
    item_start = text.rfind("\n  - !", 0, marker_start)
    next_item_start = text.find("\n  - !", marker_start + len(marker))
    if next_item_start < 0:
        next_item_start = len(text)
    return text[item_start:next_item_start]
