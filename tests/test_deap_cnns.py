from pathlib import Path

import pandas as pd
import pytest
import yaml

import optical_loop
from opticalloop.applications.deap_cnns import (
    DeapArchitectureSetting,
    DeapDeviceSpec,
    DeapResultValidator,
    default_deap_workflow,
    write_deap_artifacts,
)


class FakeStats:
    cycles = 10
    cycle_seconds = 2e-10
    latency = 2e-9
    energy = 3e-6
    area = 4e-3
    tops = 5.0
    tops_per_w = 6.0
    tops_per_mm2 = 7.0
    computes = 8.0
    per_component_energy = {"laser": 1e-6}
    per_component_area = {"laser": 2e-3}
    per_component_power = {"laser": 3.0}


def test_deap_device_spec_matches_article_constants() -> None:
    device = DeapDeviceSpec()

    assert device.mrr_precision_bits == 7
    assert device.mrr_self_coupling == pytest.approx(0.99)
    assert device.mrr_loss == pytest.approx(0.99)
    assert device.max_wavelengths == 100
    assert device.max_modulators == 1024
    assert device.waveguide_width_nm == 500
    assert device.waveguide_thickness_nm == 220
    assert device.waveguide_bend_radius_um == 5
    assert device.output_cycle_ps == 200
    assert device.laser_power_mw == pytest.approx(100.0)
    assert device.adc_power_mw == pytest.approx(76.0)


def test_deap_architecture_constraints() -> None:
    mnist = DeapArchitectureSetting("mnist-default", 1, 5, 8)
    edge_small = DeapArchitectureSetting("edge-small", 1, 3, 113)
    edge_large = DeapArchitectureSetting("edge-large", 1, 10, 10)

    assert mnist.n_wavelengths == 25
    assert mnist.n_modulators == 200
    assert edge_small.n_wavelengths == 9
    assert edge_small.n_modulators == 1017
    assert edge_large.n_wavelengths == 100
    assert edge_large.n_modulators == 1000

    with pytest.raises(ValueError, match="1024"):
        DeapArchitectureSetting("paper-edge-large-stated", 1, 10, 12)


def test_deap_timeloop_asset_contains_device_calibration() -> None:
    variables_path = (
        Path("workspace")
        / "models"
        / "arch"
        / "1_macro"
        / "deap_cnns"
        / "variables_iso.yaml"
    )
    variables = yaml.safe_load(variables_path.read_text())["variables"]

    assert variables["DEAP_ADC_POWER_MW"] == 76
    assert variables["ADC_ENERGY_SCALE"] == pytest.approx(0.076 / 0.52136)
    assert variables["DEAP_OUTPUT_CYCLE_PS"] == 200


def test_deap_component_csv_encodes_device_power() -> None:
    components_path = (
        Path("workspace")
        / "models"
        / "arch"
        / "1_macro"
        / "deap_cnns"
        / "components"
        / "7nm_components.csv"
    )
    text = components_path.read_text()

    assert "7nm,deapcnns,7,leak,100e9" in text
    assert text.count("7nm,deapcnns,7,leak,19.5e9") == 3
    assert "7nm,deapcnns,7,write|update,5.2" in text
    assert "7nm,deapcnns,7,convert|read,3.4" in text


def test_deap_workflow_forwards_to_timeloop_backend(tmp_path: Path) -> None:
    calls = []

    def fake_quick_run(**kwargs):
        calls.append(kwargs)
        return FakeStats()

    from opticalloop.backend import TimeloopBackend

    workflow = default_deap_workflow(
        architecture_name="mnist-default",
        results_dir=tmp_path,
        backend=TimeloopBackend(quick_run=fake_quick_run),
    )

    workflow.run_sweeps()

    assert [call["layer"] for call in calls] == [
        "deap_mnist/conv0",
        "deap_mnist/conv1",
    ]
    assert all(call["macro"] == "deap_cnns" for call in calls)
    assert all(call["system"] == "fetch_all_lpddr4" for call in calls)
    assert calls[0]["variables"] == {
        "SCALING": '"deapcnns"',
        "N_TILES": 1,
        "N_PES": 1,
        "N_COLUMNS": 25,
        "N_ROWS": 8,
        "VOLTAGE_DAC_RESOLUTION": 7,
    }
    assert calls[0]["max_utilization"] is False


def test_deap_artifacts_and_validator(tmp_path: Path) -> None:
    outputs = write_deap_artifacts(tmp_path)

    assert set(outputs) == {
        "device_parameters_deap_cnns.csv",
        "architecture_summary_deap_cnns.csv",
        "validation_report.csv",
    }
    report = DeapResultValidator(tmp_path).validate()
    assert report["passed"].all(), report[~report["passed"]].to_string(index=False)

    device = pd.read_csv(tmp_path / "device_parameters_deap_cnns.csv")
    assert "mrr_precision_bits" in set(device["parameter"])


def test_deap_cli_report(capsys) -> None:
    args = type(
        "Args",
        (),
        {
            "architecture": "mnist-default",
            "results_dir": Path("results"),
            "n_jobs": 1,
            "mode": "cache",
            "stage": "report",
            "artifact_dir": Path("examples/deap_cnns"),
            "validation_report": None,
        },
    )()

    optical_loop._run_deap(args)

    output = capsys.readouterr().out
    assert "OpticalLoop DEAP-CNNs report" in output
    assert "mnist-default" in output
