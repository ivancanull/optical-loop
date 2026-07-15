from opticalloop.backend import TimeloopBackend, TimeloopRun
from opticalloop.config.architecture import MRRMacroConfig, TimeloopMacroConfig
from opticalloop.config.workload import TimeloopLayerRef


class FakeStats:
    cycles = 10
    cycle_seconds = 2e-9
    latency = 2e-8
    energy = 3e-6
    area = 4e-3
    tops = 5.0
    tops_per_w = 6.0
    tops_per_mm2 = 7.0
    computes = 8.0
    per_component_energy = {"adc": 1e-6}
    per_component_area = {"adc": 2e-3}
    per_component_power = {"adc": 3.0}
    mapping = "for P in [0:1)\n  << Compute >>"


def test_backend_run_layer_forwards_timeloop_kwargs() -> None:
    calls = []

    def fake_quick_run(**kwargs):
        calls.append(kwargs)
        return FakeStats()

    backend = TimeloopBackend(quick_run=fake_quick_run)
    layer = TimeloopLayerRef(network="toy", layer_path="toy/0")
    architecture = MRRMacroConfig(
        n_tiles=1,
        n_pes=2,
        n_cols=3,
        n_rows=4,
        macro="proposed_mrr",
        system="fetch_all_lpddr4",
        voltage_dac_resolution=2,
        max_utilization=True,
    )

    result = backend.run_layer(layer, architecture)

    assert calls == [
        {
            "macro": "proposed_mrr",
            "layer": "toy/0",
            "variables": {
                "SCALING": '"aggressive"',
                "N_TILES": 1,
                "N_PES": 2,
                "N_COLUMNS": 3,
                "N_ROWS": 4,
                "VOLTAGE_DAC_RESOLUTION": 2,
            },
            "system": "fetch_all_lpddr4",
            "max_utilization": True,
        }
    ]
    assert result.metadata["network"] == "toy"
    assert result.metadata["architecture"] == "T1, P2, C3, R4"
    assert result.mapping_text == FakeStats.mapping


def test_backend_run_batch_forwards_only_timeloop_kwargs() -> None:
    calls = []

    def fake_quick_run(**kwargs):
        calls.append(kwargs)
        return FakeStats()

    backend = TimeloopBackend(quick_run=fake_quick_run)
    architecture = MRRMacroConfig(
        n_tiles=1,
        n_pes=2,
        n_cols=3,
        n_rows=4,
        macro="proposed_mrr",
        system="fetch_all_lpddr4",
        voltage_dac_resolution=2,
        max_utilization=False,
    )
    runs = [
        TimeloopRun(
            layer=TimeloopLayerRef(network="toy", layer_path="toy/0"),
            architecture=architecture,
            metadata={"row_index": "toy_row"},
        )
    ]

    results = backend.run_batch(runs, n_jobs=4)

    assert calls == [
        {
            "macro": "proposed_mrr",
            "layer": "toy/0",
            "variables": {
                "SCALING": '"aggressive"',
                "N_TILES": 1,
                "N_PES": 2,
                "N_COLUMNS": 3,
                "N_ROWS": 4,
                "VOLTAGE_DAC_RESOLUTION": 2,
            },
            "system": "fetch_all_lpddr4",
            "max_utilization": False,
        }
    ]
    assert len(results) == 1
    assert results[0].metadata["row_index"] == "toy_row"
    assert results[0].energy_breakdown == {"adc": 1e-6}


def test_backend_run_layer_forwards_generic_timeloop_config() -> None:
    calls = []

    def fake_quick_run(**kwargs):
        calls.append(kwargs)
        return FakeStats()

    backend = TimeloopBackend(quick_run=fake_quick_run)
    layer = TimeloopLayerRef(network="deap_deepbench", layer_path="deap_deepbench/bench0")
    architecture = TimeloopMacroConfig(
        macro="deap_cnns",
        system="fetch_all_lpddr4",
        variables={"N_COLUMNS": 100, "N_ROWS": 12, "N_Conv": 1},
        architecture_key="deap_cnns:R10_D12",
        max_utilization=False,
    )

    result = backend.run_layer(layer, architecture)

    assert calls == [
        {
            "macro": "deap_cnns",
            "layer": "deap_deepbench/bench0",
            "variables": {"N_COLUMNS": 100, "N_ROWS": 12, "N_Conv": 1},
            "system": "fetch_all_lpddr4",
            "max_utilization": False,
        }
    ]
    assert result.metadata["architecture"] == "deap_cnns:R10_D12"
    assert result.metadata["macro"] == "deap_cnns"
