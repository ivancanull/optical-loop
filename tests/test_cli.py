from argparse import Namespace

import optical_loop
from opticalloop.result import SimulationResult


def test_layer_command_uses_layer_simulator(monkeypatch, capsys) -> None:
    captured = {}

    class FakeLayerSimulator:
        def __init__(self, *, layer, architecture, backend, cache):
            captured["layer"] = layer
            captured["architecture"] = architecture
            captured["backend"] = backend
            captured["cache"] = cache

        def run(self):
            return SimulationResult(
                cycles=42,
                latency_s=1e-6,
                energy_j=2e-6,
                area_mm2=3e-3,
                tops_per_w=4.0,
                energy_breakdown={},
            )

    monkeypatch.setattr(optical_loop, "LayerSimulator", FakeLayerSimulator)
    args = Namespace(
        network="alexnet",
        layer="alexnet/0",
        macro="proposed_mrr_optical_shift_add",
        system="fetch_all_lpddr4",
        tiles=1,
        pes=1,
        cols=100,
        rows=12,
        voltage_dac_resolution=1,
        scaling='"aggressive"',
        max_utilization=False,
        cache_results_dir=None,
        cache_save_name="deapcnns",
        cache_output_postfix="_1bit_input_osa",
    )

    optical_loop._run_layer(args)

    output = capsys.readouterr().out
    assert "OpticalLoop Timeloop-backed layer result" in output
    assert captured["layer"].layer_path == "alexnet/0"
    assert captured["architecture"].architecture_key == "T1, P1, C100, R12"
    assert captured["architecture"].macro == "proposed_mrr_optical_shift_add"
