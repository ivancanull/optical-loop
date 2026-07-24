"""Minimal MRR variation model used by the local accuracy runtime.

The equations and constants match the pinned ONNSim MRR implementation. This
module intentionally contains only the behavior exercised by ResNet18 accuracy
evaluation, avoiding imports through ONNSim's package initializer.
"""

from __future__ import annotations

import torch


class MRRConfig:
    resonance_wavelength = 1538.739
    bandwidth = 1.5068
    gamma = bandwidth / 2
    heater_resistance = 50
    thermal_resistance = 50
    thermal_optic_coefficient = 1.86e-4


class VariationAwareMRR:
    """Map values through an MRR while applying thermal and DAC variation."""

    def __init__(self, config: MRRConfig | None = None) -> None:
        self.config = config or MRRConfig()
        self.has_mapping = False

    def define_thermal_diffusion_behavior(self, noise_std: float = 0.1) -> None:
        self.noise_std = noise_std

    def define_DAC_variation_behavior(self, dac_noise_std: float = 0.05) -> None:
        self.dac_noise_std = dac_noise_std

    def voltage_to_delta_temperature(self, voltage):
        base = (
            voltage**2 / self.config.heater_resistance
        ) * self.config.thermal_resistance
        if isinstance(base, torch.Tensor):
            noise = torch.normal(0, self.noise_std, size=base.shape).to(base.device)
        else:
            # Preserve ONNSim's scalar RNG consumption while mapping at zero noise.
            noise = torch.normal(0, self.noise_std, size=(1,)).item()
        return base + noise

    def delta_temperature_to_voltage(self, delta_temperature):
        return torch.sqrt(
            delta_temperature
            * self.config.heater_resistance
            / self.config.thermal_resistance
        )

    def delta_temperature_to_delta_wavelength(self, delta_temperature):
        beta = self.config.thermal_optic_coefficient
        return self.config.resonance_wavelength * (
            beta * delta_temperature
        ) / (3.48 + beta * delta_temperature)

    def delta_wavelength_to_delta_temperature(self, delta_wavelength):
        beta = self.config.thermal_optic_coefficient
        return (3.48 * delta_wavelength) / (
            beta * (self.config.resonance_wavelength - delta_wavelength)
        )

    def voltage_to_resonant_wavelength(self, voltage):
        return self.config.resonance_wavelength + self.delta_temperature_to_delta_wavelength(
            self.voltage_to_delta_temperature(voltage)
        )

    def wavelength_to_transmission(self, wavelength, resonance_wavelength):
        gamma_squared = self.config.gamma**2
        return gamma_squared / (
            (wavelength - resonance_wavelength) ** 2 + gamma_squared
        )

    def set_weight_mapping_range(
        self, voltage_min, voltage_max, reference_wavelength,
        value_min=-1, value_max=1,
    ) -> None:
        thermal_std = self.noise_std
        dac_std = self.dac_noise_std
        self.noise_std = 0.0
        self.dac_noise_std = 0.0
        self.value_min = value_min
        self.value_max = max(value_max, 1e-6)
        self.value_range = self.value_max - self.value_min
        self.reference_wavelength = reference_wavelength
        self.transmission_upper = self.wavelength_to_transmission(
            reference_wavelength, self.voltage_to_resonant_wavelength(voltage_min)
        )
        self.transmission_lower = self.wavelength_to_transmission(
            reference_wavelength, self.voltage_to_resonant_wavelength(voltage_max)
        )
        self.transmission_range = self.transmission_upper - self.transmission_lower
        if self.transmission_lower >= self.transmission_upper:
            raise ValueError("MRR mapping transmission range must be increasing")
        self.has_mapping = True
        self.noise_std = thermal_std
        self.dac_noise_std = dac_std

    def weights_to_voltages(self, values):
        if not self.has_mapping:
            raise ValueError("MRR mapping range has not been set")
        transmission = (
            (values - self.value_min)
            / self.value_range
            * self.transmission_range
            + self.transmission_lower
        )
        delta_wavelength = (
            torch.sqrt(
                self.config.gamma**2 / transmission - self.config.gamma**2
            )
            + self.reference_wavelength
            - self.config.resonance_wavelength
        )
        return self.delta_temperature_to_voltage(
            self.delta_wavelength_to_delta_temperature(delta_wavelength)
        )

    def voltages_to_weights(self, voltages):
        resonance = self.voltage_to_resonant_wavelength(voltages)
        transmission = self.wavelength_to_transmission(
            self.reference_wavelength, resonance
        )
        return (
            (transmission - self.transmission_lower)
            / self.transmission_range
            * self.value_range
            + self.value_min
        )

    def simulate_variation(self, values):
        voltages = self.weights_to_voltages(values)
        varied_voltages = voltages + torch.normal(
            0, self.dac_noise_std, size=voltages.shape
        ).to(voltages.device)
        return self.voltages_to_weights(varied_voltages)
