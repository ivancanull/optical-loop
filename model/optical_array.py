"""Optical array model wrappers."""

from dataclasses import dataclass

from opticalloop.config.architecture import MRRMacroConfig


@dataclass(frozen=True)
class OpticalArray:
    """Runtime model for an MRR optical macro array."""

    config: MRRMacroConfig

    @property
    def parallel_macs(self) -> int:
        return self.config.parallel_macs
