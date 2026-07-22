from dataclasses import dataclass
from error_analysis.paramater_sweep import ParameterSweep


@dataclass
class SeedSweep:
    parameter_name: str
    sweeps: dict[int, ParameterSweep]