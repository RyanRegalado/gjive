from dataclasses import dataclass
from error_analysis.experiment_result import ExperimentResult
from typing import Any

@dataclass
class ParameterSweep:
    seed: int
    parameter_name: str
    values: list[Any]
    experiments: dict[Any, list[ExperimentResult]]