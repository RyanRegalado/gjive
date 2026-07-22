from dataclasses import dataclass
from typing import Any


@dataclass
class ExperimentResult:
    seed: int
    parameter_name: str
    parameter_value: Any
    matrix_name: str
    frob_norm: float