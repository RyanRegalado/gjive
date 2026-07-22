from typing import Sequence
from dataclasses import dataclass

@dataclass
class EstimateSpec:
    r: int
    rfk: Sequence[int]
    rk: Sequence[int]
    use_irlb: bool = True


# Currently assumes ranks match generated simulation.
# Future experiments may intentionally vary these values
# independently from SimulationSpec.