from typing import Sequence
from dataclasses import dataclass
from gjive.simulation_spec import SimulationSpec

@dataclass
class EstimateSpec:
    r: int
    rfk: Sequence[int]
    rk: Sequence[int]
    use_irlb: bool = True

    @classmethod
    def from_simulation(cls, sim: SimulationSpec) -> "EstimateSpec":
        return cls(
            r=sim.r,
            rfk = sim.rfk.copy(),
            rk = sim.rk.copy(),
        )


# Currently assumes ranks match generated simulation.
# Future experiments may intentionally vary these values
# independently from SimulationSpec.