from dataclasses import dataclass, field
import numpy as np


@dataclass(slots=True)
class SimulationSpec:
    n: int
    K: int
    r: int
    rfk: list[int]
    rk: list[int]
    p: float
    seed: int
    group_assignment: list[int] = field(init=False)

    def __post_init__(self) -> None:

        if len(self.rk) != self.K:
            raise ValueError("Length of rk must equal K.")

        if len(self.rfk) != 2:
            raise ValueError(
                "rfk must contain exactly two group ranks for binary grouping."
            )

        if self.r < 0 or self.r > self.n:
            raise ValueError(
                "Joint rank r must be between 0 and n."
            )

        if any(rank < 0 or rank > self.n for rank in self.rfk):
            raise ValueError(
                "Group ranks must be between 0 and n."
            )

        if any(rank < 0 or rank > self.n for rank in self.rk):
            raise ValueError(
                "Individual ranks must be between 0 and n."
            )

        if self.p <= 0 or self.p >= 1:
            raise ValueError(
                "p must be strictly between 0 and 1."
            )
        
        rng = np.random.default_rng(self.seed)

        self.group_assignment = rng.binomial(
            n=1,
            p=self.p,
            size=self.K
        ).tolist()
