from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any

import numpy as np


@dataclass(slots=True)
class GjiveData:
    """
    Container for a simulated GJIVE dataset.

    A GjiveData object loads a previously generated simulation from disk and
    provides convenient access to the stored matrices and metadata.

    Expected folder layout
    ----------------------
    simulation_xxxx/
        simulation.npz
        metadata.json

    simulation.npz contains

    - A   : (K, n, n)
    - U   : (n, r)
    - Uf  : collection of group-wise subspaces
    - Uk  : collection of individual subspaces

    metadata.json contains

    - group_assignment
    - ranks
    - seed
    - other simulation parameters
    """

    path: Path
    A: np.ndarray = field(init=False)
    U: np.ndarray = field(init=False)
    Uf: dict = field(init=False)
    Uk: list = field(init=False)
    V: np.ndarray = field(init = False)
    W: np.ndarray = field(init = False)
    X: np.ndarray = field(init = False)
    metadata: dict[str, Any] = field(init=False)
    group_assignment: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

        npz_path = self.path / "simulation.npz"
        metadata_path = self.path / "metadata.json"

        if not npz_path.exists():
            raise FileNotFoundError(f"Could not find '{npz_path}'.")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Could not find '{metadata_path}'.")

        data = np.load(npz_path, allow_pickle=True)

        self.A: np.ndarray = data["A"]
        self.U: np.ndarray = data["U"]
        self.Uf: np.ndarray = data["Uf"]
        self.Uk: np.ndarray = data["Uk"]

        self.V: np.ndarray = data["V"]
        self.W: np.ndarray = data["W"]
        self.X: np.ndarray = data["X"]

        with metadata_path.open("r") as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.group_assignment = np.asarray(
            self.metadata["group_assignment"],
            dtype=int,
        )

    # ------------------------------------------------------------------
    # Basic dataset information
    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        """Number of matrices in the dataset."""
        return self.A.shape[0]

    @property
    def n(self) -> int:
        """Ambient dimension."""
        return self.A.shape[1]

    @property
    def n_groups(self) -> int:
        """Number of groups."""
        return len(self.Uf)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_matrix(self, k: int) -> np.ndarray:
        """Return observation Ak."""
        return self.A[k]

    def get_joint_subspace(self) -> np.ndarray:
        """Return the joint subspace U."""
        return self.U

    def get_group(self, k: int) -> int:
        """Return the group index for observation k."""
        return int(self.group_assignment[k])

    def get_group_subspace(self, k: int) -> np.ndarray:
        """
        Return the group-wise subspace Uf(k) corresponding to observation k.
        """
        group = self.get_group(k)
        return self.Uf[group]

    def get_individual_subspace(self, k: int) -> np.ndarray:
        """Return the individual subspace Uk."""
        return self.Uk[k]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_observations

    def __getitem__(self, k: int) -> np.ndarray:
        """Equivalent to get_matrix(k)."""
        return self.get_matrix(k)

    def __repr__(self) -> str:
        return (
            f"GjiveData("
            f"observations={self.n_observations}, "
            f"dimension={self.n}, "
            f"groups={self.n_groups})"
        )
    
    def __eq__(self, other: object) -> bool:
        """
        Check whether two GjiveData objects contain identical simulations.

        The storage path is ignored; only simulation contents are compared.
        """
        if not isinstance(other, GjiveData):
            return NotImplemented

        # Compare standard arrays
        if not np.array_equal(self.A, other.A):
            return False

        if not np.array_equal(self.U, other.U):
            return False

        if not np.array_equal(self.group_assignment, other.group_assignment):
            return False

        # Compare object arrays / collections
        if len(self.Uf) != len(other.Uf):
            return False

        for uf_self, uf_other in zip(self.Uf, other.Uf):
            if not np.array_equal(uf_self, uf_other):
                return False

        if len(self.Uk) != len(other.Uk):
            return False

        for uk_self, uk_other in zip(self.Uk, other.Uk):
            if not np.array_equal(uk_self, uk_other):
                return False

        # Compare metadata dictionaries
        if self.metadata != other.metadata:
            return False

        return True