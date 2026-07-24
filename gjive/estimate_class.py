from pathlib import Path
from typing import Any
from dataclasses import dataclass
import json
import numpy as np


@dataclass
class GjiveEstimate:
    U: np.ndarray
    Uf: np.ndarray
    Uk: np.ndarray
    V: np.ndarray
    W: np.ndarray
    X: np.ndarray
    metadata: dict[str, Any]


    def get_U(self) -> np.ndarray:
        """Return the estimated joint subspace."""
        return self.U

    def get_Uf(self, group: int) -> np.ndarray:
        """Return the estimated group-specific subspace."""
        if not 0 <= group < len(self.Uf):
            raise IndexError(f"Invalid group index {group}.")
        return self.Uf[group]

    def get_all_Uf(self) -> np.ndarray:
        """Return all estimated group-specific subspaces."""
        return self.Uf

    def get_Uk(self, index: int) -> np.ndarray:
        """Return the estimated individual subspace for one matrix."""
        if not 0 <= index < len(self.Uk):
            raise IndexError(f"Invalid matrix index {index}.")
        return self.Uk[index]

    def get_all_Uk(self) -> np.ndarray:
        """Return all estimated individual subspaces."""
        return self.Uk

    def get_V(self, index: int) -> np.ndarray:
        """Return the joint loadings for one matrix."""
        if not 0 <= index < len(self.V):
            raise IndexError(f"Invalid matrix index {index}.")
        return self.V[index]

    def get_W(self, index: int) -> np.ndarray:
        """Return the group-specific loadings for one matrix."""
        if not 0 <= index < len(self.W):
            raise IndexError(f"Invalid matrix index {index}.")
        return self.W[index]

    def get_X(self, index: int) -> np.ndarray:
        """Return the individual loadings for one matrix."""
        if not 0 <= index < len(self.X):
            raise IndexError(f"Invalid matrix index {index}.")
        return self.X[index]


    "WILL HAVE TO UPDATE THE CODEBASE TO TAKE THIS FUNCTION"
    @classmethod
    def from_path(cls, path: Path | str) -> "GjiveEstimate":
        path = Path(path)

        npz_path = path / "estimate.npz"
        metadata_path = path / "metadata.json"

        if not npz_path.exists():
            raise FileNotFoundError(f"Could not find '{npz_path}'")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Could not find '{metadata_path}'")

        with np.load(npz_path, allow_pickle=True) as f:
            U = f["U"]
            Uf = f["Uf"]
            Uk = f["Uk"]
            V = f["V"]
            W = f["W"]
            X = f["X"]

        with metadata_path.open("r") as f:
            metadata = json.load(f)

        return cls(
            U=U,
            Uf=Uf,
            Uk=Uk,
            V=V,
            W=W,
            X=X,
            metadata=metadata,
        )

    