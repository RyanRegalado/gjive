from pathlib import Path
from typing import Any
import json
from dataclasses import dataclass
import numpy as np

@dataclass
class GjiveEstimate:

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

        npz_path = self.path / "estimate.npz"
        metadata_path = self.path / "metadata.json"

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Could not find '{npz_path}'"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Could not find '{metadata_path}'"
            )

        with np.load(npz_path, allow_pickle=True) as f:
            self.U = f["U"]
            self.Uf = f["Uf"]
            self.Uk = f["Uk"]
            self.V = f["V"]
            self.W = f["W"]
            self.X = f["X"]

        with metadata_path.open("r") as f:
            self.metadata: dict[str, Any] = json.load(f)