# gjive

A Python package for generating and evaluating the group, joint, and individual variance explained (GJIVE) model.

## Installation

Install the project in editable mode from the repository root:

```bash
python -m pip install -e .
```

## Usage

Import from the package anywhere in the project:

```python
from gjive.estimate import GjiveData
from gjive.generate import generate_simulation_data
from gjive.specifications import SimulationSpec
```

## Package layout

Only the `gjive` directory is installed as a Python package.

Do not package asset or application directories such as `app`, `scripts`, `data`, `estimates`, `plots`, `figures`, or `results`.
