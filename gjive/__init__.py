"""Public package interface for the GJIVE package."""

from .dataset import GjiveData
from .estimate import U_ind, U_group, U_joint, estimate_data, estimate_loadings
from .estimate_class import GjiveEstimate
from .estimate_spec import EstimateSpec
from .generate import generate_group, generate_ind, generate_joint, generate_matrix
from .utils import (
    check_orthogonal,
    check_orthonormal,
    generate_random_matrix,
    matrix_neg_half,
    orthonormalize,
    to_object_array,
    M_joint
)

__all__ = [
    "GjiveData",
    "GjiveEstimate",
    "EstimateSpec",
    "estimate_data",
    "estimate_loadings",
    "U_joint",
    "U_group",
    "U_ind",
    "generate_group",
    "generate_ind",
    "generate_joint",
    "generate_matrix",
    "check_orthogonal",
    "check_orthonormal",
    "generate_random_matrix",
    "matrix_neg_half",
    "orthonormalize",
    "to_object_array",
    "M_joint"
]
