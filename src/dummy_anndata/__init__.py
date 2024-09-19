from importlib.metadata import version

from .generate_vector import generate_vector, vector_generators
from .generate_matrix import generate_matrix, matrix_generators
from .generate_dict import generate_scalar, generate_dict, scalar_generators
from .generate_dataframe import generate_dataframe
from .generate_dataset import generate_dataset

__all__ = [
    "generate_vector",
    "generate_matrix",
    "generate_scalar",
    "generate_dict",
    "generate_dataframe",
    "generate_dataset",
    "vector_generators",
    "matrix_generators",
    "scalar_generators"
]

__version__ = version("dummy-anndata")
