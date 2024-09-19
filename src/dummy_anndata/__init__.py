from importlib.metadata import version

from .generate_dataframe import generate_dataframe
from .generate_dataset import generate_dataset
from .generate_dict import generate_scalar, generate_type, generate_dict
from .generate_matrix import generate_matrix
from .generate_vector import generate_vector

__version__ = version("dummy-anndata")
