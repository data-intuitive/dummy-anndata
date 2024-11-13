import numpy as np

from typing import Union

from .generate_matrix import matrix_generators, generated_matrix_types
from .generate_vector import vector_generators, generated_vector_types

scalar_generators = {
    "string": "version",
    "char": "a",
    "integer": 1,
    "float": 1.0,
    "boolean": True,
    "none": None,
    # "NA": pd.NA, cannot write to h5 group
    "nan": np.nan,
}

generated_scalar_types = Union[str, int, float, bool, None, np.float64]

def generate_scalar(scalar_type):
    if scalar_type[:7] == "scalar_":
        return vector_generators[scalar_type[7:]](1)
    return scalar_generators[scalar_type]


def generate_type(type, n_rows, n_cols):
    if type in scalar_generators or type[:7] == "scalar_":
        return generate_scalar(type)
    if type in vector_generators:
        return vector_generators[type](n_rows)
    if type in matrix_generators:
        return matrix_generators[type](n_rows, n_cols)
    return None

all_types = generated_scalar_types | generated_vector_types | generated_matrix_types
generated_dict_types = dict[str, all_types | dict[str, all_types]]

def generate_dict(
    n_rows: int, n_cols: int, types: list[str] | None = None, nested_uns_types: list[str] | None = None
) -> generated_dict_types:
    """
    Generates a dictionary with specified types of data.

    Parameters:
    n_rows (int): Number of rows for the generated data.
    n_cols (int): Number of columns for the generated data.
    types (list[str] | None): List of types to generate. If None, defaults to all available types.
    nested_uns_types (list[str] | None): List of types for nested 'uns' data. If None, defaults to all available types.

    Returns:
    A dictionary containing the generated data.
    """
    if types is None:  # types are all vectors and all matrices
        types = (
            list(scalar_generators.keys())
            + [f"scalar_{t}" for t in vector_generators.keys()]
            + list(vector_generators.keys())
            + list(matrix_generators.keys())
        )

    if nested_uns_types is None:
        nested_uns_types = (
            list(scalar_generators.keys())
            + [f"scalar_{t}" for t in vector_generators.keys()]
            + list(vector_generators.keys())
            + list(matrix_generators.keys())
        )

    data = {}
    if types:  # types is not empty
        data = {t: generate_type(t, n_rows, n_cols) for t in types}
    if nested_uns_types:
        data["nested"] = generate_dict(n_rows, n_cols, types=nested_uns_types, nested_uns_types=[])

    return data
