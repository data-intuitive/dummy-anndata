import numpy as np

from .generate_matrix import matrix_generators
from .generate_vector import vector_generators

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


def generate_dict(n_rows, n_cols, types=None, nested_uns_types=None):
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
        data["nested"] = generate_dict(n_rows, n_cols, types = nested_uns_types, nested_uns_types=[])

    return data
