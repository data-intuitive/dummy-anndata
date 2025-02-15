import numpy as np
import pandas as pd


def nullable_integer_array(n):
    assert n > 0, "an integer array must be at least one value"
    nullable_array = [i for i in range(n)]
    # np.nan, pd.NA and None should all end up as null values, masked in the h5ad file
    nullable_array[0] = np.nan
    return pd.array(nullable_array, dtype="Int64")


def nullable_boolean_array(n):
    assert n > 0, "a boolean array must be at least one value"
    nullable_array = pd.array([[True, False][i % 2] for i in range(n)], dtype="boolean")
    # np.nan, pd.NA and None should all end up as null values, masked in the h5ad file
    nullable_array[0] = pd.NA
    return nullable_array


def missing_values_categorical(n, ordered=True):
    assert n > 0, "a categorical must be at least one value"
    missing_values = pd.Categorical(
        [["Value1", "Value2"][i % 2] for i in range(n)],
        categories=["Value1", "Value2"],
        ordered=ordered,
    )
    # They should all end up as code -1 in the h5ad file
    missing_values[0] = np.nan
    return missing_values


vector_generators = {
    "categorical": lambda n: pd.Categorical([["Value1", "Value2"][i % 2] for i in range(n)]),
    "categorical_ordered": lambda n: pd.Categorical([["Value1", "Value2"][i % 2] for i in range(n)], ordered=True),
    "categorical_missing_values": lambda n: missing_values_categorical(n, ordered=False),
    "categorical_ordered_missing_values": lambda n: missing_values_categorical(n, ordered=True),
    "string_array": lambda n: np.array([f"value_{i}" for i in range(n)]),
    # should we also check a 1d sparse array? We should probably leave it for the matrix generation?
    "dense_array": lambda n: np.arange(n, dtype=float) + 0.5,
    "integer_array": lambda n: np.array([i for i in range(n)]),
    "nullable_integer_array": nullable_integer_array,
    "boolean_array": lambda n: np.array([[True, False][i % 2] for i in range(n)]),
    "nullable_boolean_array": nullable_boolean_array,
}

generated_vector_types = np.ndarray | pd.Categorical | pd.arrays.IntegerArray | pd.arrays.BooleanArray

def generate_vector(n: int, vector_type: str) -> generated_vector_types:
    """
    Generate a vector of a specified type and length.

    Parameters:
        n (int): The length of the vector to generate.
        vector_type (str): The type of vector to generate. Must be one of the keys in the `vector_generators` dictionary.

    Returns:
        np.ndarray | pd.Categorical | pd.arrays.IntegerArray | pd.arrays.BooleanArray:
        A vector of the specified type and length.

    Raises:
        AssertionError: If `vector_type` is not a valid key in `vector_generators`.
    """

    # check if vector_type is valid
    assert vector_type in vector_generators, f"Unknown vector type: {vector_type}"

    return vector_generators[vector_type](n)
