import numpy as np
import scipy as sp


def float_mtx(n_obs, n_vars, NAs=False):
    # add 0.5 to easily spot conversion issues
    mtx = np.arange(n_obs * n_vars, dtype=float).reshape(n_obs, n_vars) + 0.5
    if NAs:  # numpy matrices do no support pd.NA
        mtx[0, 0] = np.nan
    return mtx

def int_mtx(n_obs, n_vars):
    mtx = np.arange(n_obs * n_vars).reshape(n_obs, n_vars)
    return mtx

def float_mtx_nd(nr_values, dimensions, NAs=False):
    # add 0.5 to easily spot conversion issues
    mtx = np.arange(nr_values, dtype=float).reshape(dimensions) + 0.5
    if NAs:  # numpy matrices do no support pd.NA
        mtx[0, 0] = np.nan
    return mtx

def int_mtx_nd(nr_values, dimensions, NAs=False):
    mtx = np.arange(nr_values).reshape(dimensions)
    if NAs:  # numpy matrices do no support pd.NA
        mtx[0, 0] = np.nan
    return mtx

def float_mtx_sparse_nd(nr_values, dimensions, row_major=True, NAs=False):
    mtx = float_mtx_nd(nr_values, dimensions, NAs)
    if row_major:
        return sp.sparse.csr_matrix(mtx)
    else:
        return sp.sparse.csc_matrix(mtx)


# Possible matrix generators
# integer matrices do not support NAs in Python
matrix_generators = {
    "float_matrix": lambda n_obs, n_vars: float_mtx(n_obs, n_vars),
    "float_matrix_nas": lambda n_obs, n_vars: float_mtx(n_obs, n_vars, NAs=True),
    "float_csparse": lambda n_obs, n_vars: sp.sparse.csc_matrix(float_mtx(n_obs, n_vars)),
    "float_csparse_nas": lambda n_obs, n_vars: sp.sparse.csc_matrix(float_mtx(n_obs, n_vars, NAs=True)),
    "float_rsparse": lambda n_obs, n_vars: sp.sparse.csr_matrix(float_mtx(n_obs, n_vars)),
    "float_rsparse_nas": lambda n_obs, n_vars: sp.sparse.csr_matrix(float_mtx(n_obs, n_vars, NAs=True)),
    "integer_matrix": lambda n_obs, n_vars: int_mtx(n_obs, n_vars),
    "integer_csparse": lambda n_obs, n_vars: sp.sparse.csc_matrix(int_mtx(n_obs, n_vars)),
    "integer_rsparse": lambda n_obs, n_vars: sp.sparse.csr_matrix(int_mtx(n_obs, n_vars)),
    "float_matrix_3d": lambda n_obs, n_vars: float_mtx_nd(n_obs * n_vars * 3, (n_obs, n_vars, 3)),
    "integer_matrix_3d": lambda n_obs, n_vars: int_mtx_nd(n_obs * n_vars * 3, (n_obs, n_vars, 3)),
}

def string_matrix_nd(nr_values, dimensions):
    return np.array(['a' for _ in range(nr_values)]).reshape(dimensions)

def bool_matrix_nd(nr_values, dimensions):
    return np.array([True if i % 2 else False for i in range(nr_values)]).reshape(dimensions)

extra_uns_matrix_generators = {
    "string_matrix": lambda n_obs, n_vars: np.array(['a' for _ in range(n_obs * n_vars)]).reshape(n_obs, n_vars),
    "bool_matrix": lambda n_obs, n_vars: np.array([True for _ in range(n_obs * n_vars)]).reshape(n_obs, n_vars),
    "string_matrix_3d": lambda n_obs, n_vars: string_matrix_nd(n_obs * n_vars * 3, (n_obs, n_vars, 3)),
    "bool_matrix_3d": lambda n_obs, n_vars: bool_matrix_nd(n_obs * n_vars * 3, (n_obs, n_vars, 3)),

generated_matrix_types = np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix

def generate_matrix(n_obs: int, n_vars: int, matrix_type: str) -> generated_matrix_types:
    """
    Generate a matrix of given dimensions and type.

    Parameters
    ----------
        n_obs (int): The number of observations (rows) in the matrix.
        n_vars (int): The number of variables (columns) in the matrix.
        matrix_type (str): The type of matrix to generate.

    Returns
    -------
        np.ndarray | sp.sparse.csc_matrix | sp.sparse.csr_matrix:
        The generated matrix.

    Raises
    ------
        AssertionError: If the matrix_type is unknown.

    """
    assert matrix_type in matrix_generators.keys() or matrix_type in extra_uns_matrix_generators.keys(), f"Unknown matrix type: {matrix_type}"

    if matrix_type in matrix_generators:
        return matrix_generators[matrix_type](n_obs, n_vars)
    else:
        return extra_uns_matrix_generators[matrix_type](n_obs, n_vars)

