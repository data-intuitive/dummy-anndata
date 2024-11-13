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
}

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
    assert matrix_type in matrix_generators, f"Unknown matrix type: {matrix_type}"

    return matrix_generators[matrix_type](n_obs, n_vars)
