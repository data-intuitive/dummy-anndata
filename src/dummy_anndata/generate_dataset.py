from collections.abc import Iterable

import anndata as ad

from .generate_dataframe import generate_dataframe
from .generate_dict import generate_dict, scalar_generators
from .generate_matrix import matrix_generators
from .generate_vector import vector_generators


def generate_dataset(
    n_obs: int = 10,
    n_vars: int = 20,
    x_type: str | None = None,
    layer_types: list[str] | None = None,
    obs_types: list[str] | None = None,
    var_types: list[str] | None = None,
    obsm_types: list[str] | None = None,
    varm_types: list[str] | None = None,
    obsp_types: list[str] | None = None,
    varp_types: list[str] | None = None,
    uns_types: list[str] | None = None,
    nested_uns_types: list[str] | None = None,
) -> ad.AnnData:
    """
    Generate a synthetic AnnData dataset with specified dimensions and data types.

    Parameters
    ----------
    n_obs : int, optional (default=10)
        Number of observations (cells).
    n_vars : int, optional (default=20)
        Number of variables (genes).
    x_type : str, optional
        Type of matrix to generate for the main data matrix `X`. Must be a key in `matrix_generators`.
    layer_types : list of str, optional
        Types of matrices to generate for layers. Each type must be a key in `matrix_generators`.
    obs_types : list of str, optional
        Types of vectors to generate for `obs`. Each type must be a key in `vector_generators`.
    var_types : list of str, optional
        Types of vectors to generate for `var`. Each type must be a key in `vector_generators`.
    obsm_types : list of str, optional
        Types of matrices or vectors to generate for `obsm`. Each type must be a key in `matrix_generators` or `vector_generators`.
    varm_types : list of str, optional
        Types of matrices or vectors to generate for `varm`. Each type must be a key in `matrix_generators` or `vector_generators`,
        or should be a key in `vector_generators` prepended by `df_`, and will be used in the generation of a dataframe with the
        corresponding vector_generators.
    obsp_types : list of str, optional
        Types of matrices to generate for `obsp`. Each type must be a key in `matrix_generators`.
    varp_types : list of str, optional
        Types of matrices to generate for `varp`. Each type must be a key in `matrix_generators`.
    uns_types : list of str, optional
        Types of data to generate for `uns`. Each type must be a key in `vector_generators`, `matrix_generators`, or `scalar_generators`.
    nested_uns_types : list of str, optional
        Types of data to generate for the nested `uns` dictionary. They will be a new dictionary at the key `nested`.
        Each type must be a key in `vector_generators`, `matrix_generators`, or `scalar_generators`.

    Returns
    -------
    ad.AnnData
        An AnnData object containing the generated dataset with the specified dimensions and data types.

    Raises
    ------
    AssertionError
        If any of the specified types are not recognized by the corresponding generator dictionaries.
    """
    assert x_type is None or x_type in matrix_generators, f"Unknown matrix type: {x_type}"

    check_iterable_types(layer_types, "layer_types")
    check_iterable_types(obs_types, "obs_types")
    check_iterable_types(var_types, "var_types")
    check_iterable_types(obsm_types, "obsm_types")
    check_iterable_types(varm_types, "varm_types")
    check_iterable_types(obsp_types, "obsp_types")
    check_iterable_types(varp_types, "varp_types")
    check_iterable_types(uns_types, "uns_types")
    check_iterable_types(nested_uns_types, "nested_uns_types")

    assert layer_types is None or all(t in matrix_generators.keys() for t in layer_types), "Unknown layer type"
    assert obs_types is None or all(t in vector_generators.keys() for t in obs_types), "Unknown obs type"
    assert var_types is None or all(t in vector_generators.keys() for t in var_types), "Unknown var type"
    assert obsm_types is None or all(
        t in matrix_generators.keys() or t in vector_generators.keys() or t[3:] in vector_generators and t[:3] == "df_" for t in obsm_types
    ), "Unknown obsm type"
    assert varm_types is None or all(
        t in matrix_generators.keys() or t in vector_generators.keys() or t[3:] in vector_generators and t[:3] == "df_" for t in varm_types
    ), "Unknown varm type"
    assert obsp_types is None or all(t in matrix_generators.keys() for t in obsp_types), "Unknown obsp type"
    assert varp_types is None or all(t in matrix_generators.keys() for t in varp_types), "Unknown varp type"
    # TODO uns types

    if layer_types is None:  # layer_types are all matrices
        layer_types = list(matrix_generators.keys())
    if obs_types is None:  # obs_types are all vectors
        obs_types = list(vector_generators.keys())
    if var_types is None:  # var_types are all vectors
        var_types = list(vector_generators.keys())
    if obsm_types is None:  # obsm_types are all matrices or vectors, except for categoricals and nullables
        vector_not_allowed = set(
            [
                "categorical",
                "categorical_ordered",
                "categorical_missing_values",
                "categorical_ordered_missing_values",
                "nullable_integer_array",
                "nullable_boolean_array",
            ]
        )
        obsm_types = list(set(matrix_generators.keys()) - vector_not_allowed) + [f"df_{t}" for t in vector_generators.keys()]
    if varm_types is None:  # varm_types are all matrices or vectors, except for categoricals and nullables
        vector_not_allowed = set(
            [
                "categorical",
                "categorical_ordered",
                "categorical_missing_values",
                "categorical_ordered_missing_values",
                "nullable_integer_array",
                "nullable_boolean_array",
            ]
        )
        varm_types = list(set(matrix_generators.keys()) - vector_not_allowed) + [f"df_{t}" for t in vector_generators.keys()]

    if obsp_types is None:  # obsp_types are all matrices
        obsp_types = list(matrix_generators.keys())
    if varp_types is None:  # varp_types are all matrices
        varp_types = list(matrix_generators.keys())
    if uns_types is None:
        uns_types = list(vector_generators.keys()) + list(matrix_generators.keys()) + list(scalar_generators.keys())
    if nested_uns_types is None:
        nested_uns_types = (
            list(vector_generators.keys()) + list(matrix_generators.keys()) + list(scalar_generators.keys())
        )

    X = None
    if x_type is not None:
        X = matrix_generators[x_type](n_obs, n_vars)
    layers = {t: matrix_generators[t](n_obs, n_vars) for t in layer_types}

    obs_names = [f"Cell{i:03d}" for i in range(n_obs)]
    var_names = [f"Gene{i:03d}" for i in range(n_vars)]

    obs = generate_dataframe(n_obs, obs_types)
    var = generate_dataframe(n_vars, var_types)
    obs.index = obs_names
    var.index = var_names

    obsm = {}
    for t in obsm_types:
        if t in matrix_generators.keys():
            obsm[t] = matrix_generators[t](n_obs, n_obs)
        elif t in vector_generators.keys():
            obsm[t] = vector_generators[t](n_obs)
    df_obs_types = [t[3:] for t in obsm_types if t[:3] == "df_"]
    obsm["dataframe"] = generate_dataframe(n_obs, df_obs_types)

    varm = {}
    for t in varm_types:
        if t in matrix_generators.keys():
            varm[t] = matrix_generators[t](n_vars, n_vars)
        elif t in vector_generators.keys():
            varm[t] = vector_generators[t](n_vars)
    df_varm_types = [t[3:] for t in varm_types if t[:3] == "df_"]
    varm["dataframe"] = generate_dataframe(n_vars, df_varm_types)

    obsp = {t: matrix_generators[t](n_obs, n_obs) for t in obsp_types}
    varp = {t: matrix_generators[t](n_vars, n_vars) for t in varp_types}

    uns = generate_dict(n_obs, n_vars, uns_types, nested_uns_types)

    return ad.AnnData(
        X=X,
        layers=layers,
        obs=obs,
        var=var,
        obsm=obsm,
        varm=varm,
        obsp=obsp,
        varp=varp,
        uns=uns,
    )


def check_iterable_types(iterable_types, name):
    assert iterable_types is None or (
        isinstance(iterable_types, Iterable) and not isinstance(iterable_types, str)
    ), f"{name} should be a non-string iterable type"
