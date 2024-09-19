# dummy-anndata

[![Tests][badge-tests]][link-tests]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/data-intuitive/dummy-anndata/test.yaml?branch=main
[link-tests]: https://github.com/LouiseDck/dummy-anndata/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/dummy-anndata

Allows generating dummy anndata objects, which can be useful for testing purposes.

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

```bash
pip install git+https://github.com/data-intuitive/dummy-anndata.git@main
```

## Example usage
```{python}
import anndata as ad
import dummy_anndata as da

dummy_anndata_dataset = da.generate_dataset(n_obs=100, n_vars= 50)

```


## Contact

If you found a bug, please use the [issue tracker][issue-tracker].


[issue-tracker]: https://github.com/data-intuitive/dummy-anndata/issues

