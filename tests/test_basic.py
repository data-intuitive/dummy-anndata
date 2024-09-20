import pytest

import dummy_anndata


def test_package_has_version():
    assert dummy_anndata.__version__ is not None


# This test test whether or not all the functions in the package
# work.
def test_generating_dataset(tmp_path):
    dummy = dummy_anndata.generate_dataset()
    filename = tmp_path / "dummy.h5ad"
    dummy.write_h5ad(filename)
