import dummy_anndata
import pytest

def test_package_has_version():
    assert dummy_anndata.__version__ is not None


# This test test whether or not all the functions in the package
# work.
def test_generating_dataset(tmp_path):
    dummy = dummy_anndata.generate_dataset()
    filename = tmp_path / "dummy.h5ad"
    dummy.write_h5ad(filename)


def test_uns():
    dummy_empty = dummy_anndata.generate_dataset(uns_types=[], nested_uns_types=[])
    assert dummy_empty.uns == {}

    dummy_nested = dummy_anndata.generate_dataset(uns_types=[])
    assert "nested" in dummy_nested.uns and dummy_nested.uns["nested"] != {}

    dummy_no_nested = dummy_anndata.generate_dataset(nested_uns_types=[])
    assert "nested" not in dummy_no_nested.uns


def test_empty_x():
    dummy = dummy_anndata.generate_dataset()
    assert dummy.X is None


def test_forbidden_obsm_varm_types():
    with pytest.raises(AssertionError) as ae:
        dummy = dummy_anndata.generate_dataset(obsm_types=["categorical"])

    assert ae.match("Forbidden obsm type")

    with pytest.raises(AssertionError) as ae:
        dummy = dummy_anndata.generate_dataset(varm_types=["categorical"])

    assert ae.match("Forbidden varm type")
