import dummy_anndata


def test_package_has_version():
    assert dummy_anndata.__version__ is not None


# This test test whether or not all the functions in the package
# work.
def test_generating_dataset(tmp_path):
    dummy = dummy_anndata.generate_dataset()
    filename = tmp_path / "dummy.h5ad"
    dummy.write_h5ad(filename)


def test_empty_uns():
    dummy = dummy_anndata.generate_dataset(uns_types=[], nested_uns_types=[])

    assert dummy.uns == {}


def test_nested_uns():
    dummy = dummy_anndata.generate_dataset(uns_types=[])

    assert "nested" in dummy.uns and dummy.uns["nested"] == {}

def test_nested_uns_types():
    dummy = dummy_anndata.generate_dataset(uns_types=[])

    assert "nested" not in dummy.uns
