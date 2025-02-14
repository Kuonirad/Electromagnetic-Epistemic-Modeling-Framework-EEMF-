"""Unit tests for FDTD output functionality."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from em_sim.fdtd.output import create_metadata, save_results
from em_sim.fdtd.solver import FDTDParameters


@pytest.fixture
def test_data():
    """Create test data for output tests."""
    Ex = np.linspace(-1, 1, 100)
    Hy = np.sin(Ex)
    data = np.column_stack((Ex, Hy))
    params = FDTDParameters(grid_size=100, total_time=500)
    metadata = create_metadata(params, params.dx, params.dt)
    return data, metadata


def test_save_hdf5(test_data, tmp_path):
    """Test HDF5 output format with compression."""
    data, metadata = test_data
    output_file = save_results(
        data, metadata, output_format="hdf5", output_dir=tmp_path
    )

    assert Path(output_file).exists()
    with h5py.File(output_file, "r") as hf:
        # Verify data
        assert np.allclose(data[:, 0], hf["Ex"][:])
        assert np.allclose(data[:, 1], hf["Hy"][:])
        # Verify metadata
        assert str(metadata["grid_size"]) == hf.attrs["grid_size"]
        assert str(metadata["time_steps"]) == hf.attrs["time_steps"]


def test_save_npz(test_data, tmp_path):
    """Test NPZ output format for parallel I/O."""
    data, metadata = test_data
    output_file = save_results(
        data, metadata, output_format="npz", output_dir=tmp_path
    )

    assert Path(output_file).exists()
    loaded = np.load(output_file)
    assert np.allclose(data[:, 0], loaded["Ex"])
    assert np.allclose(data[:, 1], loaded["Hy"])
    assert isinstance(loaded["metadata"], np.ndarray)


def test_invalid_output_format(test_data, tmp_path):
    """Test error handling for invalid output format."""
    data, metadata = test_data
    with pytest.raises(ValueError, match="Unsupported output format"):
        save_results(
            data, metadata, output_format="invalid", output_dir=tmp_path
        )


def test_output_directory_creation(test_data, tmp_path):
    """Test automatic creation of output directory."""
    data, metadata = test_data
    output_dir = tmp_path / "nested" / "output"
    output_file = save_results(
        data, metadata, output_format="txt", output_dir=output_dir
    )

    assert Path(output_file).exists()
    assert output_dir.exists()


def test_precision_control(test_data, tmp_path):
    """Test precision control in text output."""
    data, metadata = test_data
    output_file = save_results(
        data, metadata, output_format="txt", precision=3, output_dir=tmp_path
    )

    loaded = np.loadtxt(output_file)
    # Check that values are rounded to 3 decimal places
    assert np.allclose(loaded, np.round(data, 3))
