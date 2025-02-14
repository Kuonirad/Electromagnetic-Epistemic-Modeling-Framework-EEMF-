"""Unit tests for the FDTD solver implementation."""

import numpy as np

from em_sim.fdtd.solver import FDTD1DSolver, FDTDParameters


def test_fdtd_parameters_default():
    """Test FDTDParameters initialization with default values."""
    params = FDTDParameters()
    assert params.grid_size == 200
    assert params.total_time == 1000
    assert params.dx == 1e-3
    assert params.c == 3e8
    assert params.dt == params.dx / (2 * params.c)


def test_fdtd_solver_initialization():
    """Test FDTD solver initialization."""
    solver = FDTD1DSolver()
    assert solver.Ex.shape == (200,)
    assert solver.Hy.shape == (199,)
    assert np.all(solver.Ex == 0)
    assert np.all(solver.Hy == 0)


def test_fdtd_field_updates():
    """Test that field updates maintain expected properties."""
    solver = FDTD1DSolver()

    # Initial energy should be zero
    initial_energy = np.sum(solver.Ex**2 + solver.Hy**2)
    assert initial_energy == 0

    # Run a few timesteps
    for _ in range(10):
        solver.update_fields()
        solver.apply_boundary_conditions()

    # Fields should still be zero without source
    assert np.allclose(solver.Ex, 0)
    assert np.allclose(solver.Hy, 0)


def test_gaussian_source_injection():
    """Test Gaussian pulse source injection."""
    solver = FDTD1DSolver()

    # Inject source at t=30 (peak of Gaussian)
    solver.inject_source(30, position=100)

    # Check if pulse was injected at correct position
    assert solver.Ex[100] > 0
    assert solver.Ex[0] == 0
    assert solver.Ex[-1] == 0


def test_pec_boundary_condition():
    """Test PEC boundary condition at left boundary."""
    solver = FDTD1DSolver()

    # Set non-zero field values
    solver.Ex[0] = 1.0
    solver.apply_boundary_conditions()

    # PEC should force Ex=0 at boundary
    assert solver.Ex[0] == 0.0


def test_field_data_export():
    """Test field data export functionality."""
    solver = FDTD1DSolver()
    Ex, Hy = solver.get_field_data()

    assert Ex.shape == (solver.params.grid_size,)
    assert Hy.shape == (solver.params.grid_size,)
    assert np.all(Ex == solver.Ex)
    assert np.all(Hy[:-1] == solver.Hy)
    assert Hy[-1] == 0  # Last Hy value should be padded with 0
