"""Unit tests for Maxwell eigenvalue solver."""

import numpy as np
import pytest

from em_sim.quantum.circuit_mapper import CircuitConfig, MaxwellCircuitMapper
from em_sim.quantum.eigenvalue_solver import MaxwellEigenvalueSolver


@pytest.fixture
def eigenvalue_solver():
    """Create eigenvalue solver fixture."""
    return MaxwellEigenvalueSolver(shots=100)


@pytest.fixture
def test_circuit():
    """Create test quantum circuit."""
    config = CircuitConfig(qubit_count=4)
    mapper = MaxwellCircuitMapper(config)
    dimensions = [1.0, 1.0, 1.0]
    boundary_conditions = {"type": "PEC"}
    return mapper.map_cavity_modes(dimensions, boundary_conditions)


def test_solver_initialization(eigenvalue_solver):
    """Test eigenvalue solver initialization."""
    assert eigenvalue_solver.shots == 100
    assert eigenvalue_solver.optimization_level == 3


def test_solve_cavity_modes(eigenvalue_solver, test_circuit):
    """Test solving cavity mode eigenvalue problem."""
    eigenvalues, eigenvectors = eigenvalue_solver.solve(test_circuit)

    assert isinstance(eigenvalues, np.ndarray)
    assert isinstance(eigenvectors, np.ndarray)
    assert len(eigenvalues.shape) == 1
    assert eigenvectors.shape[0] == eigenvalues.shape[0]


def test_solver_optimization_level(eigenvalue_solver):
    """Test solver optimization level configuration."""
    solver = MaxwellEigenvalueSolver(optimization_level=2)
    assert solver.optimization_level == 2
