"""Unit tests for quantum-enabled field solver."""

import pytest
import torch

from em_sim.core.field_solver import EMFieldSolver
from em_sim.quantum.circuit_mapper import CircuitConfig
from em_sim.quantum.hardware_config import HardwareConfig


@pytest.fixture
def quantum_config():
    """Create quantum configuration fixture."""
    return {
        "circuit_config": CircuitConfig(qubit_count=4),
        "hardware_config": HardwareConfig.create_superconducting(),
    }


@pytest.fixture
def field_solver(quantum_config):
    """Create field solver fixture with quantum config."""
    return EMFieldSolver(grid_size=(4, 4, 4), quantum_config=quantum_config)


def test_quantum_solver_initialization(field_solver):
    """Test initialization with quantum configuration."""
    assert field_solver.quantum_config is not None
    assert isinstance(field_solver.circuit_mapper.config, CircuitConfig)
    assert isinstance(field_solver.hardware_config, HardwareConfig)


def test_quantum_solver_step(field_solver):
    """Test field evolution with quantum solver."""
    # Initial field state
    field_solver.E = torch.ones((4, 4, 4, 3))
    field_solver.B = torch.ones((4, 4, 4, 3))

    # Evolve with quantum solver
    result = field_solver.step(use_quantum_solver=True)

    assert "E" in result
    assert "B" in result
    assert result["E"].shape == (4, 4, 4, 3)
    assert result["B"].shape == (4, 4, 4, 3)


def test_quantum_solver_error_handling():
    """Test error handling when quantum solver requested without config."""
    solver = EMFieldSolver(grid_size=(4, 4, 4))

    with pytest.raises(
        ValueError,
        match="Quantum solver requested but no quantum config provided",
    ):
        solver.step(use_quantum_solver=True)


def test_quantum_classical_consistency(field_solver):
    """Test consistency between quantum and classical solvers."""
    # Initial field state
    field_solver.E = torch.ones((4, 4, 4, 3))
    field_solver.B = torch.ones((4, 4, 4, 3))

    # Evolve with both solvers
    classical_result = field_solver.step(use_quantum_solver=False)
    quantum_result = field_solver.step(use_quantum_solver=True)

    # Check energy conservation
    def compute_energy(E, B):
        return torch.sum(E**2 + B**2)

    classical_energy = compute_energy(
        classical_result["E"], classical_result["B"]
    )
    quantum_energy = compute_energy(quantum_result["E"], quantum_result["B"])

    # Energy difference should be small (ΔE < 0.01 Ha)
    assert torch.abs(classical_energy - quantum_energy) < 0.01
