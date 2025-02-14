"""Unit tests for electromagnetic field solver."""

import pytest
import torch
import numpy as np
from em_sim.core.field_solver import EMFieldSolver

@pytest.fixture
def solver():
    """Create EMFieldSolver instance for testing."""
    return EMFieldSolver(grid_size=(32, 32, 32), dt=1e-6)

def test_field_initialization(solver):
    """Test proper initialization of EM fields."""
    assert solver.E.shape == (32, 32, 32, 3)
    assert solver.B.shape == (32, 32, 32, 3)
    assert torch.all(solver.E == 0)
    assert torch.all(solver.B == 0)

def test_step_without_consciousness(solver):
    """Test single time step evolution without consciousness coupling."""
    result = solver.step()
    assert "E" in result
    assert "B" in result
    assert not torch.all(result["E"] == 0)  # Fields should evolve
    assert not torch.all(result["B"] == 0)

def test_step_with_consciousness(solver):
    """Test evolution with consciousness field coupling."""
    consciousness = torch.ones((32, 32, 32, 3), device=solver.device)
    result = solver.step(consciousness)
    assert "E" in result
    assert "B" in result
    # Consciousness should affect E field more than B field
    assert torch.norm(result["E"]) > torch.norm(result["B"])

def test_energy_conservation(solver):
    """Test approximate energy conservation in field evolution."""
    initial_E = solver.E.clone()
    initial_B = solver.B.clone()
    
    # Set some initial field configuration
    solver.E[0,0,0,0] = 1.0
    solver.B[0,0,0,1] = 1.0
    
    initial_energy = torch.sum(solver.E**2 + solver.B**2)
    
    # Evolve for 10 steps
    for _ in range(10):
        solver.step()
    
    final_energy = torch.sum(solver.E**2 + solver.B**2)
    
    # Energy should be approximately conserved
    assert np.isclose(initial_energy.item(), final_energy.item(), rtol=1e-2)

def test_mpi_decomposition(solver):
    """Test MPI domain decomposition."""
    field = torch.ones((32, 32, 32, 3), device=solver.device)
    local_field = solver._decompose_domain(field)
    
    # Local field should have correct shape based on MPI rank
    expected_shape = (32 // solver.comm.size, 32, 32, 3)
    assert local_field.shape == expected_shape

def test_invalid_consciousness_field(solver):
    """Test handling of invalid consciousness field."""
    invalid_consciousness = torch.ones((10, 10, 10, 3))  # Wrong shape
    with pytest.raises(ValueError):
        solver.step(invalid_consciousness)
