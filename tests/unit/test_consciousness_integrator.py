"""Unit tests for consciousness integration."""

import pytest
import torch
from em_sim.core.consciousness_integrator import ConsciousnessIntegrator

@pytest.fixture
def integrator():
    """Create ConsciousnessIntegrator instance for testing."""
    return ConsciousnessIntegrator(field_strength=1.0, coherence_time=1e-3)

def test_coupling_field_shape(integrator):
    """Test shape of consciousness coupling field."""
    em_state = {
        "E": torch.zeros((32, 32, 32, 3)),
        "B": torch.zeros((32, 32, 32, 3))
    }
    coupling = integrator.compute_coupling_field(em_state)
    assert coupling.shape == (32, 32, 32, 3)

def test_decoherence_rate(integrator):
    """Test quantum decoherence rate calculation."""
    rate = integrator.get_decoherence_rate()
    assert rate > 0
    assert isinstance(rate, torch.Tensor)

def test_neural_modulation(integrator):
    """Test neural activity modulation of consciousness field."""
    neural_state = torch.rand((32, 32, 32))
    modulation = integrator._neural_modulation(neural_state)
    assert torch.all(modulation >= 0) and torch.all(modulation <= 1)

def test_parameter_update(integrator):
    """Test updating consciousness parameters."""
    old_strength = integrator.field_strength
    old_coherence = integrator.coherence_time
    
    integrator.update_parameters(field_strength=2.0, coherence_time=2e-3)
    
    assert integrator.field_strength == 2.0
    assert integrator.coherence_time == 2e-3
    assert integrator.field_strength != old_strength
    assert integrator.coherence_time != old_coherence

def test_coupling_field_strength(integrator):
    """Test consciousness field strength scaling."""
    em_state = {
        "E": torch.ones((32, 32, 32, 3)),
        "B": torch.ones((32, 32, 32, 3))
    }
    
    coupling_1 = integrator.compute_coupling_field(em_state)
    
    integrator.update_parameters(field_strength=2.0)
    coupling_2 = integrator.compute_coupling_field(em_state)
    
    # Field strength should scale linearly
    assert torch.allclose(2 * coupling_1, coupling_2)

def test_coherence_time_effect(integrator):
    """Test effect of coherence time on coupling field."""
    em_state = {
        "E": torch.ones((32, 32, 32, 3)),
        "B": torch.ones((32, 32, 32, 3))
    }
    
    coupling_1 = integrator.compute_coupling_field(em_state)
    
    # Shorter coherence time should lead to faster decoherence
    integrator.update_parameters(coherence_time=1e-4)
    coupling_2 = integrator.compute_coupling_field(em_state)
    
    assert torch.mean(coupling_2) < torch.mean(coupling_1)
