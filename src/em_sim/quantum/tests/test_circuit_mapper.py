"""Unit tests for Maxwell circuit mapper."""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from em_sim.quantum.circuit_mapper import MaxwellCircuitMapper, CircuitConfig
from em_sim.quantum.hardware_config import HardwareConfig


@pytest.fixture
def circuit_mapper():
    """Create circuit mapper fixture."""
    config = CircuitConfig()
    return MaxwellCircuitMapper(config)


def test_cavity_mode_mapping(circuit_mapper):
    """Test cavity mode mapping with PEC boundaries."""
    dimensions = [1.0, 1.0, 1.0]
    boundary_conditions = {"type": "PEC"}
    
    circuit = circuit_mapper.map_cavity_modes(dimensions, boundary_conditions)
    
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == circuit_mapper.config.qubit_count


def test_waveguide_mode_mapping(circuit_mapper):
    """Test waveguide mode mapping with material properties."""
    cross_section = [0.5, 0.5]
    material_properties = {
        "epsilon": 2.1,
        "mu": 1.0
    }
    
    circuit = circuit_mapper.map_waveguide_modes(cross_section, material_properties)
    
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == circuit_mapper.config.qubit_count


def test_topological_boundary_mapping(circuit_mapper):
    """Test topological boundary operator mapping."""
    surface_geometry = np.random.rand(5, 2)
    correlation_map = np.array([[0, 1], [1, 2], [2, 3]])
    
    circuit = circuit_mapper.map_topological_boundary(surface_geometry, correlation_map)
    
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == circuit_mapper.config.qubit_count


def test_hardware_optimization(circuit_mapper):
    """Test hardware-specific circuit optimization."""
    # Create test circuit
    dimensions = [1.0, 1.0, 1.0]
    boundary_conditions = {"type": "PEC"}
    circuit = circuit_mapper.map_cavity_modes(dimensions, boundary_conditions)
    
    # Test superconducting optimization
    hw_config = HardwareConfig.create_superconducting()
    optimized = circuit_mapper._hardware_optimize(circuit, hw_config)
    assert isinstance(optimized, QuantumCircuit)
    
    # Test trapped ion optimization
    hw_config = HardwareConfig.create_trapped_ion()
    optimized = circuit_mapper._hardware_optimize(circuit, hw_config)
    assert isinstance(optimized, QuantumCircuit)
