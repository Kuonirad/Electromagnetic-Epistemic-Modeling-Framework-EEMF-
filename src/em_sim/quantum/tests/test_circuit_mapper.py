"""Unit tests for Maxwell circuit mapper."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

from em_sim.quantum.circuit_mapper import CircuitConfig, MaxwellCircuitMapper
from em_sim.quantum.error_mitigation import compute_error_bounds
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
    material_properties = {"epsilon": 2.1, "mu": 1.0}

    circuit = circuit_mapper.map_waveguide_modes(
        cross_section, material_properties
    )

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == circuit_mapper.config.qubit_count


def test_topological_boundary_mapping(circuit_mapper):
    """Test topological boundary operator mapping."""
    surface_geometry = np.random.rand(5, 2)
    correlation_map = np.array([[0, 1], [1, 2], [2, 3]])

    circuit = circuit_mapper.map_topological_boundary(
        surface_geometry, correlation_map
    )

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == circuit_mapper.config.qubit_count


def test_hardware_optimization():
    """Test hardware-specific circuit optimization."""
    # Create test circuit with 6 qubits to match coupling map
    config = CircuitConfig(qubit_count=6)
    mapper = MaxwellCircuitMapper(config)
    dimensions = [1.0, 1.0, 1.0]
    boundary_conditions = {"type": "PEC"}
    circuit = mapper.map_cavity_modes(dimensions, boundary_conditions)

    # Test superconducting optimization
    hw_config = HardwareConfig.create_superconducting()
    optimized = mapper._hardware_optimize(circuit, hw_config)
    assert isinstance(optimized, QuantumCircuit)

    # Test trapped ion optimization
    hw_config = HardwareConfig.create_trapped_ion()
    optimized = mapper._hardware_optimize(circuit, hw_config)
    assert isinstance(optimized, QuantumCircuit)


def test_zne_error_mitigation():
    """Test Zero-Noise Extrapolation error mitigation."""
    config = CircuitConfig(
        qubit_count=6,  # Match coupling map size
        error_mitigation=True,
        zne_config={"scale_factors": [1, 2, 3]},
    )
    mapper = MaxwellCircuitMapper(config)

    # Create test circuit
    circuit = QuantumCircuit(6)  # Match coupling map size
    circuit.h(0)
    circuit.cx(0, 1)

    # Apply optimization with ZNE
    _ = mapper._hardware_optimize(
        circuit, HardwareConfig.create_superconducting()
    )

    # Verify error metrics
    assert hasattr(mapper, "_error_metrics")
    assert "mitigated_value" in mapper._error_metrics
    assert "error_bound" in mapper._error_metrics
    assert mapper._error_metrics["error_bound"] > 0


def test_error_bound_calculation():
    """Test error bound computation."""
    scales = [1, 2, 3]
    results = [0.95, 0.85, 0.75]
    error_bound = compute_error_bounds(scales, results)
    assert error_bound > 0
    assert error_bound < 1.0  # Reasonable bound for test data
