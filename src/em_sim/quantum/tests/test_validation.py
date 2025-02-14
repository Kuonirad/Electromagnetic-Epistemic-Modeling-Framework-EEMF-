"""Unit tests for quantum circuit validation utilities."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Operator

from em_sim.quantum.validation import (compute_error_bounds,
                                       validate_causality, validate_unitarity)


@pytest.fixture
def test_circuit():
    """Create test quantum circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def test_validate_unitarity(test_circuit):
    """Test unitarity validation."""
    assert validate_unitarity(test_circuit)

    # Test non-unitary circuit
    non_unitary = QuantumCircuit(1)
    non_unitary.reset(0)
    assert not validate_unitarity(non_unitary)


def test_validate_causality(test_circuit):
    """Test causality validation."""
    assert validate_causality(test_circuit)


def test_compute_error_bounds():
    """Test error bound computation."""
    results = {
        "eigenvalues": np.array([1.0, 1.1, 0.9]),
        "energy": -2.0,
        "variance": 0.1,
    }

    error_metrics = compute_error_bounds(results)

    assert "eigenvalue_std" in error_metrics
    assert "v_score" in error_metrics
    assert error_metrics["v_score"] == 0.1 / 2.0


def test_error_bounds_missing_data():
    """Test error bound computation with missing data."""
    results = {"eigenvalues": np.array([1.0, 1.1, 0.9])}
    error_metrics = compute_error_bounds(results)

    assert "eigenvalue_std" in error_metrics
    assert "v_score" not in error_metrics
