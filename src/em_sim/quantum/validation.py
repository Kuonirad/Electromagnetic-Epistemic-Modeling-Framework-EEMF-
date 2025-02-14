"""Validation utilities for quantum circuits."""

from typing import Dict

import numpy as np
from qiskit import QuantumCircuit


def validate_unitarity(circuit: QuantumCircuit) -> bool:
    """Verify unitarity preservation of quantum circuit.

    Args:
        circuit: Quantum circuit to validate

    Returns:
        True if circuit preserves unitarity within tolerance
    """
    from qiskit.exceptions import QiskitError
    from qiskit.quantum_info import Operator

    try:
        # Operator() will fail for non-unitary circuits
        Operator(circuit)
        return True
    except QiskitError:
        return False

    return True


def validate_causality(circuit: QuantumCircuit) -> bool:
    """Verify causality preservation in circuit.

    Args:
        circuit: Quantum circuit to validate

    Returns:
        True if circuit preserves causality
    """
    # Check temporal ordering of operations
    for instruction in circuit.data:
        if hasattr(instruction.operation, "duration"):
            duration = instruction.operation.duration
            if duration is not None and duration < 0:
                return False
    return True


def compute_error_bounds(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute error bounds for eigenvalue results.

    Args:
        results: Dictionary containing eigenvalue computation results

    Returns:
        Dictionary with error metrics
    """
    error_metrics = {}

    # Compute statistical error
    if "eigenvalues" in results:
        error_metrics["eigenvalue_std"] = np.std(results["eigenvalues"])

    # Compute V-score (variance-to-energy ratio)
    if "energy" in results and "variance" in results:
        v_score = results["variance"] / abs(results["energy"])
        error_metrics["v_score"] = v_score

    return error_metrics


def validate_error_mitigation(
    circuit: QuantumCircuit, mitigated_value: float, error_bound: float
) -> bool:
    """Validate error mitigation results.

    Args:
        circuit: Original quantum circuit
        mitigated_value: Value after error mitigation
        error_bound: Computed error bound

    Returns:
        True if error mitigation results are valid
    """
    # Verify error bound is reasonable
    if error_bound <= 0 or error_bound >= 1.0:
        return False

    # Verify mitigated value is within physical bounds
    if abs(mitigated_value) > 1.0:
        return False

    return True
