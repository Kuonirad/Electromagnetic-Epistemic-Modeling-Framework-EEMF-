"""Validation utilities for quantum circuits."""

from typing import Dict, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, process_fidelity

def validate_unitarity(circuit: QuantumCircuit) -> bool:
    """Verify unitarity preservation of quantum circuit.
    
    Args:
        circuit: Quantum circuit to validate
        
    Returns:
        True if circuit preserves unitarity within tolerance
    """
    # Compute process matrix
    process_mat = circuit.to_operator()
    
    # Check if U†U ≈ I
    identity = np.eye(2**circuit.num_qubits)
    product = process_mat.adjoint() @ process_mat
    
    return np.allclose(product, identity, atol=1e-6)

def validate_causality(circuit: QuantumCircuit) -> bool:
    """Verify causality preservation in circuit.
    
    Args:
        circuit: Quantum circuit to validate
        
    Returns:
        True if circuit preserves causality
    """
    # Check temporal ordering of operations
    for instruction in circuit.data:
        if hasattr(instruction.operation, 'duration'):
            if instruction.operation.duration < 0:
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
    if 'eigenvalues' in results:
        error_metrics['eigenvalue_std'] = np.std(results['eigenvalues'])
        
    # Compute V-score (variance-to-energy ratio)
    if 'energy' in results and 'variance' in results:
        v_score = results['variance'] / abs(results['energy'])
        error_metrics['v_score'] = v_score
        
    return error_metrics
