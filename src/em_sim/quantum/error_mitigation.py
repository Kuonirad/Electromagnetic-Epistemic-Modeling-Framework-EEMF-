"""Zero-Noise Extrapolation utilities for error mitigation."""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
from qiskit.utils.mitigation import zne


def apply_zne(circuit: QuantumCircuit, config: Dict) -> Tuple[float, float]:
    """Apply Zero-Noise Extrapolation and return mitigated value with error bounds.
    
    Args:
        circuit: Circuit to apply ZNE to
        config: ZNE configuration dictionary
        
    Returns:
        Tuple of (mitigated_value, error_bound)
    """
    scale_factors = config['scale_factors']
    noise_amplifier = zne.scaling.LocalFoldingAmplifier()
    extrapolator = zne.extrapolators.LinearExtrapolator()
    
    # Scale circuits
    scaled_circuits = [
        noise_amplifier.fold(circuit, scale)
        for scale in scale_factors
    ]
    
    # Execute scaled circuits
    backend = AerSimulator()
    results = [
        execute(
            circ,
            backend,
            shots=1000,
            optimization_level=0  # Disable optimizations to preserve noise scaling
        ).result().get_counts()
        for circ in scaled_circuits
    ]
    
    # Convert counts to expectation values
    expectations = [
        sum(k.count('1') * v for k, v in result.items()) / sum(result.values())
        for result in results
    ]
    
    # Extrapolate and compute error bounds
    mitigated = extrapolator.extrapolate(scale_factors, expectations)
    error_bound = compute_error_bounds(scale_factors, expectations)
    
    return mitigated, error_bound


def compute_error_bounds(scales: List[float], results: List[float]) -> float:
    """Compute error bounds using linear regression statistics.
    
    Args:
        scales: List of noise scaling factors
        results: List of measured expectation values
        
    Returns:
        95% confidence interval for the zero-noise extrapolation
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(scales, results)
    residuals = np.array(results) - (intercept + slope * np.array(scales))
    return 1.96 * np.std(residuals) / np.sqrt(len(scales))
