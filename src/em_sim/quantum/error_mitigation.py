"""Zero-Noise Extrapolation utilities for error mitigation."""

from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError
from qiskit.primitives import Sampler
from scipy import stats


def apply_zne(
    circuit: QuantumCircuit,
    config: Dict,
) -> Tuple[float, float]:
    """Apply Zero-Noise Extrapolation and return mitigated value with bounds.

    Args:
        circuit: Circuit to apply ZNE to
        config: ZNE configuration dictionary

    Returns:
        Tuple of (mitigated_value, error_bound)
    """
    scale_factors = config["scale_factors"]

    # Create noise model with configurable error rates
    noise_model = NoiseModel()
    from qiskit.circuit.library import IGate, XGate

    # Create noise model with basic depolarizing error
    error = QuantumError(
        [(IGate(), 0.999), (XGate(), 0.001)]  # 99.9% identity  # 0.1% bit flip
    )
    noise_model.add_all_qubit_quantum_error(error, ["id", "u1", "u2", "u3"])

    # Scale circuits by repeating gates
    scaled_circuits = []
    for scale in scale_factors:
        # Create new circuit with same registers as original
        scaled = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        for inst in circuit.data:
            # Repeat each gate 'scale' times
            for _ in range(int(scale)):
                scaled.append(inst.operation, inst.qubits, inst.clbits)
        scaled_circuits.append(scaled)

    # Add measurements and execute scaled circuits
    backend = AerSimulator(noise_model=noise_model)
    measured_circuits = []
    for circ in scaled_circuits:
        measured = circ.copy()
        measured.measure_all()
        measured_circuits.append(measured)

    sampler = Sampler()
    results = [
        sampler.run(circ, shots=1000, backend=backend).result().quasi_dists[0]
        for circ in measured_circuits
    ]

    # Convert counts to expectation values
    expectations = [
        sum(k.count("1") * v for k, v in result.items()) / sum(result.values())
        for result in results
    ]

    # Linear extrapolation to zero noise
    slope, intercept = np.polyfit(scale_factors, expectations, 1)
    mitigated = intercept  # Value at scale factor 0
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
    if len(scales) < 2 or len(results) < 2:
        return 0.1  # Default error bound for insufficient data

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        scales, results
    )
    residuals = np.array(results) - (intercept + slope * np.array(scales))
    error = 1.96 * np.std(residuals) / np.sqrt(len(scales))

    # Ensure non-zero error bound
    return max(error, 0.01)
