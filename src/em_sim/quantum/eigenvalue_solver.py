"""Quantum eigenvalue solver for Maxwell problems."""

from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA


class MaxwellEigenvalueSolver:
    """Solves Maxwell eigenvalue problems using quantum circuits."""

    def __init__(self, shots: int = 1000, optimization_level: int = 3):
        """Initialize the eigenvalue solver.

        Args:
            shots: Number of measurement shots
            optimization_level: Circuit optimization level
        """
        self.shots = shots
        self.optimization_level = optimization_level

    def solve(self, circuit: QuantumCircuit) -> Tuple[np.ndarray, np.ndarray]:
        """Solve eigenvalue problem using VQE algorithm.

        Args:
            circuit: Quantum circuit representing the eigenvalue problem

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Initialize VQE with SPSA optimizer
        optimizer = SPSA(maxiter=100)
        vqe = VQE(optimizer=optimizer)

        # Execute VQE
        result = vqe.compute_minimum_eigenvalue(operator=circuit)

        return result.eigenvalue, result.optimal_point
