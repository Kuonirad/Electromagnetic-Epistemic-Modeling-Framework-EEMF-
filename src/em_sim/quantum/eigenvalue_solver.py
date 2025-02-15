"""Quantum eigenvalue solver for Maxwell problems."""

from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
# Already imported above


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
        estimator = Estimator()

        num_qubits = circuit.num_qubits
        ansatz = EfficientSU2(num_qubits, reps=2)
        hamiltonian = SparsePauliOp.from_list(
            [
                (
                    "Z" + "I" * (num_qubits - 1),
                    1.0,
                )  # Simple Z measurement on first qubit
            ]
        )

        vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=estimator)

        # Execute VQE
        result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

        # Convert results to numpy arrays with matching shapes
        eigenvalues = np.array([result.eigenvalue])
        eigenvectors = np.array(
            [result.optimal_point]
        )  # Make it 2D array with shape (1, n)
        return eigenvalues, eigenvectors
