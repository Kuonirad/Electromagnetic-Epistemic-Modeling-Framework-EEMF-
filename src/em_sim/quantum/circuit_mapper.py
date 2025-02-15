"""Maxwell eigenvalue problem to quantum circuit mapper."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap, transpile

from .error_mitigation import apply_zne

# Define Pauli operators
X = SparsePauliOp.from_list([("X", 1.0)])
Y = SparsePauliOp.from_list([("Y", 1.0)])
Z = SparsePauliOp.from_list([("Z", 1.0)])
IDENTITY = SparsePauliOp.from_list([("I", 1.0)])


@dataclass
class CircuitConfig:
    """Configuration for quantum circuit mapping."""

    qubit_count: int = 10
    depth: int = 100
    architecture: str = "superconducting"
    error_mitigation: bool = True
    rs_compression: bool = True  # Riemann-Silberstein vector compression
    zne_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "scale_factors": [1, 2, 3],
            "extrapolation_method": "linear",
            "noise_amplifier": "folding",
        }
    )


class MaxwellCircuitMapper:
    """Maps Maxwell eigenvalue problems to quantum circuits."""

    def __init__(self, config: CircuitConfig):
        self.config = config
        self._initialize_hamiltonian_templates()

    def _initialize_hamiltonian_templates(self):
        """Precompute common Hamiltonian components for Maxwell systems."""
        # Use SparsePauliOp for operator composition
        zzi = SparsePauliOp.from_list([("ZZI", 1.0)])
        xxi = SparsePauliOp.from_list([("XXI", 1.0)])
        yyz = SparsePauliOp.from_list([("YYZ", 1.0)])
        xzx = SparsePauliOp.from_list([("XZX", 1.0)])

        self.cavity_hamiltonian = {
            "TE": (zzi + xxi).simplify(),
            "TM": (xxi + zzi).simplify(),
        }

        # Waveguide mode operators
        self.waveguide_hamiltonian = {
            "TE": (yyz * 1.2).simplify(),
            "TEM": (xzx * 0.8).simplify(),
        }

    def _apply_boundary_conditions(self, circ: QuantumCircuit, bc_type: str):
        """Implement PEC/PMC boundary conditions via rotation gates."""
        if bc_type == "PEC":
            circ.rz(np.pi / 2, range(circ.num_qubits))
        elif bc_type == "PMC":
            circ.rx(-np.pi / 2, range(circ.num_qubits))

    def _hardware_optimize(
        self, circ: QuantumCircuit, hw_config
    ) -> QuantumCircuit:
        """Apply architecture-specific optimizations with error mitigation."""
        # Apply base optimizations
        if hw_config.architecture == "superconducting":
            # Convert connectivity map to edge list for CouplingMap
            edges = []
            if hasattr(hw_config, "connectivity_map"):
                edges = [
                    (k, v)
                    for k, adj in hw_config.connectivity_map.items()
                    for v in adj
                ]
            # Use LightSabre-inspired layout optimization
            optimized = transpile(
                circ,
                coupling_map=CouplingMap(edges),
                routing_method="sabre",
                optimization_level=3,
            )
        elif hw_config.architecture == "trapped_ion":
            # Implement all-to-all connectivity optimization
            optimized = transpile(
                circ,
                coupling_map=None,
                basis_gates=["rxx", "rz", "ry", "cx", "h"],
            )
        else:
            optimized = circ

        # Apply error mitigation if enabled
        if self.config.error_mitigation:
            mitigated, error_bound = apply_zne(
                optimized, self.config.zne_config
            )
            # Store error metrics
            self._error_metrics = {
                "mitigated_value": mitigated,
                "error_bound": error_bound,
            }

        return optimized

    def map_cavity_modes(
        self, dimensions: List[float], boundary_conditions: Dict[str, str]
    ) -> QuantumCircuit:
        """Implement RS vector-compressed cavity mapping."""
        q = QuantumRegister(self.config.qubit_count)
        qc = QuantumCircuit(q)

        if self.config.rs_compression:
            # Riemann-Silberstein vector compression scheme
            # Apply Hadamard gates to available qubits
            qc.h(range(0, self.config.qubit_count, 2))
            # Apply CNOT gates between adjacent qubits
            for i in range(0, self.config.qubit_count - 1, 2):
                qc.cx(i, i + 1)

        # Apply boundary condition gates
        self._apply_boundary_conditions(
            qc, boundary_conditions.get("type", "PEC")
        )

        # Generate TE/TM Hamiltonians
        for qubit in range(self.config.qubit_count - 2):
            # Apply evolution only if we have enough qubits
            if qubit + 2 < self.config.qubit_count:
                qc.append(
                    PauliEvolutionGate(
                        self.cavity_hamiltonian["TE"], time=dimensions[0] / 2
                    ),
                    [qubit, qubit + 1, qubit + 2],
                )

        qc.barrier()
        return qc

    def map_waveguide_modes(
        self, cross_section: List[float], material_properties: Dict[str, float]
    ) -> QuantumCircuit:
        """Implement waveguide mode mapping with material dispersion."""
        q = QuantumRegister(self.config.qubit_count)
        qc = QuantumCircuit(q)

        # Material property encoding
        eps_r = material_properties["epsilon"]
        mu_r = material_properties["mu"]
        z0 = np.sqrt(mu_r / eps_r)

        # Create FermionicOp with proper format
        # Create SparsePauliOp directly instead of using FermionicOp
        waveguide_op = SparsePauliOp.from_list(
            [
                ("X" * self.config.qubit_count, (3 * z0) / 2),
                ("Y" * self.config.qubit_count, (5 * (1 / z0)) / 2),
            ]
        )

        qc.append(
            PauliEvolutionGate(waveguide_op, time=cross_section[0]),
            range(self.config.qubit_count),
        )

        return qc

    def map_topological_boundary(
        self, surface_geometry: np.ndarray, correlation_map: np.ndarray
    ) -> QuantumCircuit:
        """Implement non-Hermitian boundary mapping."""
        q = QuantumRegister(self.config.qubit_count)
        qc = QuantumCircuit(q)

        # Surface geometry embedding
        for i in range(surface_geometry.shape[0]):
            qc.rzx(
                surface_geometry[i, 0],
                i % self.config.qubit_count,
                (i + 1) % self.config.qubit_count,
            )

        # Correlation surface mapping
        for src, tgt in correlation_map:
            qc.cx(src, tgt)
            qc.cz(src, tgt)

        return qc
