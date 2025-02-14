"""Electromagnetic field solver with consciousness coupling."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from mpi4py import MPI

from ..quantum.circuit_mapper import CircuitConfig, MaxwellCircuitMapper
from ..quantum.eigenvalue_solver import MaxwellEigenvalueSolver
from ..quantum.hardware_config import HardwareConfig


class EMFieldSolver:
    """Solves electromagnetic field equations with consciousness coupling."""

    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (128, 128, 128),
        dt: float = 1e-6,
        device: str = "cuda",
        quantum_config: Optional[
            Dict[str, Union[CircuitConfig, HardwareConfig]]
        ] = None,
    ):
        """Initialize the EM field solver.

        Args:
            grid_size: 3D grid dimensions for field discretization
            dt: Time step for numerical integration
            device: Computation device ('cuda' or 'cpu')
            quantum_config: Optional configuration for quantum circuit mapping
        """
        self.grid_size = grid_size
        self.dt = dt
        self.device = device
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Initialize quantum components if config provided
        self.quantum_config = quantum_config
        if quantum_config is not None:
            circuit_config = quantum_config.get(
                "circuit_config", CircuitConfig()
            )
            self.circuit_mapper = MaxwellCircuitMapper(circuit_config)
            self.eigenvalue_solver = MaxwellEigenvalueSolver()
            self.hardware_config = quantum_config.get(
                "hardware_config", HardwareConfig.create_superconducting()
            )

        # Initialize field tensors on GPU if available
        if self.device == "cuda":
            self.E = torch.zeros((*grid_size, 3), device="cuda")
            self.B = torch.zeros((*grid_size, 3), device="cuda")
        else:
            self.E = torch.zeros((*grid_size, 3))
            self.B = torch.zeros((*grid_size, 3))

    def step(
        self,
        consciousness_field: Optional[torch.Tensor] = None,
        use_quantum_solver: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Perform one time step of field evolution.

        Args:
            consciousness_field: Optional coupling field from consciousness
                integration
            use_quantum_solver: Whether to use quantum circuit for eigenvalue
                computation

        Returns:
            Dict containing updated E and B fields

        Raises:
            ValueError: If quantum_solver=True but no quantum config was
                provided
        """
        if use_quantum_solver and self.quantum_config is None:
            raise ValueError(
                "Quantum solver requested but no quantum config provided"
            )
        # MPI domain decomposition
        local_E = self._decompose_domain(self.E)
        local_B = self._decompose_domain(self.B)

        if use_quantum_solver:
            # Map to quantum circuit and solve eigenvalue problem
            circuit = self.circuit_mapper.map_cavity_modes(
                dimensions=self.grid_size, boundary_conditions={"type": "PEC"}
            )
            circuit = self.circuit_mapper._hardware_optimize(
                circuit, self.hardware_config
            )
            eigenvalues, _ = self.eigenvalue_solver.solve(circuit)

            # Use eigenvalues to update fields
            local_E_new = self._update_E_field_quantum(
                local_E, local_B, eigenvalues, consciousness_field
            )
            local_B_new = self._update_B_field_quantum(
                local_E, local_B, eigenvalues
            )
        else:
            # Update fields with classical Maxwell's equations
            params = (local_E, local_B, consciousness_field)
            local_E_new = self._update_E_field(*params)
            local_B_new = self._update_B_field(local_E, local_B)

        # Gather results across MPI ranks
        self.E = self._gather_domain(local_E_new)
        self.B = self._gather_domain(local_B_new)

        return {"E": self.E, "B": self.B}

    def _decompose_domain(self, field: torch.Tensor) -> torch.Tensor:
        """Decompose field for parallel processing."""
        # Implement MPI domain decomposition
        field_cpu = field.cpu().numpy()
        chunks = np.array_split(field_cpu, self.comm.size, axis=0)
        local_chunk = torch.tensor(chunks[self.rank], device=self.device)
        return local_chunk

    def _gather_domain(self, local_field: torch.Tensor) -> torch.Tensor:
        """Gather decomposed field from all processes."""
        # Implement MPI gather operation
        gathered = self.comm.gather(local_field.cpu().numpy(), root=0)
        if self.rank == 0:
            field = np.concatenate(gathered, axis=0)
            return torch.tensor(field, device=self.device)
        # Non-root processes return original field
        return self.E

    def _update_E_field(
        self,
        E: torch.Tensor,
        B: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update electric field using Maxwell's equations."""
        # Update E field
        grad_E = torch.gradient(E)[0]
        E_new = E + self.dt * torch.cross(B, grad_E)

        # Add consciousness coupling if present
        if consciousness_field is not None:
            E_new += self.dt * consciousness_field

        return E_new

    def _update_B_field(
        self, E: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        """Update magnetic field using Maxwell's equations."""
        # Update B field
        grad_B = torch.gradient(B)[0]
        cross_prod = torch.cross(E, grad_B)
        B_new = B - self.dt * cross_prod
        return B_new

    def _update_E_field_quantum(
        self,
        E: torch.Tensor,
        B: torch.Tensor,
        eigenvalues: np.ndarray,
        consciousness_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update electric field using quantum eigenvalue solutions."""
        # Convert eigenvalues to appropriate device/type
        eig_tensor = torch.tensor(eigenvalues, device=self.device)

        # Use eigenvalues to modify field evolution
        grad_E = torch.gradient(E)[0]
        E_new = E + self.dt * (
            torch.cross(B, grad_E) * torch.exp(-eig_tensor[0])
        )

        # Add consciousness coupling if present
        if consciousness_field is not None:
            E_new += self.dt * consciousness_field

        return E_new

    def _update_B_field_quantum(
        self,
        E: torch.Tensor,
        B: torch.Tensor,
        eigenvalues: np.ndarray,
    ) -> torch.Tensor:
        """Update magnetic field using quantum eigenvalue solutions."""
        # Convert eigenvalues to appropriate device/type
        eig_tensor = torch.tensor(eigenvalues, device=self.device)

        # Use eigenvalues to modify field evolution
        grad_B = torch.gradient(B)[0]
        cross_prod = torch.cross(E, grad_B)
        B_new = B - self.dt * (cross_prod * torch.exp(-eig_tensor[0]))

        return B_new
