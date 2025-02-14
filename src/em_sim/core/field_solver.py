"""Electromagnetic field solver with consciousness coupling."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from mpi4py import MPI


class EMFieldSolver:
    """Solves electromagnetic field equations with consciousness coupling."""

    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (128, 128, 128),
        dt: float = 1e-6,
        device: str = "cuda",
    ):
        """Initialize the EM field solver.

        Args:
            grid_size: 3D grid dimensions for field discretization
            dt: Time step for numerical integration
            device: Computation device ('cuda' or 'cpu')
        """
        self.grid_size = grid_size
        self.dt = dt
        self.device = device
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Initialize field tensors on GPU if available
        if self.device == "cuda":
            self.E = torch.zeros((*grid_size, 3), device="cuda")
            self.B = torch.zeros((*grid_size, 3), device="cuda")
        else:
            self.E = torch.zeros((*grid_size, 3))
            self.B = torch.zeros((*grid_size, 3))

    def step(
        self, consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Perform one time step of field evolution.

        Args:
            consciousness_field: Optional coupling field from
                consciousness integration

        Returns:
            Dict containing updated E and B fields
        """
        # MPI domain decomposition
        local_E = self._decompose_domain(self.E)
        local_B = self._decompose_domain(self.B)

        # Update fields with Maxwell's equations
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
