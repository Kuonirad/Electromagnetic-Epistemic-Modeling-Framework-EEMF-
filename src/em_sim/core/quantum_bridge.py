"""Quantum-Classical Bridge for EM field state conversion."""

import numpy as np
import torch
from typing import Dict, Optional, Tuple

class QuantumClassicalBridge:
    """Handles conversion between quantum and classical EM field states."""
    
    def __init__(
        self,
        hilbert_dim: int = 64,
        device: str = "cuda"
    ):
        """Initialize the quantum-classical bridge.
        
        Args:
            hilbert_dim: Dimension of the Hilbert space for quantum states
            device: Computation device ('cuda' or 'cpu')
        """
        self.hilbert_dim = hilbert_dim
        self.device = device
        
        # Initialize basis states
        self.basis_states = self._create_basis_states()
    
    def _create_basis_states(self) -> torch.Tensor:
        """Create computational basis states."""
        basis = torch.eye(self.hilbert_dim, device=self.device)
        return basis
    
    def classical_to_quantum(
        self,
        classical_field: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Convert classical EM field to quantum state representation.
        
        Args:
            classical_field: Dictionary containing E and B field tensors
            
        Returns:
            Quantum state tensor in the computational basis
        """
        # Normalize field amplitudes
        E_norm = torch.norm(classical_field["E"])
        B_norm = torch.norm(classical_field["B"])
        
        # Create superposition state
        alpha = E_norm / (E_norm + B_norm)
        beta = B_norm / (E_norm + B_norm)
        
        # Ensure normalization
        quantum_state = torch.sqrt(alpha) * self.basis_states[0] + \
                       torch.sqrt(beta) * self.basis_states[1]
                       
        return quantum_state
    
    def quantum_to_classical(
        self,
        quantum_state: torch.Tensor,
        grid_size: Tuple[int, int, int]
    ) -> Dict[str, torch.Tensor]:
        """Convert quantum state to classical EM field representation.
        
        Args:
            quantum_state: Quantum state tensor
            grid_size: Dimensions for classical field grid
            
        Returns:
            Dictionary containing reconstructed E and B fields
        """
        # Project onto basis states
        proj_0 = torch.abs(torch.dot(quantum_state, self.basis_states[0])) ** 2
        proj_1 = torch.abs(torch.dot(quantum_state, self.basis_states[1])) ** 2
        
        # Reconstruct classical fields
        E = torch.randn((*grid_size, 3), device=self.device) * torch.sqrt(proj_0)
        B = torch.randn((*grid_size, 3), device=self.device) * torch.sqrt(proj_1)
        
        return {"E": E, "B": B}
    
    def compute_entanglement(self, quantum_state: torch.Tensor) -> float:
        """Calculate the entanglement entropy of the quantum state.
        
        Args:
            quantum_state: Input quantum state
            
        Returns:
            von Neumann entropy as entanglement measure
        """
        # Compute density matrix
        rho = torch.outer(quantum_state, quantum_state.conj())
        
        # Calculate eigenvalues
        eigenvals = torch.linalg.eigvalsh(rho)
        
        # Compute von Neumann entropy
        entropy = -torch.sum(eigenvals * torch.log2(eigenvals + 1e-10))
        
        return entropy.item()
