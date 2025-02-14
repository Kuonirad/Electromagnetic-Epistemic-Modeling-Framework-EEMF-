"""Quantum hardware configuration."""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class HardwareConfig:
    """Quantum hardware configuration."""
    
    architecture: str
    connectivity_map: Dict[int, List[int]]
    decoherence_times: Dict[str, float]
    gate_errors: Dict[str, float]
    
    @classmethod
    def create_superconducting(cls) -> 'HardwareConfig':
        """Create configuration for superconducting qubit architecture."""
        return cls(
            architecture="superconducting",
            connectivity_map={
                0: [1, 2],
                1: [0, 3],
                2: [0, 4],
                3: [1, 5],
                4: [2, 5],
                5: [3, 4]
            },
            decoherence_times={
                'T1': 100e-6,  # 100 μs
                'T2': 150e-6   # 150 μs
            },
            gate_errors={
                'single_qubit': 1e-3,
                'two_qubit': 1e-2
            }
        )
    
    @classmethod
    def create_trapped_ion(cls) -> 'HardwareConfig':
        """Create configuration for trapped ion architecture."""
        return cls(
            architecture="trapped_ion",
            connectivity_map={i: list(range(10)) for i in range(10)},  # All-to-all
            decoherence_times={
                'T1': 10.0,     # 10 s
                'T2': 1.0       # 1 s
            },
            gate_errors={
                'single_qubit': 1e-4,
                'two_qubit': 1e-3
            }
        )
