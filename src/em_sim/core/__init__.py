"""Core components of the EM simulation framework."""

from .consciousness_integrator import ConsciousnessIntegrator
from .field_solver import EMFieldSolver
from .quantum_bridge import QuantumClassicalBridge

__all__ = [
    "EMFieldSolver",
    "ConsciousnessIntegrator",
    "QuantumClassicalBridge",
]
