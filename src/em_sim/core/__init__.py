"""Core components of the EM simulation framework."""

from .field_solver import EMFieldSolver
from .consciousness_integrator import ConsciousnessIntegrator
from .quantum_bridge import QuantumClassicalBridge

__all__ = ["EMFieldSolver", "ConsciousnessIntegrator", "QuantumClassicalBridge"]
