"""FDTD (Finite-Difference Time-Domain) module.

Provides FDTD solvers for electromagnetic wave propagation, including
boundary conditions and visualization tools.
"""

from .solver import FDTD1DSolver, FDTDParameters

__all__ = ["FDTD1DSolver", "FDTDParameters"]
