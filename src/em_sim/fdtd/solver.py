"""1D FDTD solver for electromagnetic waves.

Provides support for:
- PEC (Perfect Electric Conductor) boundary conditions
- Mur absorbing boundary conditions
- Gaussian pulse source injection
- Real-time visualization capabilities
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FDTDParameters:
    """Configuration parameters for the FDTD simulation."""

    grid_size: int = 200
    total_time: int = 1000
    dx: float = 1e-3  # Spatial step (1mm)
    c: float = 3e8  # Speed of light

    @property
    def dt(self) -> float:
        """Calculate timestep using Courant condition."""
        return self.dx / (2 * self.c)


class FDTD1DSolver:
    """One-dimensional FDTD solver for electromagnetic wave propagation."""

    def __init__(self, params: Optional[FDTDParameters] = None):
        """Initialize the FDTD solver with given parameters."""
        self.params = params or FDTDParameters()

        # Initialize field arrays
        self.Ex = np.zeros(self.params.grid_size, dtype=np.float64)
        self.Hy = np.zeros(self.params.grid_size - 1, dtype=np.float64)
        self.Ex_old = np.zeros_like(self.Ex)  # For Mur ABC

        # Physical constants
        self.epsilon_0 = 8.85e-12  # Vacuum permittivity
        self.mu_0 = 4e-7 * np.pi  # Vacuum permeability

        # Visualization setup
        self.fig = None
        self.ax = None
        self.line = None

    def setup_visualization(self) -> None:
        """Initialize real-time visualization."""
        plt.ion()
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot(self.Ex)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_title("1D FDTD Simulation")
        self.ax.set_xlabel("Grid Position")
        self.ax.set_ylabel("Electric Field (Ex)")

    def update_fields(self) -> None:
        """Update magnetic and electric fields using FDTD equations."""
        # Update Magnetic Field (Hy)
        self.Hy += (self.params.dt / (self.mu_0 * self.params.dx)) * (
            self.Ex[:-1] - self.Ex[1:]
        )

        # Update Electric Field (Ex)
        self.Ex[1:-1] += (
            self.params.dt / (self.epsilon_0 * self.params.dx)
        ) * (self.Hy[:-1] - self.Hy[1:])

    def apply_boundary_conditions(self) -> None:
        """Apply PEC and Mur absorbing boundary conditions."""
        # PEC Left Boundary
        self.Ex[0] = 0.0

        # Mur First-Order ABC Right Boundary
        self.Ex[-1] = self.Ex_old[-2] + (
            (self.params.c * self.params.dt - self.params.dx)
            / (self.params.c * self.params.dt + self.params.dx)
        ) * (self.Ex[-1] - self.Ex_old[-2])
        self.Ex_old[:] = self.Ex[:]

    def inject_source(self, n: int, position: int = 100) -> None:
        """Inject a Gaussian pulse source at the specified position."""
        pulse = np.exp(-0.5 * ((n - 30) / 10) ** 2)
        self.Ex[position] += pulse

    def update_visualization(self) -> None:
        """Update the visualization plot."""
        if self.line is not None:
            self.line.set_ydata(self.Ex)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def get_field_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current electric and magnetic field data."""
        return self.Ex.copy(), np.append(self.Hy, 0)

    def save_field_data(self, filename: str) -> None:
        """Save the current field data to a file."""
        Ex, Hy = self.get_field_data()
        np.savetxt(
            filename, np.column_stack((Ex, Hy)), header="Ex Hy", comments=""
        )

    def run_simulation(
        self, visualize: bool = True, save_output: bool = True
    ) -> None:
        """Run the complete FDTD simulation."""
        if visualize:
            self.setup_visualization()

        for n in range(self.params.total_time):
            self.update_fields()
            self.apply_boundary_conditions()
            self.inject_source(n)

            if visualize and n % 50 == 0:
                self.update_visualization()

        if save_output:
            self.save_field_data("fdtd_output.txt")
