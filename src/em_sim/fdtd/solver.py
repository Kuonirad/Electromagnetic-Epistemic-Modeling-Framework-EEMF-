"""1D FDTD solver for electromagnetic waves.

Provides support for:
- PEC (Perfect Electric Conductor) boundary conditions
- Mur absorbing boundary conditions
- Gaussian pulse source injection
- Real-time visualization capabilities
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from . import output


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

    def save_field_data(
        self,
        filename: str,
        output_format: str = "txt",
        precision: int = 6,
        output_dir: Union[str, Path] = ".",
    ) -> str:
        """Save the current field data to a file.

        Args:
            filename: Base filename without extension
            output_format: One of 'txt', 'csv', 'hdf5', or 'npz'
            precision: Number of decimal places for text output
            output_dir: Directory to save output files

        Returns:
            Path to the saved file

        Raises:
            ValueError: If output_format is not one of: txt, csv, hdf5, npz
            OSError: If there are issues creating output directory or writing
                files
        """
        if output_format not in ["txt", "csv", "hdf5", "npz"]:
            raise ValueError(
                "Unsupported output format: "
                f"{output_format}. Supported formats: txt, csv, hdf5, npz"
            )

        Ex, Hy = self.get_field_data()
        data = np.column_stack((Ex, Hy))
        params = self.params
        metadata = output.create_metadata(params, params.dx, params.dt)

        return output.save_results(
            data=data,
            metadata=metadata,
            output_format=output_format,
            precision=precision,
            base_name=filename,
            output_dir=output_dir,
        )

    def run_simulation(
        self,
        visualize: bool = True,
        save_output: bool = True,
        output_format: str = "txt",
        precision: int = 6,
    ) -> Optional[str]:
        """Run the complete FDTD simulation.

        Args:
            visualize: Whether to show real-time visualization
            save_output: Whether to save simulation results
            output_format: One of 'txt', 'csv', 'hdf5', or 'npz'
            precision: Number of decimal places for text output

        Returns:
            Path to the output file if save_output is True

        Raises:
            ValueError: If output_format is not one of: txt, csv, hdf5, npz
            OSError: If there are issues saving output files to disk
        """
        if save_output and output_format not in ["txt", "csv", "hdf5", "npz"]:
            raise ValueError(
                "Unsupported output format: "
                f"{output_format}. Supported formats: txt, csv, hdf5, npz"
            )

        if visualize:
            self.setup_visualization()

        for n in range(self.params.total_time):
            self.update_fields()
            self.apply_boundary_conditions()
            self.inject_source(n)

            if visualize and n % 50 == 0:
                self.update_visualization()

        if save_output:
            return self.save_field_data(
                "fdtd_output", output_format=output_format, precision=precision
            )
        return None
