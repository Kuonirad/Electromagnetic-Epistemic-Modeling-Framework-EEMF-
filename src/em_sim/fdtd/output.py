"""Output handlers for FDTD simulation results."""

from pathlib import Path
from time import strftime
from typing import TYPE_CHECKING, Dict, Union

import h5py
import numpy as np

if TYPE_CHECKING:
    from .solver import FDTDParameters  # noqa: F401


def create_metadata(params: "FDTDParameters", dx: float, dt: float) -> Dict:
    """Generate standardized metadata dictionary.

    Includes:
    - Simulation parameters (grid size, time steps, dx, dt)
    - Timestamp of simulation
    - Source position and grid configuration
    - Boundary conditions
    - Physical constants used
    """
    return {
        "simulation_date": strftime("%Y-%m-%d %H:%M:%S"),
        "grid_size": params.grid_size,
        "time_steps": params.total_time,
        "dx": dx,
        "dt": dt,
        "source_position": params.grid_size // 2,
        "grid_configuration": {
            "spatial_step": dx,
            "time_step": dt,
            "courant_number": dt * params.c / dx,
        },
        "boundary_conditions": {
            "left": "PEC",
            "right": "Mur 1st Order ABC",
        },
        "physical_constants": {
            "speed_of_light": params.c,
            "vacuum_permittivity": 8.85e-12,
            "vacuum_permeability": 4e-7 * np.pi,
        },
    }


def save_results(
    data: np.ndarray,
    metadata: Dict,
    output_format: str = "txt",
    precision: int = 6,
    base_name: str = None,
    output_dir: Union[str, Path] = ".",
) -> str:
    """Save simulation results in the specified format with metadata.

    Args:
        data: Array containing Ex and Hy field data
        metadata: Dictionary of simulation metadata
        output_format: One of 'txt', 'csv', 'hdf5', or 'npz'
        precision: Number of decimal places for text output
        base_name: Base filename without extension
        output_dir: Directory to save output files

    Returns:
        Path to the saved file

    Raises:
        ValueError: If output_format is not one of: txt, csv, hdf5, npz
        OSError: If there are issues creating output directory or writing files
    """
    if output_format not in ["txt", "csv", "hdf5", "npz"]:
        raise ValueError(
            "Unsupported output format: "
            f"{output_format}. Valid formats: txt, csv, hdf5, npz"
        )

    try:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
    except OSError as e:
        raise OSError(f"Failed to create output directory: {e}") from e

    if base_name is None:
        base_name = f"fdtd_g{metadata['grid_size']}_t{metadata['time_steps']}"

    if output_format == "hdf5":
        return _save_hdf5(data, metadata, output_dir / f"{base_name}.h5")
    elif output_format == "npz":
        return _save_npz(data, metadata, output_dir / f"{base_name}.npz")
    else:
        return _save_text(
            data,
            metadata,
            output_dir / f"{base_name}.{output_format}",
            precision,
            delimiter="," if output_format == "csv" else " ",
        )


def _save_hdf5(data: np.ndarray, metadata: Dict, filepath: Path) -> str:
    """Save data in HDF5 format with compression.

    Raises:
        OSError: If there are issues writing to the HDF5 file
    """
    try:
        with h5py.File(filepath, "w") as hf:
            # Save field data with compression
            hf.create_dataset("Ex", data=data[:, 0], compression="gzip")
            hf.create_dataset("Hy", data=data[:, 1], compression="gzip")

            # Save metadata as attributes
            for key, value in metadata.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    for subkey, subvalue in value.items():
                        hf.attrs[f"{key}_{subkey}"] = str(subvalue)
                else:
                    hf.attrs[key] = str(value)
    except OSError as e:
        raise OSError(f"Failed to write HDF5 file: {e}") from e

    return str(filepath)


def _save_npz(data: np.ndarray, metadata: Dict, filepath: Path) -> str:
    """Save data in NPZ format for parallel I/O.

    Raises:
        OSError: If there are issues writing to the NPZ file
    """
    try:
        np.savez(
            filepath,
            Ex=data[:, 0],
            Hy=data[:, 1],
            metadata=np.array(str(metadata), dtype="S"),
        )
    except OSError as e:
        raise OSError(f"Failed to write NPZ file: {e}") from e

    return str(filepath)


def _save_text(
    data: np.ndarray,
    metadata: Dict,
    filepath: Path,
    precision: int,
    delimiter: str,
) -> str:
    """Save data in text format with metadata header.

    Raises:
        OSError: If there are issues writing to the text file
    """
    header = "\n".join(
        [
            (
                f"# {k}: {v}"
                if not isinstance(v, dict)
                else "\n".join(f"# {k}_{sk}: {sv}" for sk, sv in v.items())
            )
            for k, v in metadata.items()
        ]
    )

    try:
        np.savetxt(
            filepath,
            data,
            delimiter=delimiter,
            header=header,
            comments="",
            fmt=f"%.{precision}e",
        )
    except OSError as e:
        raise OSError(f"Failed to write text file: {e}") from e

    return str(filepath)
