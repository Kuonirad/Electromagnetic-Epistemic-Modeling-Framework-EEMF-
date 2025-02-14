# Quantum Circuit Mapping for Maxwell's Eigenvalue Problems

## Overview
The quantum circuit mapping module provides capabilities for solving Maxwell's eigenvalue problems using quantum computing resources. It supports both superconducting and trapped ion architectures, with optimizations for each platform.

## Key Features
- Riemann-Silberstein vector compression (5-10 qubits)
- LightSabre compiler for SWAP reduction
- Hardware-specific optimizations
- Comprehensive error mitigation
- Integration with classical EMFieldSolver

## Configuration

### Circuit Configuration
```python
from em_sim.quantum.circuit_mapper import CircuitConfig

config = CircuitConfig(
    qubit_count=10,          # Number of qubits to use
    depth=100,               # Circuit depth
    architecture="superconducting",  # or "trapped_ion"
    error_mitigation=True,   # Enable error mitigation
    rs_compression=True      # Enable Riemann-Silberstein compression
)
```

### Hardware Configuration
```python
from em_sim.quantum.hardware_config import HardwareConfig

# For superconducting qubits
hw_config = HardwareConfig.create_superconducting()

# For trapped ions
hw_config = HardwareConfig.create_trapped_ion()
```

## Usage Examples

### Cavity Mode Eigenvalue Problem
```python
from em_sim.core.field_solver import EMFieldSolver
from em_sim.quantum.circuit_mapper import CircuitConfig
from em_sim.quantum.hardware_config import HardwareConfig

# Configure quantum solver
quantum_config = {
    'circuit_config': CircuitConfig(qubit_count=10),
    'hardware_config': HardwareConfig.create_superconducting()
}

# Initialize solver
solver = EMFieldSolver(
    grid_size=(128, 128, 128),
    quantum_config=quantum_config
)

# Evolve fields using quantum solver
fields = solver.step(use_quantum_solver=True)
```

### Waveguide Mode Analysis
```python
from em_sim.quantum.circuit_mapper import MaxwellCircuitMapper, CircuitConfig

# Initialize circuit mapper
mapper = MaxwellCircuitMapper(CircuitConfig())

# Map waveguide problem
circuit = mapper.map_waveguide_modes(
    cross_section=[0.5, 0.5],
    material_properties={
        "epsilon": 2.1,
        "mu": 1.0
    }
)
```

## Error Mitigation Strategies

### Hardware-Specific Optimizations
- Superconducting qubits: Uses LightSabre compiler for SWAP reduction
- Trapped ions: Leverages all-to-all connectivity
- Automatic basis gate translation

### Validation
- Unitarity preservation checks
- Causality validation
- V-score computation (target < 0.1)
- Eigenvalue consistency (ΔE < 0.01 Ha) against FDTD

### Error Bounds
```python
from em_sim.quantum.validation import compute_error_bounds

error_metrics = compute_error_bounds(results)
v_score = error_metrics['v_score']
eigenvalue_std = error_metrics['eigenvalue_std']
```

## Performance Considerations

### Riemann-Silberstein Vector Compression
- Reduces qubit requirements by ~70%
- Maintains accuracy within error bounds
- Automatically enabled with `rs_compression=True`

### Hardware-Aware Circuit Optimization
- Superconducting: Optimizes for limited connectivity
- Trapped ions: Exploits long coherence times
- Automatic gate decomposition for target hardware

## Integration with Classical Solver

### Hybrid Classical-Quantum Approach
1. Classical domain decomposition
2. Quantum eigenvalue computation
3. Field updates using eigenvalues
4. MPI-based result gathering

### Consistency Checks
- Energy conservation validation
- Field normalization preservation
- Boundary condition satisfaction

## Troubleshooting

### Common Issues
1. Quantum solver errors
   - Check hardware configuration
   - Verify qubit count sufficiency
   - Ensure proper error mitigation

2. Performance issues
   - Enable RS compression
   - Adjust circuit depth
   - Use hardware-specific optimizations

3. Accuracy concerns
   - Check V-score
   - Verify eigenvalue consistency
   - Validate boundary conditions

### Validation Tools
```python
from em_sim.quantum.validation import (
    validate_unitarity,
    validate_causality,
    compute_error_bounds
)

# Check circuit properties
is_unitary = validate_unitarity(circuit)
preserves_causality = validate_causality(circuit)

# Compute error metrics
error_bounds = compute_error_bounds(results)
```
