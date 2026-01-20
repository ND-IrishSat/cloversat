# Control Quality Metrics Usage Guide

## Overview

The `Simulator` class now includes methods to compute and analyze control quality metrics from the `nadirError` and `errorQuats` data collected during simulation.

## Methods Added

### `compute_control_quality(pointing_threshold=0.1, steady_state_window=0.1, convergence_threshold=0.05)`

Computes comprehensive quality metrics for the nadir pointing control run. Should be called **after** the simulation is complete.

**Parameters:**
- `pointing_threshold`: Error threshold for considering system "pointing" (default 0.1)
- `steady_state_window`: Fraction of simulation to use for steady-state calculation (default 0.1 = last 10%)
- `convergence_threshold`: Error threshold for convergence time calculation (default 0.05)

**Returns:** Dictionary containing all computed quality metrics

**Metrics Computed:**
- `pointing_detected`: Whether a pointing phase was detected
- `pointing_start_time`: Time when pointing mode began (seconds)
- `pointing_end_time`: Time when pointing mode ended (seconds)
- `pointing_duration`: Total time spent in pointing mode (seconds)
- `initial_error`: Error at the start of pointing phase
- `final_error`: Error at the end of simulation
- `steady_state_error`: Average error over the last portion of pointing phase
- `convergence_time`: Time to first reach within convergence_threshold (seconds, or None)
- `convergence_threshold_used`: The threshold used for convergence calculation
- `rms_error`: Root mean square error during pointing phase
- `peak_error`: Maximum error during pointing phase
- `settling_time`: Time to reach and stay within pointing_threshold (seconds, or None)
- `settling_threshold_used`: The threshold used for settling calculation
- `error_reduction_percent`: Percentage reduction from initial to final error
- `pointing_indices`: Tuple of (start_index, end_index) for pointing phase

### `print_control_quality()`

Prints a formatted summary of control quality metrics. Must be called after `compute_control_quality()`.

## Usage Examples

### Basic Usage

```python
# After running your simulation
sim = Simulator(mag_sat, B_earth)

# ... run simulation loop ...

# After simulation completes, compute quality metrics
sim.compute_control_quality()

# Print formatted summary
sim.print_control_quality()

# Access metrics programmatically
quality = sim.controlQuality
print(f"RMS Error: {quality['rms_error']}")
print(f"Convergence Time: {quality['convergence_time']}")
```

### Custom Thresholds

```python
# Use custom thresholds for your specific requirements
sim.compute_control_quality(
    pointing_threshold=0.05,      # Stricter pointing requirement
    steady_state_window=0.2,       # Use last 20% for steady-state
    convergence_threshold=0.02    # Stricter convergence requirement
)
sim.print_control_quality()
```

### Integration with Existing Code

Add to your existing simulation scripts (e.g., `nearspace_simulator.py`):

```python
# After simulation loop completes
sim.plot_and_viz_results()

# Compute and print control quality metrics
sim.compute_control_quality()
sim.print_control_quality()

# Optionally access metrics for further analysis
if sim.controlQuality['pointing_detected']:
    print(f"Control performance: RMS error = {sim.controlQuality['rms_error']:.6f}")
    if sim.controlQuality['convergence_time']:
        print(f"Converged in {sim.controlQuality['convergence_time']:.2f} seconds")
```

### Comparing Multiple Runs

```python
results = []
for run in range(num_runs):
    sim = Simulator(mag_sat, B_earth)
    # ... run simulation ...
    sim.compute_control_quality()
    results.append(sim.controlQuality)

# Compare metrics across runs
rms_errors = [r['rms_error'] for r in results if r['pointing_detected']]
convergence_times = [r['convergence_time'] for r in results 
                     if r['pointing_detected'] and r['convergence_time']]
print(f"Average RMS Error: {np.mean(rms_errors)}")
print(f"Average Convergence Time: {np.mean(convergence_times)}")
```

## Interpreting Metrics

- **RMS Error**: Lower is better. Indicates overall pointing accuracy during the pointing phase.
- **Steady-State Error**: Lower is better. Indicates final pointing accuracy after system has settled.
- **Convergence Time**: Lower is better. Time to reach a small error threshold.
- **Settling Time**: Lower is better. Time to reach and maintain pointing accuracy.
- **Error Reduction**: Higher is better. Percentage improvement from start to end of pointing phase.
- **Peak Error**: Lower is better. Maximum error during pointing phase (indicates overshoot).

## Notes

- The method automatically identifies the pointing phase by looking for `mode >= 0` (pointing modes are 0 or 1)
- If no pointing phase is detected, most metrics will be `None`
- All time-based metrics are in seconds
- Error values are the magnitude of the error quaternion vector component (x,y,z)

