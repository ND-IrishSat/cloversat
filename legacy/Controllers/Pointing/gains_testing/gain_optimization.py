'''
gain_optimization.py

BAD AND UNFINISHED I BELIEVE

Runs multiple simulations with different starting conditions and simulation lengths
to evaluate the overall quality of different gain sets (KP, KD).

This script:
1. Tests multiple gain combinations (KP, KD)
2. Runs simulations with various initial conditions (quaternions, angular velocities)
3. Tests different simulation lengths
4. Collects control quality metrics for each run
5. Aggregates results and identifies the best gain set

Usage:
    python gain_optimization.py

The script will:
- Generate initial conditions automatically
- Pre-generate magnetic field data (if not using constant field)
- Run all simulations without plotting (for speed)
- Save results to a JSON file
- Print a comparison table of all gain sets

Configuration:
    Modify the gain_sets, sim_lengths, and num_initial_conditions variables
    in the main() function to customize the study.
'''

import numpy as np
import math
import sys
import os
from datetime import datetime
import json

# Add to path
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])

from Simulator.simulator import Simulator
from Simulator.sat_model import Magnetorquer_Sat
from Simulator.magnetorquer import Magnetorquer
from params import *

# Import PySOL
import Simulator.PySOL.wmm as wmm
from Simulator.PySOL.sol_sim import *
import Simulator.PySOL.spacecraft as sp
import Simulator.PySOL.orb_tools as ot


def generate_initial_conditions(num_conditions=10, seed=42):
    '''
    Generate a variety of initial conditions for testing.
    
    @params:
        num_conditions: Number of different initial conditions to generate
        seed: Random seed for reproducibility
    
    @returns:
        List of tuples: [(quaternion, velocity), ...]
    '''
    np.random.seed(seed)
    conditions = []
    
    # Add some predefined interesting starting conditions
    conditions.append((np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])))  # Nadir, no rotation
    conditions.append((np.array([0.707, 0.707, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])))  # 90 deg rotation
    conditions.append((np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1])))  # Nadir, small rotation
    conditions.append((np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])))  # Nadir, medium rotation
    
    # Generate random quaternions (uniformly distributed on unit sphere)
    for _ in range(num_conditions - len(conditions)):
        # Random quaternion
        q = np.random.normal(0, 1, 4)
        q = q / np.linalg.norm(q)
        
        # Random angular velocity (in rad/s, reasonable range)
        w = np.random.uniform(-0.5, 0.5, 3)
        
        conditions.append((q, w))
    
    return conditions


def run_single_simulation(kp, kd, quat_initial, velocity_initial, sim_hours, 
                          b_earth_data=None, gps_data=None, disable_plotting=True):
    '''
    Run a single simulation with specified parameters.
    
    @params:
        kp: Proportional gain
        kd: Derivative gain
        quat_initial: Initial quaternion [w, x, y, z]
        velocity_initial: Initial angular velocity [rad/s]
        sim_hours: Simulation duration in hours
        b_earth_data: Pre-generated magnetic field data (or None to generate)
        gps_data: Pre-generated GPS data (or None to generate)
        disable_plotting: If True, set RESULT=3 to disable all plotting
    
    @returns:
        dict: Control quality metrics or None if simulation failed
    '''
    # Temporarily disable plotting if requested
    original_result = None
    if disable_plotting:
        import params
        original_result = params.RESULT
        params.RESULT = 3  # Disable all plotting
    
    try:
        # Calculate simulation parameters
        tf = int(sim_hours * 3600)
        n_steps = int(tf / DT)
        
        # Generate or use provided magnetic field data
        if b_earth_data is None:
            if CONSTANT_B_FIELD:
                b_earth = np.full((n_steps, 3), CONSTANT_B_FIELD_MAG)
            else:
                oe = ORBITAL_ELEMENTS
                b_earth, gps = generate_orbit_data(oe, sim_hours, DT, None, store_data=False, GPS=True)
                if len(b_earth) < n_steps:
                    print(f"Warning: Not enough B field data. Need {n_steps}, got {len(b_earth)}")
                    return None
                b_earth = b_earth[:n_steps]
                if gps_data is None:
                    gps_data = gps[:n_steps]
        else:
            if len(b_earth_data) < n_steps:
                b_earth = b_earth_data[:len(b_earth_data)]
                n_steps = len(b_earth)
            else:
                b_earth = b_earth_data[:n_steps]
        
        # Get GPS data if needed
        if gps_data is None and not CONSTANT_B_FIELD:
            # Generate GPS data
            oe = ORBITAL_ELEMENTS
            _, gps_data = generate_orbit_data(oe, sim_hours, DT, None, store_data=False, GPS=True)
            if len(gps_data) < n_steps:
                gps_data = gps_data[:len(gps_data)]
            else:
                gps_data = gps_data[:n_steps]
        
        # Create magnetorquers
        mag1 = Magnetorquer(n=FERRO_NUM_TURNS, area=FERRO_AREA, k=K, epsilon=FERRO_EPSILON)
        mag2 = Magnetorquer(n=FERRO_NUM_TURNS, area=FERRO_AREA, k=K, epsilon=FERRO_EPSILON)
        mag3 = Magnetorquer(n=AIR_NUM_TURNS, area=AIR_AREA, k=K, epsilon=1)
        mag_array = np.array([mag1, mag2, mag3])
        
        # Initialize satellite with custom gains
        B_body = CONSTANT_B_FIELD_MAG if CONSTANT_B_FIELD else b_earth[0]
        prevB = np.array([0, 0, 0])
        mag_sat = Magnetorquer_Sat(CUBESAT_BODY_INERTIA, mag_array, velocity_initial, 
                                   B_body, prevB, DT, GYRO_WORKING, kp, kd)
        
        # Create simulator with custom initial quaternion
        # Normalize quaternion to ensure it's a unit quaternion
        from Simulator.all_EOMs import normalize
        quat_initial_normalized = normalize(quat_initial)
        
        sim = Simulator(mag_sat, b_earth)
        sim.quaternion = quat_initial_normalized
        sim.states[0] = np.concatenate((quat_initial_normalized, velocity_initial))
        
        # Run simulation
        i = 1
        while i < sim.n:
            sim.generateData_step(sim.states[i-1], i)
            
            # Call findNadir - GPS data is needed for nadir calculation
            if gps_data is not None and i < len(gps_data):
                sim.findNadir(sim.states[i-1], gps_data[i], i)
            else:
                # If no GPS data, use a default position (approximate)
                # For constant B field, we can use a fixed position
                # For orbital B field, this is less accurate but allows simulation to continue
                default_gps = np.array([ORBITAL_ELEMENTS[1] * 1000, 0, 0])  # Rough approximation
                sim.findNadir(sim.states[i-1], default_gps, i)
            
            sim.mag_sat.state = sim.check_state(i)
            sim.controls(i)
            sim.propagate_step(i)
            sim.totalPower[i] = sim.power_output[i][0] + sim.power_output[i][1] + sim.power_output[i][2]
            i += 1
        
        # Compute control quality metrics
        sim.compute_control_quality()
        
        # Restore original RESULT setting
        if disable_plotting and original_result is not None:
            import params
            params.RESULT = original_result
        
        return sim.controlQuality
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()
        if disable_plotting and original_result is not None:
            import params
            params.RESULT = original_result
        return None


def evaluate_gain_set(kp, kd, initial_conditions, sim_lengths, b_earth_data=None, gps_data=None):
    '''
    Evaluate a gain set by running multiple simulations with different conditions.
    
    @params:
        kp: Proportional gain
        kd: Derivative gain
        initial_conditions: List of (quaternion, velocity) tuples
        sim_lengths: List of simulation lengths in hours
        b_earth_data: Optional pre-generated magnetic field data
        gps_data: Optional pre-generated GPS data
    
    @returns:
        dict: Aggregated metrics for this gain set
    '''
    results = []
    total_runs = len(initial_conditions) * len(sim_lengths)
    completed = 0
    
    print(f"\nEvaluating KP={kp:.2e}, KD={kd:.2e}...")
    print(f"Total runs: {total_runs}")
    
    for quat_init, vel_init in initial_conditions:
        for sim_hours in sim_lengths:
            completed += 1
            print(f"  Run {completed}/{total_runs}: {sim_hours:.3f} hours, "
                  f"quat=[{quat_init[0]:.2f}, {quat_init[1]:.2f}, {quat_init[2]:.2f}, {quat_init[3]:.2f}], "
                  f"vel=[{vel_init[0]:.2f}, {vel_init[1]:.2f}, {vel_init[2]:.2f}]")
            
            quality = run_single_simulation(kp, kd, quat_init, vel_init, sim_hours, 
                                         b_earth_data, gps_data, disable_plotting=True)
            
            if quality is not None:
                quality['kp'] = kp
                quality['kd'] = kd
                quality['sim_hours'] = sim_hours
                quality['quat_init'] = quat_init.tolist()
                quality['vel_init'] = vel_init.tolist()
                results.append(quality)
    
    # Aggregate results
    if len(results) == 0:
        return None
    
    pointing_results = [r for r in results if r.get('pointing_detected', False)]
    
    aggregated = {
        'kp': kp,
        'kd': kd,
        'total_runs': len(results),
        'pointing_detected_count': len(pointing_results),
        'pointing_detection_rate': len(pointing_results) / len(results) if len(results) > 0 else 0,
    }
    
    if len(pointing_results) > 0:
        # Aggregate metrics for successful pointing runs
        rms_errors = [r['rms_error'] for r in pointing_results if r['rms_error'] is not None]
        steady_state_errors = [r['steady_state_error'] for r in pointing_results if r['steady_state_error'] is not None]
        convergence_times = [r['convergence_time'] for r in pointing_results if r['convergence_time'] is not None]
        settling_times = [r['settling_time'] for r in pointing_results if r['settling_time'] is not None]
        error_reductions = [r['error_reduction_percent'] for r in pointing_results if r['error_reduction_percent'] is not None]
        
        aggregated['avg_rms_error'] = np.mean(rms_errors) if rms_errors else None
        aggregated['std_rms_error'] = np.std(rms_errors) if rms_errors else None
        aggregated['avg_steady_state_error'] = np.mean(steady_state_errors) if steady_state_errors else None
        aggregated['avg_convergence_time'] = np.mean(convergence_times) if convergence_times else None
        aggregated['avg_settling_time'] = np.mean(settling_times) if settling_times else None
        aggregated['avg_error_reduction'] = np.mean(error_reductions) if error_reductions else None
        
        aggregated['min_rms_error'] = np.min(rms_errors) if rms_errors else None
        aggregated['max_rms_error'] = np.max(rms_errors) if rms_errors else None
    else:
        aggregated['avg_rms_error'] = None
        aggregated['std_rms_error'] = None
        aggregated['avg_steady_state_error'] = None
        aggregated['avg_convergence_time'] = None
        aggregated['avg_settling_time'] = None
        aggregated['avg_error_reduction'] = None
        aggregated['min_rms_error'] = None
        aggregated['max_rms_error'] = None
    
    aggregated['all_results'] = results
    
    return aggregated


def print_gain_comparison(aggregated_results):
    '''
    Print a comparison table of all gain sets tested.
    
    @params:
        aggregated_results: List of aggregated results from evaluate_gain_set
    '''
    print("\n" + "="*100)
    print("GAIN SET COMPARISON")
    print("="*100)
    
    # Sort by pointing detection rate, then by avg RMS error
    sorted_results = sorted(aggregated_results, 
                           key=lambda x: (x['pointing_detection_rate'], 
                                         -x['avg_rms_error'] if x['avg_rms_error'] is not None else float('inf')),
                           reverse=True)
    
    print(f"\n{'KP':<12} {'KD':<12} {'Detection':<12} {'Avg RMS':<12} {'Std RMS':<12} {'Avg Conv':<12} {'Avg Settle':<12}")
    print("-" * 100)
    
    for result in sorted_results:
        kp_str = f"{result['kp']:.2e}"
        kd_str = f"{result['kd']:.2e}"
        det_rate = f"{result['pointing_detection_rate']*100:.1f}%"
        rms_str = f"{result['avg_rms_error']:.6f}" if result['avg_rms_error'] is not None else "N/A"
        std_rms_str = f"{result['std_rms_error']:.6f}" if result['std_rms_error'] is not None else "N/A"
        conv_str = f"{result['avg_convergence_time']:.1f}s" if result['avg_convergence_time'] is not None else "N/A"
        settle_str = f"{result['avg_settling_time']:.1f}s" if result['avg_settling_time'] is not None else "N/A"
        
        print(f"{kp_str:<12} {kd_str:<12} {det_rate:<12} {rms_str:<12} {std_rms_str:<12} {conv_str:<12} {settle_str:<12}")
    
    print("="*100)


def save_results(aggregated_results, filename='finalization_results.json'):
    '''
    Save results to a JSON file.
    
    @params:
        aggregated_results: List of aggregated results
        filename: Output filename (default: finalization_results.json)
    '''
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gain_optimization_results_{timestamp}.json"
    
    # Convert numpy arrays and types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8, np.uint16,
                            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.complexfloating):
            return complex(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif obj is None:
            return None
        # Try to convert if it's a numpy scalar
        try:
            if hasattr(obj, 'item'):
                return obj.item()
        except (ValueError, AttributeError):
            pass
        return obj
    
    serializable_results = convert_to_serializable(aggregated_results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {filename}")


def main():
    '''
    Main function to run gain optimization study.
    '''
    print("="*80)
    print("GAIN SET OPTIMIZATION STUDY")
    print("="*80)
    
    # ============ CONFIGURATION ============
    
    # Simulation lengths to test (hours)
    sim_lengths = [
        1.5,    # 1.5 hours per simulation
    ]
    
    # Number of initial conditions to generate
    num_initial_conditions = 20
    
    # Generate gain sets: logarithmic sweep from max_gain to min_gain
    # For each step, we decrease by one order of magnitude
    max_gain = 10.0
    min_gain = 0.00001
    step_exponent = 1  # Step by 1 order of magnitude (10^1, 10^0, 10^-1, ...)
    
    # Generate logarithmic sequence: 10^1, 10^0, 10^-1, ..., 10^-5
    # This gives us: 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001
    gain_values = []
    exponent = int(np.log10(max_gain))  # Start at 1 (for 10^1)
    while True:
        gain_val = 10.0 ** exponent
        if gain_val < min_gain:
            break
        gain_values.append(gain_val)
        exponent -= step_exponent
    
    # Ensure we include min_gain exactly (10^-5 = 0.00001)
    if len(gain_values) == 0 or abs(gain_values[-1] - min_gain) > 1e-10:
        gain_values.append(min_gain)
    
    print(f"\nGain values to test: {gain_values}")
    print(f"Total gain values: {len(gain_values)}")
    
    # Generate all combinations of KP and KD
    gain_sets = []
    for kp in gain_values:
        for kd in gain_values:
            gain_sets.append((kp, kd))
    
    print(f"Total gain combinations: {len(gain_sets)}")
    print(f"With {num_initial_conditions} initial conditions and {len(sim_lengths)} simulation lengths:")
    total_sims = len(gain_sets) * num_initial_conditions * len(sim_lengths)
    print(f"Total simulations to run: {total_sims}")
    
    # Ask for confirmation if running a large number of simulations
    if total_sims > 100:
        response = input(f"\nThis will run {total_sims} simulations. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # ============ GENERATE INITIAL CONDITIONS ============
    print("\nGenerating initial conditions...")
    initial_conditions = generate_initial_conditions(num_initial_conditions)
    print(f"Generated {len(initial_conditions)} initial conditions")
    
    # ============ PRE-GENERATE MAGNETIC FIELD DATA (OPTIONAL) ============
    # Pre-generate once for all simulations to save time
    b_earth_data = None
    gps_data = None
    if not CONSTANT_B_FIELD:
        max_sim_hours = max(sim_lengths)
        print(f"\nPre-generating magnetic field data for {max_sim_hours:.3f} hours...")
        oe = ORBITAL_ELEMENTS
        b_earth_data, gps_data = generate_orbit_data(oe, max_sim_hours, DT, None, 
                                                     store_data=False, GPS=True)
        print(f"Generated {len(b_earth_data)} data points")
    
    # ============ EVALUATE EACH GAIN SET ============
    aggregated_results = []
    
    for kp, kd in gain_sets:
        result = evaluate_gain_set(kp, kd, initial_conditions, sim_lengths, 
                                   b_earth_data, gps_data)
        if result is not None:
            aggregated_results.append(result)
    
    # ============ PRINT RESULTS ============
    print_gain_comparison(aggregated_results)
    
    # ============ SAVE RESULTS ============
    save_results(aggregated_results)
    
    # ============ PRINT BEST GAIN SET ============
    if aggregated_results:
        # Find best gain set (highest detection rate, then lowest RMS error)
        best = max(aggregated_results, 
                  key=lambda x: (x['pointing_detection_rate'], 
                                -x['avg_rms_error'] if x['avg_rms_error'] is not None else -float('inf')))
        
        print("\n" + "="*80)
        print("RECOMMENDED GAIN SET")
        print("="*80)
        print(f"KP: {best['kp']:.2e}")
        print(f"KD: {best['kd']:.2e}")
        print(f"Pointing Detection Rate: {best['pointing_detection_rate']*100:.1f}%")
        if best['avg_rms_error'] is not None:
            print(f"Average RMS Error: {best['avg_rms_error']:.6f}")
        if best['avg_convergence_time'] is not None:
            print(f"Average Convergence Time: {best['avg_convergence_time']:.2f} seconds")
        print("="*80)


if __name__ == '__main__':
    main()

