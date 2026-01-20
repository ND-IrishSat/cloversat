'''
GUI for testing attitude determination algorithms using EHS image dataset
'''

import os
import json
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, CheckButtons
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from itertools import combinations, product

# Add the parent directory (NearSpace/) to the Python path to import params
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from Pointing/ to NearSpace/
sys.path.insert(0, project_root)

from params import *
from image_processing import processImage
from Simulator.sat_model import Magnetorquer_Sat
from Simulator.all_EOMs import *
from nadir_point import determine_attitude

# CASES = {
#     # Earth in bottom of both cams
#     1: [0, 3, 51, 504, 530, 549, 555, 552],
#     # All earth in 1, sliver in other
#     2: [296, 4],
#     # Space in 1, earth in other
#     3: [537, 496, 308, 43, 294, 295, 310, 311, 486, 390, 557],
#     # Earth upside down in both cams
#     4: [12, 36, 60, 264, 288],
#     # Earth is sideways in both cams
#     5: [141, 437, 472, 463, 445, 126, 180],
#     # Earth is in opposite corners (similar to #5)
#     6: [77, 547, 93],
# }
CASES = {
    # Earth in bottom of both cams
    1: [0, 51, 504, 530, 549, 555, 552],
    # All earth in 1, sliver in other
    2: [296, 4, 303, 3],
    # Space in 1, earth in other
    3: [537, 496, 308, 43, 294, 295, 310, 311, 313, 486, 390, 557],
    # Earth is never upside down in both cams (all case 3)
    4: [126],
    # Earth is sideways in both cams
    5: [141, 437, 472, 463, 445, 126, 180],
    # Earth is in opposite corners (similar to #5)
    6: [77, 547, 93, 523, 208, 209, 198, 199],
}

DEFAULT_DATASET_DIR = os.path.join(
    project_root,
    "Maya_project",
    "IrishSat_Simulator",
    "images",
    "dataset",
)

CROPPED_DATASET_DIR = os.path.join(
    project_root,
    "Maya_project",
    "IrishSat_Simulator",
    "images",
    "dataset_cropped",
)

class AttitudeDeterminationGUI:
    def __init__(self):
        self.selected_images = []
        self.metadata_list = []
        self.results = []
        self.case_listboxes = {}
        self.using_cropped_dataset = False
        self.dataset_dir = self._resolve_dataset_dir()
        self.load_metadata()
        self.create_main_window()

    def _resolve_dataset_dir(self):
        """Determine which dataset directory to use for image loading."""
        if os.path.isdir(CROPPED_DATASET_DIR):
            self.using_cropped_dataset = True
            print(f"Using cropped dataset: {CROPPED_DATASET_DIR}")
            return CROPPED_DATASET_DIR

        self.using_cropped_dataset = False
        print(f"Cropped dataset not found, falling back to: {DEFAULT_DATASET_DIR}")
        return DEFAULT_DATASET_DIR

    def load_metadata(self):
        """Load the metadata file"""
        try:
            metadata_path = os.path.join(project_root, "Maya_project", "IrishSat_Simulator", "images", "metadata", "metadata.jsonl")
            with open(metadata_path, 'r') as f:
                for line in f:
                    self.metadata_list.append(json.loads(line.strip()))
                    # Replace tag with tag - 1
                    self.metadata_list[-1]['tag'] = self.metadata_list[-1]['tag'] - 1
            print(f"Loaded {len(self.metadata_list)} metadata entries")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load metadata: {e}")

    def create_main_window(self):
        """Create the main GUI window"""
        self.root = tk.Tk()
        self.root.title("Attitude Determination Testing GUI")
        self.root.geometry("1100x700")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="EHS Image Attitude Determination Analysis",
                 font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Selection frame
        selection_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="10")
        selection_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        selection_frame.columnconfigure(0, weight=3)
        selection_frame.columnconfigure(1, weight=1)

        dataset_relative = os.path.relpath(self.dataset_dir, project_root)
        dataset_note = "Using cropped dataset" if self.using_cropped_dataset else "Using original dataset"
        dataset_text = f"Dataset: {dataset_relative} ({dataset_note})"
        self.dataset_info_var = tk.StringVar(value=dataset_text)
        ttk.Label(selection_frame, textvariable=self.dataset_info_var, font=("Arial", 10, "italic")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5)
        )

        # Attitude filters
        ttk.Label(selection_frame, text="Filter by Attitude:").grid(row=1, column=0, sticky=tk.W)

        filter_frame = ttk.Frame(selection_frame)
        filter_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5, 10))

        euler_rolls = []
        euler_pitches = []
        euler_yaws = []
        for r in self.metadata_list:
            euler_rolls.append(r.get("roll", 0))
            euler_pitches.append(r.get("pitch", 0))
            euler_yaws.append(r.get("yaw", 0))

        ttk.Label(filter_frame, text="Roll:").grid(row=0, column=0, padx=(0, 5))
        self.roll_var = tk.StringVar(value="Any")
        roll_combo = ttk.Combobox(filter_frame, textvariable=self.roll_var, width=8)
        roll_combo['values'] = ["Any"] + sorted(list(set(euler_rolls)))
        roll_combo.grid(row=0, column=1, padx=(0, 15))

        ttk.Label(filter_frame, text="Pitch:").grid(row=0, column=2, padx=(0, 5))
        self.pitch_var = tk.StringVar(value="Any")
        pitch_combo = ttk.Combobox(filter_frame, textvariable=self.pitch_var, width=8)
        pitch_combo['values'] = ["Any"] + sorted(list(set(euler_pitches)))
        pitch_combo.grid(row=0, column=3, padx=(0, 15))

        ttk.Label(filter_frame, text="Yaw:").grid(row=0, column=4, padx=(0, 5))
        self.yaw_var = tk.StringVar(value="Any")
        yaw_combo = ttk.Combobox(filter_frame, textvariable=self.yaw_var, width=8)
        yaw_combo['values'] = ["Any"] + sorted(list(set(euler_yaws)))
        yaw_combo.grid(row=0, column=5)

        ttk.Button(filter_frame, text="Apply Filter",
                  command=self.apply_filter).grid(row=0, column=6, padx=(15, 0))

        # Image list
        list_frame = ttk.Frame(selection_frame)
        list_frame.grid(row=3, column=0, columnspan=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))

        # Listbox with scrollbar
        self.image_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=8)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=scrollbar.set)

        self.image_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Selection buttons
        button_frame = ttk.Frame(selection_frame)
        button_frame.grid(row=4, column=0, pady=(10, 0))

        ttk.Button(button_frame, text="Select All",
                  command=self.select_all).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="Clear Selection",
                  command=self.clear_selection).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_frame, text="Analyze Selected",
                  command=self.analyze_selected).grid(row=0, column=2, padx=(0, 10))

        # Case selection columns
        case_frame = ttk.Frame(selection_frame)
        case_frame.grid(row=3, column=1, rowspan=2, sticky=(tk.N, tk.S, tk.E), pady=(10, 0), padx=(10, 0))
        for idx, case_id in enumerate(sorted(CASES.keys())):
            case_column = ttk.Frame(case_frame)
            case_column.grid(row=0, column=idx, sticky=tk.N, padx=(0, 10))
            ttk.Label(case_column, text=f"Case {case_id}", font=("Arial", 10, "bold")).grid(row=0, column=0)
            listbox = tk.Listbox(case_column, height=8, width=10, exportselection=False, activestyle='none', justify=tk.CENTER)
            listbox.grid(row=1, column=0, pady=(5, 5))
            listbox.bind("<Button-1>", lambda event: "break")
            listbox.bind("<Key>", lambda event: "break")
            ttk.Button(case_column, text="Select", command=lambda cid=case_id: self.select_case(cid)).grid(row=2, column=0)
            self.case_listboxes[case_id] = listbox
            case_frame.columnconfigure(idx, weight=1)

        # Analysis options
        options_frame = ttk.LabelFrame(main_frame, text="Analysis Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(options_frame, text="Polynomial Degree:").grid(row=0, column=0, padx=(0, 10))
        self.degree_var = tk.StringVar(value="1")
        degree_combo = ttk.Combobox(options_frame, textvariable=self.degree_var, width=8)
        degree_combo['values'] = ["auto", "1", "2", "3"]
        degree_combo.grid(row=0, column=1, padx=(0, 20))

        self.show_individual = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Individual Results",
                       variable=self.show_individual).grid(row=0, column=2, padx=(0, 20))

        self.show_combined = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Combined Statistics",
                       variable=self.show_combined).grid(row=0, column=3)

        # Results display area
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Text widget with scrollbar for results
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=15)
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)

        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Configure main grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.apply_filter()

    def apply_filter(self):
        """Filter images based on selected criteria"""
        self.image_listbox.delete(0, tk.END)

        roll_filter = self.roll_var.get()
        pitch_filter = self.pitch_var.get()
        yaw_filter = self.yaw_var.get()

        filtered_records = []
        for record in self.metadata_list:

            if (roll_filter == "Any" or record.get("roll", 0) == int(roll_filter)) and \
               (pitch_filter == "Any" or record.get("pitch", 0) == int(pitch_filter)) and \
               (yaw_filter == "Any" or record.get("yaw", 0) == int(yaw_filter)):
                filtered_records.append(record)

        for record in filtered_records:

            display_text = f"S{record['tag']}: R={record.get('roll', 0)}° P={record.get('pitch', 0)}° Y={record.get('yaw', 0)}°"
            self.image_listbox.insert(tk.END, display_text)

        self.filtered_records = filtered_records
        self.update_case_display()
        print(f"Filtered to {len(filtered_records)} image pairs")

    def select_all(self):
        """Select all items in the listbox"""
        self.image_listbox.select_set(0, tk.END)

    def clear_selection(self):
        """Clear all selections"""
        self.image_listbox.selection_clear(0, tk.END)

    def select_case(self, case_id):
        """Select all entries in the main list that belong to the specified case"""
        case_tags = set(CASES.get(case_id, []))
        if not case_tags:
            messagebox.showwarning("Case Selection", f"Case {case_id} has no defined samples.")
            return

        self.image_listbox.selection_clear(0, tk.END)
        selected_any = False
        for idx, record in enumerate(self.filtered_records):
            if record['tag'] in case_tags:
                self.image_listbox.select_set(idx)
                if not selected_any:
                    self.image_listbox.see(idx)
                selected_any = True

        if not selected_any:
            messagebox.showinfo("Case Selection", f"No images from Case {case_id} match the current filter.")

    def update_case_display(self):
        """Update the per-case sample listings based on the current filter"""
        if not self.case_listboxes:
            return

        visible_tags = {record['tag'] for record in getattr(self, 'filtered_records', [])}
        for case_id, listbox in self.case_listboxes.items():
            listbox.delete(0, tk.END)
            for tag in CASES.get(case_id, []):
                suffix = "" if tag in visible_tags else " (filtered)"
                listbox.insert(tk.END, f"S{tag}{suffix}")

    def analyze_selected(self):
        """Analyze the selected images"""
        selected_indices = self.image_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one image pair to analyze.")
            return

        selected_records = [self.filtered_records[i] for i in selected_indices]

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Analyzing {len(selected_records)} image pairs...\n\n")
        self.root.update()

        # Analyze each selected image pair
        all_results = []
        degree = self.degree_var.get()
        if degree != "auto":
            degree = int(degree)

        for i, record in enumerate(selected_records):
            self.results_text.insert(tk.END, f"Processing pair {i+1}/{len(selected_records)}...\n")
            self.root.update()

            # Load images - fix filename format mismatch
            # Metadata has: cam1_s0_roll0_pitch0_yaw0_1.png
            # Actual files: s0_roll0_pitch0_yaw0_cam1_1.png

            # Convert metadata filename to actual filename format
            original_cam1 = record["image1"]  # cam1_s0_roll0_pitch0_yaw0_1.png
            original_cam2 = record["image2"]  # cam2_s0_roll0_pitch0_yaw0_1.png

            # Remove "cam1_" or "cam2_" prefix and "_1.png" suffix, then rebuild
            cam1_middle = original_cam1.replace("cam1_", "").replace("_1.png", "")  # s0_roll0_pitch0_yaw0
            cam2_middle = original_cam2.replace("cam2_", "").replace("_1.png", "")  # s0_roll0_pitch0_yaw0

            actual_cam1 = f"{cam1_middle}_cam1_1.png"  # s0_roll0_pitch0_yaw0_cam1_1.png
            actual_cam2 = f"{cam2_middle}_cam2_1.png"  # s0_roll0_pitch0_yaw0_cam2_1.png

            cam1_path = os.path.join(self.dataset_dir, actual_cam1)
            cam2_path = os.path.join(self.dataset_dir, actual_cam2)

            self.results_text.insert(tk.END, f"  Metadata filename: {record['image1']}\n")
            self.results_text.insert(tk.END, f"  Actual filename: {actual_cam1}\n")
            self.results_text.insert(tk.END, f"  Full path: {cam1_path}\n")

            if not os.path.exists(cam1_path) or not os.path.exists(cam2_path):
                self.results_text.insert(tk.END, f"  WARNING: Images not found for tag {record['tag']}\n")
                self.results_text.insert(tk.END, f"    Cam1 exists: {os.path.exists(cam1_path)}\n")
                self.results_text.insert(tk.END, f"    Cam2 exists: {os.path.exists(cam2_path)}\n")
                continue

            try:
                self.results_text.insert(tk.END, f"  Loading images...\n")
                image1 = cv2.imread(cam1_path)
                image2 = cv2.imread(cam2_path)

                if image1 is None or image2 is None:
                    self.results_text.insert(tk.END, f"  ERROR: Failed to load images (None returned)\n")
                    continue

                self.results_text.insert(tk.END, f"  Processing images with degree={degree}...\n")
                # Store results in mag_sat object (simulates struct of data we'll actually be passing in)
                mag_sat = Magnetorquer_Sat(CUBESAT_BODY_INERTIA, [], VELOCITY_INITIAL, CONSTANT_B_FIELD_MAG, np.array([0,0,0]), DT, GYRO_WORKING, KP, KD)

                # Process images
                image1_results, image2_results = mag_sat.update_images(image1=image1, image2=image2, degree=degree)

                # Get both quaternion and euler results from attitude determination
                determined_vector, determined_angles, determined_quat, vector_angle_error = determine_attitude(mag_sat)

                self.results_text.insert(tk.END, f"  Results: Cam1={image1_results}, Cam2={image2_results}\n")

                truth_euler = [record.get("roll", 0), record.get("pitch", 0), record.get("yaw", 0)]
                if "quaternion" in record:
                    truth_quat = np.array(record["quaternion"])
                else:
                    # Fallback: convert euler to quaternion
                    truth_quat = euler_to_quaternion(truth_euler[0], truth_euler[1], truth_euler[2])

                # If we don't have a valid analysis implemented yet, skip this image
                if determined_quat is None or determined_angles is None:
                    self.results_text.insert(tk.END, f"Image case not implemented yet, skipping\n")
                    continue

                # Fix outputed euler angles to place in same format as stored truth
                # For example, -50 should be 310
                determined_angles = [(angle + 360) % 360 for angle in determined_angles]

                analysis_result = {
                    'tag': record['tag'],
                    'truth': {'roll': truth_euler[0], 'pitch': truth_euler[1], 'yaw': truth_euler[2], 'quaternion': truth_quat, 'invariant_angle': record.get("yaw_invarient_degrees", 0)},
                    'found': {'roll': determined_angles[0], 'pitch': determined_angles[1], 'yaw': determined_angles[2], 'quaternion': determined_quat, 'invariant_angle': vector_angle_error, 'vector': determined_vector},
                    'cam1': {'roll': image1_results[0], 'pitch': image1_results[1], 'alpha': image1_results[2], 'edges': image1_results[3]},
                    'cam2': {'roll': image2_results[0], 'pitch': image2_results[1], 'alpha': image2_results[2], 'edges': image2_results[3]}
                }
                all_results.append(analysis_result)
                self.results_text.insert(tk.END, f"  SUCCESS: Added result for tag {record['tag']}\n")

            except Exception as e:
                self.results_text.insert(tk.END, f"  ERROR processing tag {record['tag']}: {e}\n")
                import traceback
                self.results_text.insert(tk.END, f"  Traceback: {traceback.format_exc()}\n")

        self.results_text.insert(tk.END, f"\nFINAL: Processed {len(all_results)} successful results out of {len(selected_records)} attempts\n\n")

        self.print_results(all_results, selected_records)
        self.create_graphs(all_results)


    def print_results(self, results, selected_records):
        """Display analysis results in the text widget"""
        # Don't clear the text - keep debug output
        # self.results_text.delete(1.0, tk.END)

        self.results_text.insert(tk.END, f"DEBUG: print_results called with {len(results)} results\n")

        if not results:
            self.results_text.insert(tk.END, "No valid results to display.\n")
            return

        self.results_text.insert(tk.END, f"ATTITUDE DETERMINATION ANALYSIS RESULTS\n")
        self.results_text.insert(tk.END, f"{'='*60}\n\n")

        # Individual results
        if self.show_individual.get():
            self.results_text.insert(tk.END, "INDIVIDUAL RESULTS:\n")
            self.results_text.insert(tk.END, f"{'-'*40}\n")

            for result in results:
                truth = result['truth']
                found = result['found']

                self.results_text.insert(tk.END, f"Tag {result['tag']}:\n")
                self.results_text.insert(tk.END, f"  Truth: R={truth['roll']:.1f}° P={truth['pitch']:.1f}° Y={truth['yaw']:.1f}°\n")
                self.results_text.insert(tk.END, f"  Found: R={found['roll']:.1f}° P={found['pitch']:.1f}° Y={found['yaw']:.1f}°\n")

                # Show quaternion-based angular error
                if 'quaternion' in truth and 'quaternion' in found:
                    ang_error = quaternion_difference(truth['quaternion'], found['quaternion'])
                    self.results_text.insert(tk.END, f"  Quaternion Error: {ang_error:.2f}°\n")

                truth_inv = truth.get('invariant_angle', 0.0)
                found_inv = found.get('invariant_angle', 0.0)
                self.results_text.insert(tk.END, f"  Invariant Angle: Found={found_inv:.2f}° Truth={truth_inv:.2f}° | Δ={abs(found_inv - truth_inv):.2f}°\n")

                self.results_text.insert(tk.END, f"  Cam1:  R={result['cam1']['roll']:.1f}° P={result['cam1']['pitch']:.1f}° α={result['cam1']['alpha']:.3f}\n")
                self.results_text.insert(tk.END, f"  Cam2:  R={result['cam2']['roll']:.1f}° P={result['cam2']['pitch']:.1f}° α={result['cam2']['alpha']:.3f}\n")
                self.results_text.insert(tk.END, "\n")

        # Combined statistics
        if self.show_combined.get():
            self.results_text.insert(tk.END, f"\nCOMBINED STATISTICS:\n")
            self.results_text.insert(tk.END, f"{'-'*40}\n")

            self.results_text.insert(tk.END, f"Sample Size: {len(results)} image pairs\n\n")

            # Find the average difference between truth and found results (Euler angles)
            roll_diffs = [abs(r['truth']['roll'] - r['found']['roll']) for r in results]
            pitch_diffs = [abs(r['truth']['pitch'] - r['found']['pitch']) for r in results]
            yaw_diffs = [abs(r['truth']['yaw'] - r['found']['yaw']) for r in results]
            avg_roll_diff = np.mean(roll_diffs)
            avg_pitch_diff = np.mean(pitch_diffs)
            avg_yaw_diff = np.mean(yaw_diffs)
            self.results_text.insert(tk.END, f"Average Absolute Differences (Euler Angles):\n")
            self.results_text.insert(tk.END, f"  Roll:  {avg_roll_diff:.2f}°\n")
            self.results_text.insert(tk.END, f"  Pitch: {avg_pitch_diff:.2f}°\n")
            self.results_text.insert(tk.END, f"  Yaw:   {avg_yaw_diff:.2f}°\n\n")

            truth_inv_values = [r['truth'].get('invariant_angle', 0.0) for r in results]
            found_inv_values = [r['found'].get('invariant_angle', 0.0) for r in results]
            inv_diffs = [abs(f - t) for f, t in zip(found_inv_values, truth_inv_values)]
            self.results_text.insert(tk.END, f"Invariant Angle Statistics:\n")
            self.results_text.insert(tk.END, f"  Average Truth: {np.mean(truth_inv_values):.2f}°\n")
            self.results_text.insert(tk.END, f"  Average Found: {np.mean(found_inv_values):.2f}°\n")
            self.results_text.insert(tk.END, f"  Avg Abs Difference: {np.mean(inv_diffs):.2f}°\n")
            self.results_text.insert(tk.END, f"  Max Abs Difference: {np.max(inv_diffs):.2f}°\n\n")


    def create_graphs(self, results):
        '''
        Given a list of JSON results from attitude determination analysis, create 3D and line plots to show how good our nadir guesses were
        '''
        if not results:
            self.results_text.insert(tk.END, "DEBUG: No results for visualization\n")
            return

        self.results_text.insert(tk.END, f"DEBUG: Creating visualization for {len(results)} results\n")

        # Import and display the unsupported case count
        from nadir_point import get_unsupported_case_count
        unsupported_count = get_unsupported_case_count()
        
        # Display the count in the GUI results text
        self.results_text.insert(tk.END, f"\n{'='*60}\n")
        self.results_text.insert(tk.END, f"UNSUPPORTED CASE COUNT: {unsupported_count}\n")
        self.results_text.insert(tk.END, f"This represents how many times the code went to the 'else' statement\n")
        self.results_text.insert(tk.END, f"in the determine_attitude elif tree (cases that need implementation)\n")
        self.results_text.insert(tk.END, f"{'='*60}\n\n")

        try:
            # compare our nadir guess to the truth values and output graphs/statistics

            # 3D vector comparison: draw simple satellite cube, then draw lines for truth and found nadir directions
            # Create in a standalone window if only 1 image was analyzed, otherwise skip
            if len(results) == 1:
                plt.figure()
                ax_3d = plt.axes(projection='3d')
                ax_3d.set_box_aspect([1, 1, 1])
                ax_3d.set_title('Nadir Guess vs Actual')
                ax_3d.set_xlabel('X (Roll)')
                ax_3d.set_ylabel('Y (Pitch)')
                ax_3d.set_zlabel('Z (Yaw)')
                ax_3d.set_xlim([-1, 1])
                ax_3d.set_ylim([-1, 1])
                ax_3d.set_zlim([-1, 1])
                ax_3d.set_xticks([-1, 1])
                ax_3d.set_yticks([-1, 1])
                ax_3d.set_zticks([-1, 1])
                ax_3d.grid(True)
                # Draw satellite cube
                r = [-0.1, 0.1]
                for s, e in combinations(np.array(list(product(r, r, r))), 2):
                    if np.sum(np.abs(s-e)) == r[1]-r[0]:
                        ax_3d.plot3D(*zip(s, e), color="b")

                # Place two cameras on +/- x near the bottom face, pointing outward along +/- x and pitched 30° down
                bottom_z = -0.1
                cam_offset_x = -0.01
                pitch_down_rad = np.radians(30)
                dir_pos_x = np.array([np.cos(pitch_down_rad), 0.0, -np.sin(pitch_down_rad)])
                origin_pos = np.array([cam_offset_x, 0.0, bottom_z])
                draw_camera_pyramid(ax_3d, origin_pos, dir_pos_x, length=0.28, width=0.08, color='orange', alpha=0.5)
                dir_neg_x = np.array([-np.cos(pitch_down_rad), 0.0, -np.sin(pitch_down_rad)])
                origin_neg = np.array([-cam_offset_x, 0.0, bottom_z])
                draw_camera_pyramid(ax_3d, origin_neg, dir_neg_x, length=0.28, width=0.08, color='orange', alpha=0.5)

                # Draw truth nadir direction (red)
                truth_vector = euler_to_vector([results[0]['truth']['roll'], results[0]['truth']['pitch'], results[0]['truth']['yaw']])
                ax_3d.quiver(0, 0, 0, truth_vector[0], truth_vector[1], truth_vector[2], color='b', label='Truth', arrow_length_ratio=0.1)

                # Draw found nadir direction (green)
                # found_vector = euler_to_vector([results[0]['found']['roll'], results[0]['found']['pitch'], results[0]['found']['yaw']])
                found_vector = results[0]['found']['vector']
                ax_3d.quiver(0, 0, 0, found_vector[0], found_vector[1], found_vector[2], color='r', label='Determined', arrow_length_ratio=0.1)

                # Draw example euler angles (for reference) (blue)
                example = [0, 0, 0]
                example_vector = euler_to_vector(example)
                ax_3d.quiver(0, 0, 0, example_vector[0], example_vector[1], example_vector[2], color='g', label='Example', arrow_length_ratio=0.1)

                ax_3d.legend()

            else:
                # 3 graphs (1 for roll, pitch, yaw) showing 2 lines each (guess vs truth)
                fig, axs = plt.subplots(2, 3, figsize=(14, 7))
                sample_tags = [r['tag'] for r in results]
                sample_indices = list(range(len(results)))
                axes_labels = ['roll', 'pitch', 'yaw']
                for i, key in enumerate(axes_labels):
                    ax = axs[0, i]
                    if len(results) == 1:
                        tag_idx = sample_indices[0]
                        truth_val = results[0]['truth'][key]
                        found_val = results[0]['found'][key]
                        ax.plot([tag_idx], [truth_val], label='Truth', color='b', marker='o')
                        ax.plot([tag_idx], [found_val], label='Determined', color='r', marker='o')
                    else:
                        truth_vals = [r['truth'][key] for r in results]
                        found_vals = [r['found'][key] for r in results]
                        ax.plot(sample_indices, truth_vals, label='Truth', color='b', marker='o')
                        ax.plot(sample_indices, found_vals, label='Determined', color='r', marker='o')
                    ax.set_title(key.capitalize())
                    ax.set_xlabel('Sample Number')
                    ax.set_xticks(sample_indices)
                    ax.set_xticklabels(sample_tags)
                    ax.legend()

                # Plot yaw invariant angle (found vs truth)for each result found
                ax_inv = axs[1, 0]
                if len(results) == 1:
                    tag_idx = sample_indices[0]
                    ax_inv.plot([tag_idx], [results[0]['truth']['invariant_angle']], label='Truth', color='b', marker='o')
                    ax_inv.plot([tag_idx], [results[0]['found']['invariant_angle']], label='Determined', color='r', marker='o')
                else:
                    ax_inv.plot(sample_indices, [r['truth']['invariant_angle'] for r in results], label='Truth', color='b', marker='o')
                    ax_inv.plot(sample_indices, [r['found']['invariant_angle'] for r in results], label='Determined', color='r', marker='o')
                ax_inv.set_title('Yaw Invariant Angle')
                ax_inv.legend()
                ax_inv.set_xlabel('Sample Number')
                ax_inv.set_ylabel('Angle (degrees)')
                ax_inv.set_xticks(sample_indices)
                ax_inv.set_xticklabels(sample_tags)

                axs[1, 1].axis('off')
                axs[1, 2].axis('off')

                fig.tight_layout(rect=(0, 0, 1, 0.96))
            plt.show()

        except Exception as e:
            self.results_text.insert(tk.END, f"DEBUG: Error creating visualization: {e}\n")
            import traceback
            self.results_text.insert(tk.END, f"DEBUG: Traceback: {traceback.format_exc()}\n")

    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def launch_gui():
    """Launch the attitude determination GUI"""
    gui = AttitudeDeterminationGUI()
    gui.run()

def draw_camera_pyramid(ax_3d, origin, direction, length=0.25, width=0.12, color='orange', alpha=0.4):
    """Draw a simple pyramid representing a camera.

    origin: (x,y,z) base point on cube
    direction: unit vector pointing where the camera faces
    length: how far the pyramid extends
    width: base half-width of pyramid
    """
    # Tip at the cube (origin); base is outward along direction
    tip = np.array(origin)

    # Create a square base centered at origin + direction*length
    dir_vec = np.array(direction) / np.linalg.norm(direction)
    # Pick an arbitrary vector not colinear with dir_vec
    arb = np.array([0.0, 0.0, 1.0]) if abs(dir_vec[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v1 = np.cross(dir_vec, arb)
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = np.cross(dir_vec, v1)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)

    # Base center further out from origin
    base_center = np.array(origin) + dir_vec * length
    corners = [base_center + ( v1 * width +  v2 * width),
               base_center + (-v1 * width +  v2 * width),
               base_center + (-v1 * width + -v2 * width),
               base_center + ( v1 * width + -v2 * width)]

    # Build faces (4 triangular faces connecting tip to each base corner + base face)
    faces = []
    for c in corners:
        faces.append([tuple(tip), tuple(c), tuple(c)])
        # triangle faces should be tip->corner->next_corner, we'll build properly below
    # Build triangular faces explicitly (tip, corner_i, corner_{i+1})
    faces = []
    for i in range(4):
        c1 = corners[i]
        c2 = corners[(i+1) % 4]
        faces.append([tuple(tip), tuple(c1), tuple(c2)])
    # Add the base face (as quad)
    faces.append([tuple(c) for c in corners])

    poly = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha)
    ax_3d.add_collection3d(poly)

if __name__ == "__main__":
    print("Launching Attitude Determination GUI...")
    launch_gui()