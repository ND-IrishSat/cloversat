'''
Satellite Thermal Image Reconstructor
Professional interface for reconstructing 24x32 thermal images from satellite downlink data
Based on NearSpace satellite infrared camera system with automatic packet ordering
'''

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import os
from itertools import permutations
import re

# Satellite camera specifications (based on C++ code analysis)
RAW_WIDTH = 32   # MLX90640 thermal camera width
RAW_HEIGHT = 24  # MLX90640 thermal camera height
PIXELS_PER_IMAGE = RAW_WIDTH * RAW_HEIGHT  # 768 total pixels
CHUNKS_PER_IMAGE = 4  # Split into 4 transmission packets
PIXELS_PER_CHUNK = PIXELS_PER_IMAGE // CHUNKS_PER_IMAGE  # 192 pixels per packet

# Packet structure constants (based on C++ constructImageDownlink)
# SYNC_BYTES = [0x50, 0x50, 0x50]  # 3 sync bytes
# CMD_BYTE_INDEX = 3
# LENGTH_BYTE_INDEX = 4
# PACKET_SEQ_HIGH_INDEX = 5
# PACKET_SEQ_LOW_INDEX = 6
# PACKET_TYPE_INDEX = 7
# PACKET_ID_INDEX = 8  # Contains camera selection and chunk index
# IMAGE_NUM_INDEX = 9  # Image number for grouping chunks
# PAYLOAD_START_INDEX = 10  # Where image data begins
PACKET_SEQ_HIGH_INDEX = 0
PACKET_SEQ_LOW_INDEX = 1
PACKET_TYPE_INDEX = 2
PACKET_ID_INDEX = 3  # Contains camera selection and chunk index
IMAGE_NUM_INDEX = 4  # Image number for grouping chunks
PAYLOAD_START_INDEX = 5  # Where image data begins


BYTE_TOKEN_RE = re.compile(r'0[xX][0-9a-fA-F]{1,2}|[0-9a-fA-F]{1,2}|\d{1,3}')

# Optional default downlink packet data for quick GUI prefill. Update these
# strings with known packet dumps to auto-populate the input boxes.
DEFAULT_PACKET_STRINGS = [
    "00 04 34 60 01 37 3D 33 38 3C 30 24 2B 2B 29 21 20 1F 1D 15 1D 1B 1C 18 1D 1B 1A 2A 32 5C 67 6F 6F 68 67 69 69 35 36 34 39 36 2A 1D 26 2B 24 21 25 21 1D 1C 1C 1D 19 15 1D 1D 1A 28 27 54 65 73 71 68 65 66 74 38 3E 37 3B 2A 23 1B 20 23 24 25 23 26 23 1E 23 1F 1A 19 1B 16 15 1D 1B 44 59 68 6C 67 62 68 6D 39 3B 38 3D 2B 27 1B 26 23 24 27 2E 26 27 20 23 21 1E 18 1A 15 17 14 17 31 47 5F 6D 67 68 77 74 3E 3D 36 44 29 25 1C 25 27 26 29 2C 2D 30 2E 24 23 1E 1C 1D 18 1A 18 19 1F 2E 5D 67 82 7F 7B 80 41 37 3D 36 32 1F 24 26 28 23 30 34 31 31 31 31 23 24 22 21 18 15 18 1A 1D 1E 44 5A 79 75 82 7F",
    "00 03 34 40 01 2E 2A 1E 1C 1C 1E 22 21 1A 1A 13 15 14 0D 11 1E 88 C0 E1 E5 CC B6 86 7A 6D 70 69 6C 64 65 69 6A 2C 28 27 27 24 26 26 27 25 25 1D 16 12 12 1A 35 CE ED F8 F6 F5 F3 AC 91 73 6E 67 67 5F 63 66 68 2F 2D 26 24 22 29 2C 32 2E 25 1D 1C 17 15 1F 58 ED FF F4 F9 FC FB B9 9D 75 6D 6C 6D 72 6C 71 72 30 2F 31 29 2F 24 2D 2B 2E 2C 1F 1D 15 18 22 38 C9 E1 F7 F7 E5 D6 93 83 78 71 73 71 6F 6C 6E 74 2D 2F 2E 37 31 2C 31 2B 2A 28 1C 20 17 1E 15 1F 48 6B 7E 83 63 50 50 64 7D 7D 6F 73 64 64 62 66 32 30 36 34 35 30 2C 29 2A 29 1E 20 17 15 14 1D 29 30 46 41 34 2D 39 4D 73 77 70 6B 62 62 63 6C",
    "00 02 34 20 01 48 45 52 5E 81 5E 2C 30 25 25 05 07 06 0E 04 07 06 06 02 08 0B 05 0A 1A 4C 5B 5E 65 5F 5B 5F 71 41 40 57 71 94 78 31 32 2A 24 08 0A 09 0C 06 07 06 04 05 09 06 04 0C 1F 56 59 5B 63 59 60 68 6E 37 3D 4D 74 87 6E 31 29 19 11 05 0B 06 09 06 08 06 07 06 0C 06 08 09 1B 4F 5C 59 59 60 64 64 72 3D 3B 40 50 66 4F 2C 2C 16 11 09 08 0C 09 07 08 07 07 08 06 0A 08 10 1D 55 5B 5D 63 6B 65 6A 67 39 34 21 25 1F 28 1E 22 13 0E 0B 0A 0A 0D 04 0D 0D 0E 17 17 18 14 26 45 62 68 66 6F 64 64 65 6A 33 2E 23 1D 1F 1B 22 23 15 15 0A 11 10 0F 0A 0A 1F 2B 45 4A 36 2F 42 53 6B 6E 6C 6A 6C 66 62 68",
    "00 01 34 00 01 70 69 27 36 2D 2A 27 31 1D 17 17 1A 15 0D 07 0C 04 08 0B 0B 08 0F 0D 0F 1F 1A 5E 74 78 75 7F 92 5C 52 33 36 30 2E 2B 29 24 1E 21 1E 0F 0A 0A 0A 0B 0C 04 0C 0F 09 0D 0F 1B 1E 60 71 74 75 7F 8F 62 55 2E 35 31 28 17 1D 20 22 26 20 0C 0D 01 0A 08 0A 01 0E 0C 0A 0D 0F 1A 32 65 69 70 70 73 8A 6A 49 3F 3C 2C 2B 1E 1D 23 1C 1D 14 07 06 05 06 09 03 02 04 09 0E 09 0F 24 3B 6D 67 6C 65 79 85 3B 40 3A 44 26 20 12 24 28 27 0D 0C 07 08 00 05 07 09 07 0A 09 09 06 0F 33 46 5C 72 6D 65 62 7F 42 43 3D 42 33 23 1C 1E 22 1E 0D 0B 05 06 05 04 07 05 02 03 05 0A 06 10 4D 4E 56 5E 5E 53 63 77"
]


def decode_byte_stream(raw_data: str) -> list:
    """Convert an arbitrary hex/decimal string dump into a list of byte values."""
    if not raw_data:
        return []

    candidates = BYTE_TOKEN_RE.findall(raw_data)
    byte_values = []

    for token in candidates:
        token_clean = token.strip()
        if not token_clean:
            continue

        # Prefer hexadecimal interpretation (mirrors typical hexdumps).
        try:
            value = int(token_clean, 16)
            if value > 0xFF:
                raise ValueError  # fallback to decimal below
        except ValueError:
            # Fall back to decimal (handles dumps that mix formats).
            value = int(token_clean, 10)

        if not 0 <= value <= 0xFF:
            raise ValueError(f"Token '{token_clean}' is not a valid byte value (0-255)")

        byte_values.append(value)

    return byte_values


def parse_packet_header(packet_bytes):
    """
    Parse satellite packet header based on actual C++ constructImageDownlink implementation
    """
    try:
        if len(packet_bytes) < PAYLOAD_START_INDEX:
            return None

        # # Check for valid sync bytes (0x50 0x50 0x50)
        # has_sync = packet_bytes[:3] == SYNC_BYTES
        # if not has_sync:
        #     print(f"Warning: Invalid sync bytes: {packet_bytes[:3]} (expected {SYNC_BYTES})")
        has_sync = True

        # Extract packet type (byte 7) - should be IMAGE type
        packet_type = packet_bytes[PACKET_TYPE_INDEX] if len(packet_bytes) > PACKET_TYPE_INDEX else 0

        # Extract packet ID (byte 8) - contains camera and chunk info
        packet_id = packet_bytes[PACKET_ID_INDEX] if len(packet_bytes) > PACKET_ID_INDEX else 0

        # Parse packet ID based on C++ implementation:
        # From C++: (sendCam1 ? 0x00 : 0x80) | static_cast<uint8_t>((chunkIndex & 0x03) << 5)
        # So: bit7 = camera, bits6-5 = chunk index (0-3)
        camera_selection = (packet_id & 0x80) >> 7  # bit 7: 0=cam1, 1=cam2
        chunk_index = (packet_id & 0x60) >> 5       # bits 6-5: chunk index (0-3)

        # Extract image number (byte 9)
        image_number = packet_bytes[IMAGE_NUM_INDEX] if len(packet_bytes) > IMAGE_NUM_INDEX else 0

        # Extract packet sequence number (bytes 5-6)
        packet_seq_num = 0
        if len(packet_bytes) > PACKET_SEQ_LOW_INDEX:
            packet_seq_num = (packet_bytes[PACKET_SEQ_HIGH_INDEX] << 8) | packet_bytes[PACKET_SEQ_LOW_INDEX]

        return {
            'chunk_index': chunk_index,
            'camera_selection': camera_selection,  # 0=cam1, 1=cam2
            'image_number': image_number,
            'packet_seq_num': packet_seq_num,
            'packet_type': packet_type,
            'payload_start': PAYLOAD_START_INDEX,
            'has_valid_header': has_sync
        }

    except (ValueError, IndexError) as e:
        print(f"Error parsing packet header: {e}")
        return None


def extract_image_payload(packet_bytes, header_info):
    """
    Extract image payload data from satellite packet
    """
    try:
        # Extract payload starting from the payload index
        payload_start = header_info['payload_start']
        payload = list(packet_bytes[payload_start:payload_start + PIXELS_PER_CHUNK])

        if len(payload) < PIXELS_PER_CHUNK:
            missing = PIXELS_PER_CHUNK - len(payload)
            print(f"Payload shorter than expected ({len(payload)} < {PIXELS_PER_CHUNK}); padding {missing} zeros")
            payload.extend([0] * missing)

        print(f"Extracted {len(payload)} payload bytes from chunk {header_info['chunk_index']}")
        return payload

    except (ValueError, IndexError) as e:
        print(f"Error extracting payload: {e}")
        # Fallback: treat entire string as payload
        fallback = list(packet_bytes[:PIXELS_PER_CHUNK])
        if len(fallback) < PIXELS_PER_CHUNK:
            fallback.extend([0] * (PIXELS_PER_CHUNK - len(fallback)))
        return fallback


def auto_order_satellite_chunks(chunk_data_list):
    """
    Automatically order satellite chunks based on embedded packet information
    """
    chunk_info = []

    # Parse header information for each chunk
    for i, chunk_hex in enumerate(chunk_data_list):
        if not chunk_hex.strip():
            chunk_info.append({
                'original_index': i,
                'chunk_index': i,  # Default ordering
                'payload': [0] * PIXELS_PER_CHUNK,  # Empty chunk
                'has_valid_header': False,
                'camera_selection': 0,
                'image_number': 0
            })
            continue

        try:
            packet_bytes = decode_byte_stream(chunk_hex)
        except ValueError as exc:
            print(f"Chunk {i+1}: Failed to parse byte stream ({exc})")
            packet_bytes = []

        if not packet_bytes:
            chunk_info.append({
                'original_index': i,
                'chunk_index': i,
                'payload': [0] * PIXELS_PER_CHUNK,
                'has_valid_header': False,
                'camera_selection': 0,
                'image_number': 0
            })
            continue

        header = parse_packet_header(packet_bytes)

        if header is None or not header['has_valid_header']:
            # No valid header found, treat as raw payload data
            print(f"Chunk {i+1}: No valid header, treating as raw payload")
            payload = list(packet_bytes[:PIXELS_PER_CHUNK])
            if len(payload) < PIXELS_PER_CHUNK:
                payload.extend([0] * (PIXELS_PER_CHUNK - len(payload)))
            chunk_info.append({
                'original_index': i,
                'chunk_index': i,  # Default ordering
                'payload': payload,
                'has_valid_header': False,
                'camera_selection': 0,
                'image_number': 0
            })
        else:
            # Valid header found, extract payload
            payload = extract_image_payload(packet_bytes, header)
            chunk_info.append({
                'original_index': i,
                'chunk_index': header['chunk_index'],
                'payload': payload,
                'has_valid_header': header['has_valid_header'],
                'camera_selection': header['camera_selection'],
                'image_number': header['image_number'],
                'packet_seq_num': header.get('packet_seq_num', 0)
            })

            print(f"Chunk {i+1}: Header parsed - Chunk index: {header['chunk_index']}, "
                  f"Camera: {'cam2' if header['camera_selection'] else 'cam1'}, "
                  f"Image: {header['image_number']}")

    # Check if we have valid headers for automatic ordering
    valid_headers = [info for info in chunk_info if info['has_valid_header']]

    if len(valid_headers) >= 2:
        print(f"🎯 Found {len(valid_headers)} chunks with valid headers - using automatic ordering")
        # Sort by embedded chunk index
        chunk_info.sort(key=lambda x: x['chunk_index'])
        ordering_method = "automatic_header_based"
    else:
        print(f"⚠️ Only {len(valid_headers)} chunks have valid headers - using input order")
        # Keep original order
        chunk_info.sort(key=lambda x: x['original_index'])
        ordering_method = "input_order_fallback"

    # Combine payloads in the determined order
    ordered_pixels = []
    chunk_order = []

    for info in chunk_info:
        payload = info['payload']
        # Pad or trim to correct size
        if len(payload) < PIXELS_PER_CHUNK:
            payload.extend([0] * (PIXELS_PER_CHUNK - len(payload)))
        elif len(payload) > PIXELS_PER_CHUNK:
            payload = payload[:PIXELS_PER_CHUNK]

        ordered_pixels.extend(payload)
        chunk_order.append({
            'original_position': info['original_index'] + 1,
            'chunk_index': info['chunk_index'],
            'camera': 'cam2' if info['camera_selection'] else 'cam1',
            'has_header': info['has_valid_header']
        })

    ordering_info = {
        'method': ordering_method,
        'chunk_order': chunk_order,
        'total_pixels': len(ordered_pixels),
        'valid_headers': len(valid_headers),
        'same_image': len(set(info['image_number'] for info in valid_headers)) <= 1 if valid_headers else True
    }

    return ordered_pixels, ordering_info


def bytes_to_image(hex_string, width=32, height=24, output_path=None, display=True):
    """
    Convert hex string to thermal grayscale image
    Based on C++ temp_to_image function implementation

    Args:
        hex_string (str): Space-separated hex values from satellite downlink
        width (int): Image width in pixels (default: 32 for MLX90640)
        height (int): Image height in pixels (default: 24 for MLX90640)
        output_path (str, optional): Path to save the image
        display (bool): Whether to display the image using matplotlib

    Returns:
        numpy.ndarray: The reconstructed thermal image as a numpy array
        dict: Statistics about the thermal image
    """

    # Split hex string and convert to integers
    hex_values = hex_string.strip().split()
    pixel_values = [int(val, 16) for val in hex_values]

    print(f"Total pixels in satellite data: {len(pixel_values)}")
    print(f"Expected pixels for thermal image ({width}×{height}): {width * height}")
    print(f"Sample thermal values: {pixel_values[:10]}")

    # Ensure we have exactly the right amount of data
    expected_size = width * height
    if len(pixel_values) < expected_size:
        # Pad with zeros if not enough data
        pixel_values.extend([0] * (expected_size - len(pixel_values)))
        print(f"Warning: Padded {expected_size - len(pixel_values)} pixels with zeros")
    elif len(pixel_values) > expected_size:
        # Trim if too much data
        pixel_values = pixel_values[:expected_size]
        print(f"Warning: Trimmed to {expected_size} pixels")

    # Convert to numpy array and reshape to image dimensions
    image_array = np.array(pixel_values, dtype=np.uint8)

    # Reshape based on how the C++ code stores the image data
    image = image_array.reshape(height, width)  # 24 rows, 32 columns

    # Calculate thermal image statistics
    stats = {
        'total_pixels': len(pixel_values),
        'min_temp': np.min(image),
        'max_temp': np.max(image),
        'mean_temp': np.mean(image),
        'hot_pixels': np.count_nonzero(image > 128),  # "Hot" threshold
        'thermal_range': np.max(image) - np.min(image)
    }

    # Display thermal image statistics
    print(f"\nThermal Image Statistics:")
    print(f"Shape: {image.shape} (height×width)")
    print(f"Temperature range: {stats['min_temp']}-{stats['max_temp']}")
    print(f"Mean temperature: {stats['mean_temp']:.2f}")
    print(f"Hot pixels (>128): {stats['hot_pixels']}")
    print(f"Thermal range: {stats['thermal_range']}")

    # Save image if path provided
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Thermal image saved to: {output_path}")

    # Display image using matplotlib with thermal colormap
    if display:
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap='hot', vmin=0, vmax=255, interpolation='nearest')
        plt.title(f'Satellite Thermal Image ({width}×{height} pixels)')
        plt.colorbar(label='Temperature Intensity')
        plt.axis('off')
        plt.show()

    return image, stats


class SatelliteThermalImageGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Satellite Thermal Image Reconstructor - NearSpace Project")
        self.root.geometry("1200x900")
        self.root.configure(bg='#f0f0f0')

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Current image data
        self.current_image = None
        self.combination_images = []
        self.selected_combination = None
        self.last_ordering_info = None

        self.create_gui()
        self.load_default_inputs(initial_load=True)

    def create_gui(self):
        # Main container with scrollbar
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="🛰️ Satellite Thermal Image Reconstructor",
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))

        subtitle_label = ttk.Label(main_frame,
                                  text="Reconstruct 32×24 thermal images from 4-part satellite downlink data\n" +
                                       "Automatic packet ordering based on embedded header information",
                                  font=('Arial', 10), foreground='gray')
        subtitle_label.pack(pady=(0, 20))

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="📡 Satellite Downlink Data (4 Chunks)", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 20))

        # Instruction label
        instruction_label = ttk.Label(input_frame,
                                    text="Paste satellite packet data below. System will automatically detect chunk order from headers:")
        instruction_label.pack(anchor=tk.W, pady=(0, 10))

        # Create 4 input boxes in a 2x2 grid
        parts_frame = ttk.Frame(input_frame)
        parts_frame.pack(fill=tk.X, pady=(0, 10))

        self.hex_inputs = []
        chunk_labels = ["Packet 1", "Packet 2", "Packet 3", "Packet 4"]

        for i in range(4):
            row = i // 2
            col = i % 2

            chunk_frame = ttk.LabelFrame(parts_frame, text=chunk_labels[i], padding="5")
            chunk_frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

            text_widget = scrolledtext.ScrolledText(chunk_frame, height=6, width=40, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True)
            self.hex_inputs.append(text_widget)

        # Configure grid weights
        parts_frame.columnconfigure(0, weight=1)
        parts_frame.columnconfigure(1, weight=1)
        parts_frame.rowconfigure(0, weight=1)
        parts_frame.rowconfigure(1, weight=1)

        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(pady=(10, 0))

        # Control buttons
        control_frame = ttk.Frame(button_frame)
        control_frame.pack(side=tk.LEFT)

        # Load defaults button
        self.defaults_button = ttk.Button(control_frame, text="📋 Load Default Packets",
                          command=self.load_default_inputs)
        self.defaults_button.pack(pady=2)

        # Auto reconstruction button (primary method)
        self.auto_button = ttk.Button(control_frame, text="🎯 Auto Reconstruct Image",
                                    command=self.auto_reconstruction)
        self.auto_button.pack(pady=2)

        # Manual combinations button (fallback)
        self.generate_button = ttk.Button(control_frame, text="🔄 Try All Combinations",
                                        command=self.generate_combinations)
        self.generate_button.pack(pady=2)

        # Clear button
        clear_button = ttk.Button(control_frame, text="🗑️ Clear All", command=self.clear_all_inputs)
        clear_button.pack(pady=2)

        # Save button (initially disabled)
        self.save_button = ttk.Button(control_frame, text="💾 Save Thermal Image",
                                    command=self.save_image, state='disabled')
        self.save_button.pack(pady=2)

        # Results display section
        self.results_frame = ttk.LabelFrame(main_frame, text="🌡️ Thermal Image Results", padding="10")
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        self.show_results_placeholder()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("🚀 Ready - Paste satellite thermal image packets and auto-reconstruct")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W, font=('Arial', 9))
        status_bar.pack(fill=tk.X, pady=(10, 0))

        # Bind mouse wheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def show_results_placeholder(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        placeholder_label = ttk.Label(
            self.results_frame,
            text=("📋 Paste satellite downlink packet data above\n"
                  "🎯 Use 'Auto Reconstruct' - system will parse packet headers automatically\n"
                  "🔄 Use 'Try All Combinations' if auto-detection fails"),
            font=('Arial', 10),
            foreground='gray',
            justify=tk.CENTER
        )
        placeholder_label.pack(expand=True)

    def reset_processing_state(self):
        self.combination_images = []
        self.current_image = None
        self.selected_combination = None
        self.last_ordering_info = None
        self.save_button.config(state='disabled')

    def auto_reconstruction(self):
        """Automatically reconstruct image using packet header information"""
        try:
            # Get hex data from all 4 inputs
            chunks = []
            for i, text_widget in enumerate(self.hex_inputs):
                hex_data = text_widget.get(1.0, tk.END).strip()
                chunks.append(hex_data)

            # Check if we have at least some data
            if not any(chunk.strip() for chunk in chunks):
                messagebox.showwarning("No Input", "Please paste satellite downlink packet data")
                return

            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()

            self.status_var.set("🔄 Auto-analyzing packet headers and reconstructing thermal image...")
            self.root.update_idletasks()

            # Automatically order chunks based on packet headers
            ordered_pixels, ordering_info = auto_order_satellite_chunks(chunks)
            self.last_ordering_info = ordering_info

            # Convert to hex string for bytes_to_image function
            combined_hex = ' '.join([f'{pixel:02X}' for pixel in ordered_pixels])

            # Generate thermal image with correct MLX90640 dimensions
            thermal_image, stats = bytes_to_image(combined_hex, width=32, height=24, display=False)
            self.current_image = thermal_image

            # Create display frame
            display_frame = ttk.Frame(self.results_frame)
            display_frame.pack(fill=tk.BOTH, expand=True)

            # Create matplotlib figure for thermal image
            fig = Figure(figsize=(10, 8), dpi=100, facecolor='white')
            ax = fig.add_subplot(111)

            # Display thermal image with appropriate colormap
            im = ax.imshow(thermal_image, cmap='hot', vmin=0, vmax=255, interpolation='nearest')

            # Create title with ordering method info
            method_text = "🎯 Auto-ordered by headers" if ordering_info['method'] == 'automatic_header_based' else "⚠️ Input order used"
            ax.set_title(f'🌡️ Satellite Thermal Image (32×24 pixels) - {method_text}\n' +
                        f'Range: {stats["min_temp"]}-{stats["max_temp"]}, ' +
                        f'Mean: {stats["mean_temp"]:.1f}, ' +
                        f'Hot pixels: {stats["hot_pixels"]} ({stats["hot_pixels"]/stats["total_pixels"]*100:.1f}%)',
                        fontsize=12)

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, label='Temperature Intensity', shrink=0.8)

            # Remove axis ticks but keep border
            ax.set_xticks([])
            ax.set_yticks([])

            # Create canvas
            canvas_widget = FigureCanvasTkAgg(fig, display_frame)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add statistics and ordering info panel
            info_frame = ttk.Frame(display_frame)
            info_frame.pack(fill=tk.X, pady=(10, 0))

            # Thermal statistics
            stats_text = (
                f"📊 Thermal Statistics:\n"
                f"• Total pixels: {stats['total_pixels']}\n"
                f"• Temperature range: {stats['thermal_range']}\n"
                f"• Hot pixels (>128): {stats['hot_pixels']}\n"
                f"• Mean temperature: {stats['mean_temp']:.2f}"
            )

            stats_label = ttk.Label(info_frame, text=stats_text, font=('Courier', 9))
            stats_label.pack(side=tk.LEFT, padx=20)

            # Ordering information
            ordering_text = f"📡 Packet Ordering ({ordering_info['method']}):\n"
            ordering_text += f"• Valid headers: {ordering_info['valid_headers']}/4\n"

            for i, chunk_info in enumerate(ordering_info['chunk_order']):
                header_status = "✓" if chunk_info['has_header'] else "✗"
                ordering_text += f"• Pos {i+1}: Packet {chunk_info['original_position']} → Chunk {chunk_info['chunk_index']} {header_status}\n"

            if not ordering_info.get('same_image', True):
                ordering_text += "⚠️ Warning: Different image numbers detected!"

            ordering_label = ttk.Label(info_frame, text=ordering_text, font=('Courier', 9))
            ordering_label.pack(side=tk.RIGHT, padx=20)

            # Enable save button
            self.save_button.config(state='normal')

            # Update status
            if ordering_info['method'] == 'automatic_header_based':
                self.status_var.set(f"✅ Auto-reconstructed using packet headers! {stats['hot_pixels']} hot pixels detected")
            else:
                self.status_var.set(f"⚠️ Reconstructed using input order (headers not found). {stats['hot_pixels']} hot pixels")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to auto-reconstruct thermal image:\n{str(e)}")
            self.status_var.set("❌ Error auto-reconstructing thermal image")

    def generate_combinations(self):
        """Generate all possible combinations if auto-detection fails"""
        try:
            # Get hex data from all 4 inputs
            chunks = []
            for i, text_widget in enumerate(self.hex_inputs):
                hex_data = text_widget.get(1.0, tk.END).strip()
                chunks.append(hex_data)

            # Check if we have at least some data
            if not any(chunk.strip() for chunk in chunks):
                messagebox.showwarning("No Input", "Please paste satellite downlink packet data")
                return

            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()

            self.combination_images = []
            self.status_var.set("🔄 Generating all possible thermal image combinations...")
            self.root.update_idletasks()

            # For combinations, treat each chunk as raw payload data (strip headers if present)
            processed_chunks = []
            for chunk in chunks:
                if chunk.strip():
                    try:
                        packet_bytes = decode_byte_stream(chunk)
                    except ValueError as exc:
                        print(f"Combination generation: could not parse chunk ({exc})")
                        processed_chunks.append(chunk)
                        continue

                    header = parse_packet_header(packet_bytes)
                    if header and header['has_valid_header']:
                        # Extract just the payload
                        payload = extract_image_payload(packet_bytes, header)
                        hex_string = ' '.join([f'{pixel:02X}' for pixel in payload])
                        processed_chunks.append(hex_string)
                    else:
                        processed_chunks.append(chunk)
                else:
                    processed_chunks.append(chunk)

            # Generate all 24 permutations using processed chunks
            combinations_frame = ttk.Frame(self.results_frame)
            combinations_frame.pack(fill=tk.BOTH, expand=True)

            # Create a grid for combinations
            for i, perm in enumerate(permutations(range(4))):
                row = i // 6
                col = i % 6

                # Get the chunks in this order
                ordered_chunks = [processed_chunks[j] for j in perm]

                # Combine the chunks sequentially
                combined_pixels = []
                for chunk_hex in ordered_chunks:
                    if chunk_hex.strip():
                        try:
                            pixel_values = decode_byte_stream(chunk_hex)[:PIXELS_PER_CHUNK]
                        except ValueError as exc:
                            print(f"Combination {perm}: failed to parse chunk ({exc})")
                            pixel_values = [0] * PIXELS_PER_CHUNK
                    else:
                        pixel_values = [0] * PIXELS_PER_CHUNK

                    # Pad to correct size
                    if len(pixel_values) < PIXELS_PER_CHUNK:
                        pixel_values.extend([0] * (PIXELS_PER_CHUNK - len(pixel_values)))

                    combined_pixels.extend(pixel_values)

                # Convert to hex string
                combined_hex = ' '.join([f'{pixel:02X}' for pixel in combined_pixels])

                # Generate thermal image
                thermal_image, stats = bytes_to_image(combined_hex, width=32, height=24, display=False)
                self.combination_images.append((thermal_image, stats, perm, combined_hex))

                # Create frame for this combination
                combo_frame = ttk.LabelFrame(combinations_frame,
                                           text=f"Option {i+1}: {perm[0]+1}-{perm[1]+1}-{perm[2]+1}-{perm[3]+1}",
                                           padding="3")
                combo_frame.grid(row=row, column=col, padx=3, pady=3, sticky=(tk.W, tk.E, tk.N, tk.S))

                # Create matplotlib figure for this combination
                fig = Figure(figsize=(2.5, 3.5), dpi=75, facecolor='white')
                ax = fig.add_subplot(111)
                ax.imshow(thermal_image, cmap='hot', vmin=0, vmax=255, interpolation='nearest')
                ax.set_title(f'Hot: {stats["hot_pixels"]}', fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

                # Create canvas for this combination
                canvas_widget = FigureCanvasTkAgg(fig, combo_frame)
                canvas_widget.draw()
                canvas_widget.get_tk_widget().pack()

                # Make it clickable
                def make_click_handler(index):
                    def on_click(event):
                        self.select_combination(index)
                    return on_click

                canvas_widget.get_tk_widget().bind("<Button-1>", make_click_handler(i))
                canvas_widget.get_tk_widget().configure(cursor="hand2")

                # Add selection indicator
                selection_frame = tk.Frame(combo_frame, bg='red', height=3)
                selection_frame.pack(fill=tk.X)
                selection_frame.pack_forget()

                canvas_widget.get_tk_widget().selection_frame = selection_frame

            # Configure grid weights
            for i in range(6):
                combinations_frame.columnconfigure(i, weight=1)
            for i in range(4):
                combinations_frame.rowconfigure(i, weight=1)

            self.status_var.set(f"🔍 Generated {len(self.combination_images)} combinations. Click the best one!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate combinations:\n{str(e)}")
            self.status_var.set("❌ Error generating combinations")

    def select_combination(self, index):
        """Select a specific thermal image combination"""
        try:
            # Hide all selection frames first
            for widget in self.results_frame.winfo_children():
                for child in widget.winfo_children():
                    if hasattr(child, 'winfo_children'):
                        for subchild in child.winfo_children():
                            if hasattr(subchild, 'selection_frame'):
                                subchild.selection_frame.pack_forget()

            # Show selection frame for chosen combination
            combo_frames = []
            for widget in self.results_frame.winfo_children():
                for child in widget.winfo_children():
                    if isinstance(child, ttk.LabelFrame):
                        combo_frames.append(child)

            if index < len(combo_frames):
                for child in combo_frames[index].winfo_children():
                    if hasattr(child, 'selection_frame'):
                        child.selection_frame.pack(fill=tk.X)

            # Store selected combination
            self.selected_combination = index
            self.current_image = self.combination_images[index][0]
            perm = self.combination_images[index][2]
            stats = self.combination_images[index][1]

            # Enable save button
            self.save_button.config(state='normal')

            self.status_var.set(f"🎯 Selected combination {index+1} (order: {perm[0]+1}-{perm[1]+1}-{perm[2]+1}-{perm[3]+1}) - {stats['hot_pixels']} hot pixels")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to select thermal image:\n{str(e)}")

    def save_image(self):
        """Save the thermal image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please reconstruct a thermal image first")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ],
                title="Save Satellite Thermal Image"
            )

            if filename:
                cv2.imwrite(filename, self.current_image)

                # Determine ordering method for save message
                if hasattr(self, 'last_ordering_info') and self.last_ordering_info:
                    if self.last_ordering_info['method'] == 'automatic_header_based':
                        ordering_msg = "Automatically ordered by packet headers"
                    else:
                        ordering_msg = "Input order used (no valid headers)"
                elif hasattr(self, 'selected_combination') and self.selected_combination is not None:
                    perm = self.combination_images[self.selected_combination][2]
                    ordering_msg = f"Manual combination: {perm[0]+1}-{perm[1]+1}-{perm[2]+1}-{perm[3]+1}"
                else:
                    ordering_msg = "Unknown ordering method"

                messagebox.showinfo("Success", f"🛰️ Satellite thermal image saved!\n{filename}\n\n{ordering_msg}")
                self.status_var.set(f"💾 Thermal image saved: {os.path.basename(filename)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save thermal image:\n{str(e)}")
            self.status_var.set("❌ Error saving thermal image")

    def clear_all_inputs(self):
        """Clear all input data"""
        for text_widget in self.hex_inputs:
            text_widget.delete(1.0, tk.END)

        self.show_results_placeholder()
        self.reset_processing_state()
        self.status_var.set("🗑️ All data cleared - Ready for new satellite downlink")

    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()

    def load_default_inputs(self, initial_load=False):
        """Populate packet inputs using DEFAULT_PACKET_STRINGS constants."""
        defaults = list(DEFAULT_PACKET_STRINGS[:4])
        if len(defaults) < 4:
            defaults.extend([''] * (4 - len(defaults)))

        has_defaults = any(chunk.strip() for chunk in defaults)
        if not has_defaults:
            if not initial_load:
                messagebox.showinfo(
                    "No Default Data",
                    "Update DEFAULT_PACKET_STRINGS near the top of the file to enable this feature."
                )
                self.status_var.set("ℹ️ No default packet data configured")
            return

        for widget, default in zip(self.hex_inputs, defaults):
            widget.delete(1.0, tk.END)
            if default:
                widget.insert(tk.END, default.strip())

        self.show_results_placeholder()
        self.reset_processing_state()

        if initial_load:
            self.status_var.set("📋 Loaded default packet data")
        else:
            self.status_var.set("📋 Loaded default packet data from presets")


def launch_gui():
    """Launch the Satellite Thermal Image Reconstructor GUI"""
    app = SatelliteThermalImageGUI()
    app.run()


if __name__ == "__main__":
    launch_gui()