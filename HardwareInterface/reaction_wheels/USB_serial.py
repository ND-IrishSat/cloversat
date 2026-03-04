'''
USB_serial.py
Authors: Andrew G

For sending a single PWM byte from Pi to Arduino over USB serial

'''

import serial
import serial.tools.list_ports
import time

class USBSerialManager:
    def __init__(self):
        self.connections = {}

    def list_usb_devices(self):
        """List all USB serial devices"""
        ports = serial.tools.list_ports.comports()
        usb_ports = []

        for port in ports:
            if 'USB' in port.description or 'ttyUSB' in port.device:
                usb_ports.append({
                    'device': port.device,
                    'description': port.description,
                    'vid': port.vid,
                    'pid': port.pid,
                    'serial_number': port.serial_number
                })

        return usb_ports

    def connect_device(self, device_path, baudrate=9600, name=None):
        """Connect to USB serial device"""
        if name is None:
            name = device_path

        try:
            ser = serial.Serial(
                port=device_path,
                baudrate=baudrate,
                timeout=1
            )
            self.connections[name] = ser
            print(f"Connected to {device_path} as {name}")
            return True

        except serial.SerialException as e:
            print(f"Failed to connect to {device_path}: {e}")
            return False

    def send_to_device(self, name, data):
        """Send data to specific device"""
        if name not in self.connections:
            return False

        try:
            self.connections[name].write(data.encode())
            return True
        except:
            return False

    def send_pwm_byte(self, name, pwm_byte):
        """Send a single byte to a device, for PWM."""
        if name not in self.connections:
            print(f"Device {name} not connected.")
            return False

        if not (0 <= pwm_byte <= 255):
            print("PWM byte must be between 0 and 255.")
            return False

        try:
            pwm_byte = bytes([pwm_byte])
            print("Sending: ")
            print(pwm_byte)
            #self.connections[name].write(bytes([pwm_byte]))
            #print(self.connections[name])
            bytes_written = self.connections[name].write(pwm_byte)
            print("Wrote ", bytes_written, " bytes")
            
            #time.sleep(.5)
            #self.connections[name].flush()
            return True
        except Exception as e:
            print(f"Failed to send PWM byte to {name}: {e}")
            return False

    def read_from_device(self, name, timeout=1):
        """Read from specific device"""
        if name not in self.connections:
            return None

        ser = self.connections[name]
        ser.timeout = timeout

        data = b''
        start_time = time.time()

        while time.time() - start_time < timeout:
            if ser.in_waiting:
                data += ser.read(ser.in_waiting)
            time.sleep(0.01)

        return data.decode() if data else None

    def close_all(self):
        """Close all connections"""
        for name, ser in self.connections.items():
            ser.close()
        self.connections.clear()

# Usage example
if __name__ == "__main__":
    manager = USBSerialManager()

    # List available devices
    devices = manager.list_usb_devices()
    print("Available USB devices:")
    for device in devices:
        print(f"  {device['device']}: {device['description']}")

    # Connect to the first available USB device, assuming it's the Arduino
    if devices:
        # On a Raspberry Pi, this might be '/dev/ttyUSB0' or '/dev/ttyACM0'
        # On Windows, it will be a 'COM' port.
        device_path = devices[0]['device']

        # The name 'motor_controller' is an arbitrary name for this connection
        if manager.connect_device(device_path, 115200, name='motor_controller'):
            print("\n--- PWM Example ---")

            # It's good practice to wait a moment for the serial connection to establish
            # and for the Arduino to reset and clear its buffer.
            time.sleep(2)

            try:
                # --- Ramp up PWM from 0 to 255 ---
                print("Ramping up PWM...")
                for pwm_val in range(0, 256, 5):
                    print(f"Sending PWM value: {pwm_val}")
                    if manager.send_pwm_byte('motor_controller', pwm_val):
                        # Reading the response from the Arduino
                        response = manager.read_from_device('motor_controller', timeout=0.1)
                        if response:
                            print(f"  Arduino response: {response.strip()}")
                    else:
                        print("  Failed to send PWM byte.")
                    time.sleep(0.1) # Delay between sends

                # --- Ramp down PWM from 255 to 0 ---
                print("\nRamping down PWM...")
                for pwm_val in range(50, -1, -5):
                    print(f"Sending PWM value: {pwm_val}")
                    if manager.send_pwm_byte('motor_controller', pwm_val):
                        response = manager.read_from_device('motor_controller', timeout=0.1)
                        if response:
                            print(f"  Arduino response: {response.strip()}")
                    else:
                        print("  Failed to send PWM byte.")
                    time.sleep(0.1)

            except KeyboardInterrupt:
                print("\nPWM test stopped by user.")
            finally:
                # Set PWM to 0 as a safety measure
                print("\nSetting PWM to 0.")
                manager.send_pwm_byte('motor_controller', 0)
                manager.close_all()
                print("Connection closed.")
    else:
        print("\nNo USB devices found.")
