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
manager = USBSerialManager()

# List available devices
devices = manager.list_usb_devices()
print("Available USB devices:")
for device in devices:
    print(f"  {device['device']}: {device['description']}")

# Connect to devices
if devices:
    device_path = devices[0]['device']
    if manager.connect_device(device_path, 115200, 'gps'):
        manager.send_to_device('gps', 'AT\r\n')
        response = manager.read_from_device('gps', timeout=2)
        print(f"GPS response: {response}")

manager.close_all()