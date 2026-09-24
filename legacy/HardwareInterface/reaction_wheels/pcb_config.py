import smbus2

# Configuration
I2C_BUS = 1            # Standard I2C bus for Raspberry Pi (Rev 2 and newer)
DEVICE_ADDRESS = 0x18  # The I2C address of your device

# Registers and values to write
REG_1 = 0x01
VAL_1 = 0xFF

REG_3 = 0x03
VAL_3 = 0x00

def main():
    # Initialize the I2C bus
    # Using a context manager (with statement) ensures the bus closes automatically safely
    try:
        with smbus2.SMBus(I2C_BUS) as bus:
            
            # Write 0xFF to register 1
            bus.write_byte_data(DEVICE_ADDRESS, REG_1, VAL_1)
            print(f"Successfully wrote {hex(VAL_1)} to register {hex(REG_1)}")

            # Write 0x00 to register 3
            bus.write_byte_data(DEVICE_ADDRESS, REG_3, VAL_3)
            print(f"Successfully wrote {hex(VAL_3)} to register {hex(REG_3)}")

    except PermissionError:
        print("Permission denied. Try running the script with 'sudo' or adding your user to the 'i2c' group.")
    except OSError as e:
        print(f"I2C Bus Error: {e}")
        print(f"Please check your wiring and ensure a device is connected at address {hex(DEVICE_ADDRESS)}.")
        print("You can verify the address by running 'i2cdetect -y 1' in your terminal.")

if __name__ == "__main__":
    main()