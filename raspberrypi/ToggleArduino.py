"""
Arduino Toggle Controller
Sends periodic True/False commands to Arduino via serial communication
Arduino controls servo based on received commands:
- True = Servo to 180 degrees
- False = Servo to 0 degrees
"""

import serial
import time
import sys

class ArduinoController:
    def __init__(self, port='COM3', baudrate=9600, timeout=1):
        """
        Initialize Arduino controller
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Communication speed (must match Arduino sketch)
            timeout: Serial timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.state = False  # Initial state
        
    def connect(self):
        """Establish serial connection with Arduino"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Give Arduino time to reset after connection
            print(f"Connected to Arduino on {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            print("Please check:")
            print("- Arduino is connected")
            print("- Correct port is specified")
            print("- Arduino IDE Serial Monitor is closed")
            return False
    
    def send_command(self, state):
        """
        Send True/False command to Arduino
        
        Args:
            state: Boolean value to send
        """
        if self.ser and self.ser.is_open:
            command = "1" if state else "0"
            self.ser.write(command.encode())
            print(f"Sent: {state} ({'1' if state else '0'}) -> Servo to {180 if state else 0}°")
            
            # Read response from Arduino (optional)
            try:
                response = self.ser.readline().decode().strip()
                if response:
                    print(f"Arduino response: {response}")
            except:
                pass  # Ignore read errors
        else:
            print("Error: Not connected to Arduino")
    
    def toggle_state(self):
        """Toggle the current state"""
        self.state = not self.state
        return self.state
    
    def run_periodic(self, interval=10):
        """
        Run periodic state toggle and send to Arduino
        
        Args:
            interval: Time interval in seconds between toggles
        """
        if not self.connect():
            return
        
        print(f"Starting periodic toggle every {interval} seconds")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        try:
            while True:
                # Toggle state
                new_state = self.toggle_state()
                
                # Send to Arduino
                self.send_command(new_state)
                
                # Wait for next toggle
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nStopping periodic toggle...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.disconnect()
    
    def manual_control(self):
        """Manual control mode for testing"""
        if not self.connect():
            return
        
        print("Manual Control Mode")
        print("Commands:")
        print("  '1' or 'true' = Set servo to 180°")
        print("  '0' or 'false' = Set servo to 0°")
        print("  'q' or 'quit' = Exit")
        print("-" * 30)
        
        try:
            while True:
                user_input = input("Enter command: ").lower().strip()
                
                if user_input in ['q', 'quit', 'exit']:
                    break
                elif user_input in ['1', 'true', 't']:
                    self.send_command(True)
                elif user_input in ['0', 'false', 'f']:
                    self.send_command(False)
                else:
                    print("Invalid command. Use '1'/'true' or '0'/'false'")
                    
        except KeyboardInterrupt:
            print("\nExiting manual control...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.disconnect()
    
    def disconnect(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected from Arduino")


def main():
    """Main function"""
    # Configuration
    ARDUINO_PORT = '/dev/ttyUSB0'  # Raspberry Pi port (change to /dev/ttyACM0 if needed)
    BAUDRATE = 9600
    TOGGLE_INTERVAL = 10  # seconds
    
    print("Arduino Servo Controller")
    print("=" * 30)
    
    # Create controller
    controller = ArduinoController(port=ARDUINO_PORT, baudrate=BAUDRATE)
    
    # Show menu
    print("Select mode:")
    print("1. Periodic toggle (every 10 seconds)")
    print("2. Manual control")
    print("3. Exit")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            controller.run_periodic(TOGGLE_INTERVAL)
        elif choice == '2':
            controller.manual_control()
        elif choice == '3':
            print("Goodbye!")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
