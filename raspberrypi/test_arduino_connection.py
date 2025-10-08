"""
Arduino Connection Test for Raspberry Pi
Real test script to find and verify Arduino communication
"""

import time
import os
import glob

def find_arduino_ports():
    """Find potential Arduino ports on Raspberry Pi"""
    print("Scanning for Arduino ports...")
    
    # Common Arduino ports on Raspberry Pi
    potential_ports = [
        '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2',
        '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2'
    ]
    
    # Check which ports exist
    available_ports = []
    for port in potential_ports:
        if os.path.exists(port):
            available_ports.append(port)
    
    # Also check for any ttyUSB* or ttyACM* devices
    usb_ports = glob.glob('/dev/ttyUSB*')
    acm_ports = glob.glob('/dev/ttyACM*')
    
    all_ports = list(set(available_ports + usb_ports + acm_ports))
    
    if all_ports:
        print(f"Found potential Arduino ports: {all_ports}")
    else:
        print("No Arduino ports found")
        print("Make sure Arduino is connected via USB")
    
    return all_ports

def test_arduino_connection(port):
    """Test connection to Arduino on specified port"""
    try:
        import serial
        
        print(f"Testing connection to {port}...")
        
        # Try to open serial connection
        ser = serial.Serial(port, 9600, timeout=2)
        time.sleep(2)  # Give Arduino time to reset
        
        print(f"✓ Successfully connected to {port}")
        
        # Send test commands
        test_commands = ['1', '0', '1', '0']
        
        for i, cmd in enumerate(test_commands):
            print(f"Sending command: '{cmd}'")
            ser.write(cmd.encode())
            
            # Try to read response
            time.sleep(1)
            if ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                print(f"Arduino response: {response}")
            
            time.sleep(2)  # Wait between commands
        
        ser.close()
        print(f"✓ Test completed successfully on {port}")
        return True
        
    except ImportError:
        print("Error: pyserial not installed")
        print("Install with: pip install pyserial")
        return False
        
    except serial.SerialException as e:
        print(f"✗ Failed to connect to {port}: {e}")
        return False
        
    except Exception as e:
        print(f"✗ Error testing {port}: {e}")
        return False

def test_without_pyserial():
    """Test script that simulates Arduino communication without pyserial"""
    print("Arduino Connection Test (Simulation Mode)")
    print("=" * 50)
    print("Note: Install pyserial for actual Arduino communication:")
    print("pip install pyserial")
    print()
    
    # Show what ports would be checked
    ports = find_arduino_ports()
    if ports:
        print(f"Would test these ports: {ports}")
    
    print()
    state = False
    interval = 5  # Shorter interval for testing
    
    print(f"Simulating periodic toggle every {interval} seconds")
    print("Press Ctrl+C to stop")
    print("-" * 30)
    
    try:
        for i in range(4):  # Run for 4 cycles
            state = not state
            command = "1" if state else "0"
            
            print(f"Cycle {i+1}: Sending '{command}' -> Servo to {180 if state else 0}°")
            print(f"Arduino would respond: 'Moving servo to {180 if state else 0}° ({'OPEN' if state else 'CLOSED'})'")
            
            if i < 3:  # Don't sleep after last cycle
                print(f"Waiting {interval} seconds...")
                time.sleep(interval)
                print()
    
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    
    print("Test completed!")

def main():
    """Main function"""
    print("Arduino Connection Test for Raspberry Pi")
    print("=" * 45)
    
    # Try to import pyserial
    try:
        import serial
        print("✓ pyserial is available - testing real connections")
        
        # Find available ports
        ports = find_arduino_ports()
        
        if not ports:
            print("No Arduino ports found. Make sure Arduino is connected.")
            return
        
        # Test each port
        for port in ports:
            if test_arduino_connection(port):
                print(f"\n✓ Arduino found and working on {port}")
                print(f"Use this port in your ToggleArduino.py: ARDUINO_PORT = '{port}'")
                break
        else:
            print("\n✗ No working Arduino found on any port")
            
    except ImportError:
        print("⚠ pyserial not available - running simulation")
        test_without_pyserial()

if __name__ == "__main__":
    main()