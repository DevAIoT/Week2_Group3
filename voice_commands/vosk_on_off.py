import vosk
import sounddevice as sd
import queue
import json
import sys
import threading
import numpy as np
import serial
import serial.tools.list_ports
import time

# Configuration
SAMPLE_RATE = 16000
MODEL_PATH = "vosk-model-small-en-us-0.15"
SERIAL_PORT = "/dev/ttyACM0"  # Change to your Arduino port (e.g., /dev/ttyACM0)
BAUD_RATE = 9600

def initialize_serial():
    """Initialize serial connection to Arduino"""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        print(f"Serial connection established on {SERIAL_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        print("Available ports:")
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"  - {port.device}")
        sys.exit(1)

def initialize_vosk():
    """Initialize Vosk model and recognizer"""
    try:
        model = vosk.Model(MODEL_PATH)
        recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
        recognizer.SetWords(True)
        return recognizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure {MODEL_PATH} exists in the current directory")
        sys.exit(1)

def send_command(ser, command):
    """Send command to Arduino via serial"""
    try:
        ser.write(f"{command}\n".encode())
        print(f"Sent to Arduino: {command}")
        
        # Optional: Read response from Arduino
        time.sleep(0.1)
        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            print(f"Arduino response: {response}")
    except Exception as e:
        print(f"Error sending command: {e}")

def main():
    # Initialize serial and vosk
    serial_connection = initialize_serial()
    recognizer = initialize_vosk()
    q = queue.Queue()
    
    def audio_callback(indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        q.put(bytes(indata))
    
    def process_audio():
        """Process audio data and recognize speech"""
        while True:
            data = q.get()
            if data is None:
                break
            
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get('text', '').strip().lower()
                
                if text:
                    print(f"\nRecognized: '{text}'")
                    
                    # Check for ON commands
                    if text == 'on' or 'turn on' in text or text.endswith(' on'):
                        print(">>> Command: LIGHT ON <<<")
                        send_command(serial_connection, "LIGHT_ON")
                        
                    # Check for OFF commands
                    elif text == 'off' or 'turn off' in text or text.endswith(' off'):
                        print(">>> Command: LIGHT OFF <<<")
                        send_command(serial_connection, "LIGHT_OFF")
            else:
                # Show partial results
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get('partial', '').strip()
                if partial_text:
                    print(f"Listening: {partial_text}...", end='\r')
    
    # Start processing thread
    thread = threading.Thread(target=process_audio, daemon=True)
    thread.start()
    
    print("=== Voice Command Recognition System ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Serial port: {SERIAL_PORT} @ {BAUD_RATE} baud")
    print("\nListening for commands: 'on', 'off', 'turn on', 'turn off'")
    print("Press Ctrl+C to stop\n")
    
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, 
                              blocksize=8000, 
                              dtype='int16',
                              channels=1, 
                              callback=audio_callback):
            # Keep running until interrupted
            while True:
                sd.sleep(100)
                
    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        q.put(None)
        thread.join()
        serial_connection.close()
        print("Serial connection closed. Goodbye!")

if __name__ == "__main__":
    main()