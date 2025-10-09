/*
  Arduino Servo Controller
  
  Receives serial commands from computer/Raspberry Pi and controls servo motor
  Commands:
  - '1' or 'true'  = Move servo to 180 degrees
  - '0' or 'false' = Move servo to 0 degrees
  
  Hardware connections:
  - Servo signal wire to digital pin 9
  - Servo power (red) to 5V
  - Servo ground (black/brown) to GND
  
  Author: Arduino Servo Controller
  Date: October 2025
*/

#include <Servo.h>

// Create servo object
Servo doorServo;

// Pin definitions
const int SERVO_PIN = 9;        // Servo signal pin
const int LED_PIN = 13;         // Built-in LED for status indication


// Servo positions
const int SERVO_OPEN = 180;     // Open position (degrees)
const int SERVO_CLOSED = 0;     // Closed position (degrees)

// Variables
bool currentState = false;      // Current state (false = closed, true = open)
String inputString = "";        // String to hold incoming data
bool stringComplete = false;    // Whether the string is complete

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Attach servo to pin
  doorServo.attach(SERVO_PIN);
  
  // Initialize LED pin
  pinMode(LED_PIN, OUTPUT);
  
  // Set initial position
  doorServo.write(SERVO_CLOSED);
  digitalWrite(LED_PIN, LOW);
  
  // Reserve string buffer
  inputString.reserve(200);
  
  // Startup message
  Serial.println("Arduino Servo Controller Ready");
  Serial.println("Commands: '1' = 180°, '0' = 0°");
  Serial.println("Current position: 0° (CLOSED)");
  Serial.println("------------------------");
  
  delay(1000); // Give servo time to reach initial position
}

void loop() {
  // Check for incoming serial data
  if (stringComplete) {
    processCommand(inputString);
    
    // Clear the string for next command
    inputString = "";
    stringComplete = false;
  }
  
  // Small delay to prevent overwhelming the processor
  delay(10);
}

void processCommand(String command) {
  // Remove any whitespace
  command.trim();
  
  // Convert to lowercase for easier comparison
  command.toLowerCase();
  
  bool newState = currentState; // Default to current state
  
  // Parse command
  if (command == "1" || command == "true" || command == "t") {
    newState = true;
  }
  else if (command == "0" || command == "false" || command == "f") {
    newState = false;
  }
  else {
    // Invalid command
    Serial.println("ERROR: Invalid command '" + command + "'");
    Serial.println("Valid commands: '1', 'true', '0', 'false'");
    return;
  }
  
  // Execute command if state changed
  if (newState != currentState) {
    currentState = newState;
    moveServo(currentState);
  }
  else {
    Serial.println("INFO: Servo already in requested position");
  }
}

void moveServo(bool state) {
  int targetAngle = state ? SERVO_OPEN : SERVO_CLOSED;
  String stateStr = state ? "OPEN" : "CLOSED";
  
  Serial.println("Moving servo to " + String(targetAngle) + "° (" + stateStr + ")");
  
  // Update LED status
  digitalWrite(LED_PIN, state ? HIGH : LOW);
  
  // Move servo gradually for smoother operation
  int currentAngle = doorServo.read();
  int step = (targetAngle > currentAngle) ? 1 : -1;
  
  for (int angle = currentAngle; angle != targetAngle; angle += step) {
    doorServo.write(angle);
    delay(15); // Smooth movement delay
  }
  
  // Ensure final position
  doorServo.write(targetAngle);
  
  Serial.println("Servo movement complete: " + String(targetAngle) + "° (" + stateStr + ")");
}

/*
  SerialEvent occurs whenever new data comes in the hardware serial RX.
  This routine is run between each time loop() runs, so using delay inside
  loop can delay response. Multiple bytes of data may be available.
*/
void serialEvent() {
  while (Serial.available()) {
    // Get the new byte
    char inChar = (char)Serial.read();
    
    // Add it to the inputString
    inputString += inChar;
    
    // If the incoming character is a newline, set a flag so the main loop can
    // do something about it, or if we receive a single character command
    if (inChar == '\n' || inChar == '\r' || inputString.length() == 1) {
      stringComplete = true;
    }
  }
}

/*
  Additional utility functions
*/

// Function to get current servo position
int getCurrentPosition() {
  return doorServo.read();
}

// Function to print status
void printStatus() {
  Serial.println("--- Status ---");
  Serial.println("Current State: " + String(currentState ? "OPEN (1)" : "CLOSED (0)"));
  Serial.println("Servo Angle: " + String(getCurrentPosition()) + "°");
  Serial.println("LED Status: " + String(digitalRead(LED_PIN) ? "ON" : "OFF"));
  Serial.println("-------------");
}