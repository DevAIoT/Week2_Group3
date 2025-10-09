/*
  Arduino Servo Controller with LED Light Control
  
  Receives serial commands from computer/Raspberry Pi and controls:
  1. Servo motor (door control)
  2. LED light (separate lighting control)
  
  Servo Commands:
  - '1' or 'true'  = Move servo to 180 degrees
  - '0' or 'false' = Move servo to 0 degrees
  
  LED Light Commands:
  - 'TURN_ON'  = Turn on LED light (pin 8)
  - 'TURN_OFF' = Turn off LED light (pin 8)
  
  Hardware connections:
  - Servo signal wire to digital pin 9
  - Servo power (red) to 5V
  - Servo ground (black/brown) to GND
  - LED light to digital pin 8 (with appropriate resistor)
  
  Author: Arduino Servo Controller
  Date: October 2025
*/

#include <Servo.h>

// Create servo object
Servo doorServo;

// Pin definitions
const int SERVO_PIN = 9;        // Servo signal pin
const int LED_PIN = 13;         // Built-in LED for status indication
const int LED_LIGHT_PIN = 8;    // Additional LED for light indication


// Servo positions
const int SERVO_OPEN = 180;     // Open position (degrees)
const int SERVO_CLOSED = 0;     // Closed position (degrees)

// Variables
bool currentState = false;      // Current servo state (false = closed, true = open)
bool lightState = false;        // Current light state (false = off, true = on)
String inputString = "";        // String to hold incoming data
bool stringComplete = false;    // Whether the string is complete

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Attach servo to pin
  doorServo.attach(SERVO_PIN);
  
  // Initialize LED pins
  pinMode(LED_PIN, OUTPUT);
  pinMode(LED_LIGHT_PIN, OUTPUT);
  
  // Set initial positions
  doorServo.write(SERVO_CLOSED);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(LED_LIGHT_PIN, LOW);
  
  // Reserve string buffer
  inputString.reserve(200);
  
  // Startup message
  Serial.println("Arduino Servo & Light Controller Ready");
  Serial.println("Servo Commands: '1' = 180°, '0' = 0°");
  Serial.println("Light Commands: 'TURN_ON', 'TURN_OFF'");
  Serial.println("Current position: 0° (CLOSED), Light: OFF");
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
  
  // Check for light control commands first
  if (command == "turn_on") {
    controlLight(true);
    return;
  }
  else if (command == "turn_off") {
    controlLight(false);
    return;
  }
  
  // Process servo commands
  bool newState = currentState; // Default to current state
  
  // Parse servo command
  if (command == "1" || command == "true" || command == "t") {
    newState = true;
  }
  else if (command == "0" || command == "false" || command == "f") {
    newState = false;
  }
  else {
    // Invalid command
    Serial.println("ERROR: Invalid command '" + command + "'");
    Serial.println("Valid commands:");
    Serial.println("  Servo: '1', 'true', '0', 'false'");
    Serial.println("  Light: 'TURN_ON', 'TURN_OFF'");
    return;
  }
  
  // Execute servo command if state changed
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

void controlLight(bool state) {
  lightState = state;
  String stateStr = state ? "ON" : "OFF";
  
  Serial.println("Setting light to " + stateStr);
  
  // Update light state
  digitalWrite(LED_LIGHT_PIN, state ? HIGH : LOW);
  
  Serial.println("Light control complete: " + stateStr);
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
  Serial.println("Servo State: " + String(currentState ? "OPEN (1)" : "CLOSED (0)"));
  Serial.println("Servo Angle: " + String(getCurrentPosition()) + "°");
  Serial.println("Status LED: " + String(digitalRead(LED_PIN) ? "ON" : "OFF"));
  Serial.println("Light State: " + String(lightState ? "ON" : "OFF"));
  Serial.println("Light Pin 8: " + String(digitalRead(LED_LIGHT_PIN) ? "HIGH" : "LOW"));
  Serial.println("-------------");
}