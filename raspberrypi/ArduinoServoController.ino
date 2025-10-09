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

// Buffer for command handling
const size_t MAX_LINE = 32;
char lineBuf[MAX_LINE];
size_t idx = 0;

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
  
  // Startup message
  Serial.println("Ready: Servo(1/0) Light(TURN_ON/OFF)");
  
  delay(1000); // Give servo time to reach initial position
}

void loop() {
  // Handle incoming serial data with buffer
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r') continue;                // ignore CR
    if (ch == '\n') {                        // end of line
      lineBuf[idx] = '\0';
      handleCommand(lineBuf);
      idx = 0;
    } else if (idx < MAX_LINE - 1) {
      lineBuf[idx++] = ch;                   // accumulate
    } else {
      idx = 0;                               // overflow -> reset
    }
  }
  
  // Small delay to prevent overwhelming the processor
  delay(10);
}

void handleCommand(const char* cmd) {
  // Light control commands
  if (strcmp(cmd, "TURN_ON") == 0 || strcmp(cmd, "turn_on") == 0) {
    controlLight(true);
  } else if (strcmp(cmd, "TURN_OFF") == 0 || strcmp(cmd, "turn_off") == 0) {
    controlLight(false);
  } else if (strcmp(cmd, "ON_LIGHTS") == 0) {
    controlLight(true);
  } else if (strcmp(cmd, "OFF_LIGHTS") == 0) {
    controlLight(false);
  } 
  // Servo commands
  else if (strcmp(cmd, "1") == 0 || strcmp(cmd, "true") == 0 || strcmp(cmd, "t") == 0) {
    if (!currentState) {  // Only move if state changed
      currentState = true;
      moveServo(currentState);
    }
  } else if (strcmp(cmd, "0") == 0 || strcmp(cmd, "false") == 0 || strcmp(cmd, "f") == 0) {
    if (currentState) {   // Only move if state changed
      currentState = false;
      moveServo(currentState);
    }
  } 
  // Status command
  else if (strcmp(cmd, "status") == 0) {
    printStatus();
  }
  // Error for unknown commands
  else {
    Serial.print("ERR:");
    Serial.println(cmd);
  }
}

void moveServo(bool state) {
  int targetAngle = state ? SERVO_OPEN : SERVO_CLOSED;
  
  // Update LED status
  digitalWrite(LED_PIN, state ? HIGH : LOW);
  
  // Move servo gradually for smoother operation
  int currentAngle = doorServo.read();
  int step = (targetAngle > currentAngle) ? 1 : -1;
  
  for (int angle = currentAngle; angle != targetAngle; angle += step) {
    doorServo.write(angle);
    delay(5); // Smooth movement delay
  }
  
  // Ensure final position
  doorServo.write(targetAngle);
  
  Serial.println("S:" + String(targetAngle));
}

void controlLight(bool state) {
  lightState = state;
  
  // Update light state
  digitalWrite(LED_LIGHT_PIN, state ? HIGH : LOW);
  
  Serial.println("L:" + String(state ? "ON" : "OFF"));
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
  Serial.println("S:" + String(getCurrentPosition()) + " L:" + String(lightState ? "ON" : "OFF"));
}