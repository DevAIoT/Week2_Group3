/*
  Arduino Servo Test Code
  
  Simple test to verify servo motor is working correctly
  This code will:
  1. Move servo through full range of motion
  2. Test specific positions (0°, 90°, 180°)
  3. Provide visual feedback with built-in LED
  4. Send status updates via Serial Monitor
  
  Hardware connections:
  - Servo signal wire to digital pin 9
  - Servo power (red) to 5V
  - Servo ground (black/brown) to GND
  
  Author: Arduino Servo Test
  Date: October 2025
*/

#include <Servo.h>

// Create servo object
Servo testServo;

// Pin definitions
const int SERVO_PIN = 9;        // Servo signal pin
const int LED_PIN = 13;         // Built-in LED for status indication

// Test variables
int testPhase = 0;
unsigned long lastMoveTime = 0;
const unsigned long MOVE_DELAY = 2000; // 2 seconds between moves

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Attach servo to pin
  testServo.attach(SERVO_PIN);
  
  // Initialize LED pin
  pinMode(LED_PIN, OUTPUT);
  
  // Startup message
  Serial.println("=================================");
  Serial.println("    Arduino Servo Test Started");
  Serial.println("=================================");
  Serial.println();
  Serial.println("This will test your servo motor by:");
  Serial.println("1. Moving to 0° (CLOSED position)");
  Serial.println("2. Moving to 90° (MIDDLE position)");
  Serial.println("3. Moving to 180° (OPEN position)");
  Serial.println("4. Sweeping back and forth");
  Serial.println();
  Serial.println("Watch your servo and LED for movement!");
  Serial.println("---------------------------------");
  
  // Start at 0 degrees
  testServo.write(0);
  digitalWrite(LED_PIN, LOW);
  Serial.println("Starting position: 0°");
  
  delay(2000); // Give servo time to reach position
  lastMoveTime = millis();
}

void loop() {
  unsigned long currentTime = millis();
  
  // Check if it's time for next test phase
  if (currentTime - lastMoveTime >= MOVE_DELAY) {
    runTestPhase();
    lastMoveTime = currentTime;
  }
  
  // Blink LED to show Arduino is running
  blinkStatusLED();
}

void runTestPhase() {
  switch (testPhase) {
    case 0:
      // Test position 0°
      Serial.println("Phase 1: Testing 0° (CLOSED) position");
      moveServoTo(0);
      digitalWrite(LED_PIN, LOW);
      break;
      
    case 1:
      // Test position 90°
      Serial.println("Phase 2: Testing 90° (MIDDLE) position");
      moveServoTo(90);
      digitalWrite(LED_PIN, HIGH);
      break;
      
    case 2:
      // Test position 180°
      Serial.println("Phase 3: Testing 180° (OPEN) position");
      moveServoTo(180);
      digitalWrite(LED_PIN, LOW);
      break;
      
    case 3:
      // Sweep test
      Serial.println("Phase 4: Performing sweep test");
      performSweepTest();
      break;
      
    case 4:
      // Quick position test
      Serial.println("Phase 5: Quick position changes");
      quickPositionTest();
      break;
      
    default:
      // Reset to beginning
      Serial.println("Test cycle complete! Restarting...");
      Serial.println("---------------------------------");
      testPhase = -1; // Will become 0 after increment
      break;
  }
  
  testPhase++;
}

void moveServoTo(int angle) {
  Serial.println("Moving to " + String(angle) + " degrees...");
  
  // Get current position
  int currentAngle = testServo.read();
  
  // Move gradually for smooth operation
  int step = (angle > currentAngle) ? 2 : -2;
  
  if (angle != currentAngle) {
    for (int pos = currentAngle; 
         (step > 0) ? (pos <= angle) : (pos >= angle); 
         pos += step) {
      testServo.write(pos);
      delay(20); // Smooth movement
    }
  }
  
  // Ensure final position
  testServo.write(angle);
  
  Serial.println("✓ Servo at " + String(angle) + "°");
  Serial.println("Current servo reading: " + String(testServo.read()) + "°");
  Serial.println();
}

void performSweepTest() {
  Serial.println("Sweeping from 0° to 180° and back...");
  
  // Sweep up
  for (int pos = 0; pos <= 180; pos += 5) {
    testServo.write(pos);
    delay(50);
  }
  
  Serial.println("✓ Reached 180°, sweeping back...");
  
  // Sweep down
  for (int pos = 180; pos >= 0; pos -= 5) {
    testServo.write(pos);
    delay(50);
  }
  
  Serial.println("✓ Sweep test complete!");
  Serial.println();
}

void quickPositionTest() {
  Serial.println("Testing quick position changes...");
  
  int positions[] = {0, 180, 0, 90, 180, 90, 0};
  int numPositions = sizeof(positions) / sizeof(positions[0]);
  
  for (int i = 0; i < numPositions; i++) {
    testServo.write(positions[i]);
    Serial.println("Quick move to: " + String(positions[i]) + "°");
    delay(800);
  }
  
  Serial.println("✓ Quick position test complete!");
  Serial.println();
}

void blinkStatusLED() {
  static unsigned long lastBlink = 0;
  static bool ledState = false;
  
  if (millis() - lastBlink > 500) { // Blink every 500ms
    ledState = !ledState;
    // Only blink if not being used for position indication
    if (testPhase != 1) {
      digitalWrite(LED_PIN, ledState);
    }
    lastBlink = millis();
  }
}

// Additional test functions
void printServoInfo() {
  Serial.println("=== Servo Information ===");
  Serial.println("Servo Pin: " + String(SERVO_PIN));
  Serial.println("Current Position: " + String(testServo.read()) + "°");
  Serial.println("Test Phase: " + String(testPhase));
  Serial.println("========================");
}

// Function to manually test specific angle (call from Serial Monitor)
void testAngle(int angle) {
  if (angle >= 0 && angle <= 180) {
    Serial.println("Manual test: Moving to " + String(angle) + "°");
    moveServoTo(angle);
  } else {
    Serial.println("ERROR: Angle must be between 0 and 180 degrees");
  }
}