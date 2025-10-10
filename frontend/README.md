# Smart Door IoT Dashboard

A Next.js dashboard for controlling a smart door system with facial recognition using Raspberry Pi and ESP32.

## System Architecture

- **Raspberry Pi**: Runs MediaPipe ML model for facial recognition
- **ESP32**: Controls servo motor (door lock) and LED (status light)
- **Next.js Dashboard**: Web interface for monitoring and control

## Backend Implementation Required

This dashboard includes placeholder API routes that need to be connected to your hardware:

### API Endpoints to Implement

1. **`/api/face-recognition`** - Connect to Raspberry Pi MediaPipe
   - Expected response: `{ faceDetected: boolean, user: string | null, confidence: number }`

2. **`/api/camera`** - Stream video from Raspberry Pi camera
   - Should return video stream from MediaPipe

3. **`/api/door`** - Control ESP32 servo motor
   - POST: `{ locked: boolean }` - Lock/unlock door
   - GET: Returns current door status

4. **`/api/light`** - Control ESP32 LED
   - POST: `{ on: boolean }` - Turn light on/off
   - GET: Returns current light status

5. **`/api/system/status`** - Check device health
   - Returns connection status for both Raspberry Pi and ESP32

## Setup Instructions

### Raspberry Pi Setup
1. Install MediaPipe and set up facial recognition
2. Create a REST API endpoint that returns face detection data
3. Update the API routes with your Pi's IP address

### ESP32 Setup
1. Program ESP32 to control servo motor and LED
2. Create HTTP endpoints for device control
3. Update the API routes with your ESP32's IP address

### Dashboard Setup
\`\`\`bash
npm install
npm run dev
\`\`\`

## Current Status

The dashboard currently runs in **simulation mode** with mock data. Once you implement the backend connections to your Raspberry Pi and ESP32, the dashboard will automatically switch to using real data.

A yellow warning banner will display when the backend is not connected.
