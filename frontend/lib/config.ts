// Backend configuration
export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000"

export const API_ENDPOINTS = {
  camera: `${BACKEND_URL}/api/camera`,
  faceRecognition: `${BACKEND_URL}/api/face-recognition`,
  door: `${BACKEND_URL}/api/door`,
  light: `${BACKEND_URL}/api/light`,
  systemStatus: `${BACKEND_URL}/api/system/status`,
} as const
