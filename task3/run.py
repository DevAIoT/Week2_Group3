import time
import numpy as np
import tensorflow as tf

model_paths = {
    "Accelerometer Only": "model_accel.tflite",
    "Gyroscope Only": "model_gyro.tflite",
    "Smaller Model": "model_small.tflite",
    "Larger Model": "model_large.tflite"
}

for model_name, model_path in model_paths.items():
    print(f"Measuring latency for: {model_name}")

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assuming a single input tensor, get its shape and type
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Generate some dummy data matching the input shape and type
    # In a real scenario, you would use actual sensor data
    dummy_input_data = np.random.rand(*input_shape).astype(input_dtype)

    # Prepare the input tensor
    interpreter.set_tensor(input_details[0]['index'], dummy_input_data)

    # Run inference and measure time
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    # Get the output tensor (optional, but good for completeness)
    # output_data = interpreter.get_tensor(output_details[0]['index'])

    # Calculate latency
    latency_ms = (end_time - start_time) * 1000

    print(f"Inference latency for {model_name}: {latency_ms:.4f} ms\n")