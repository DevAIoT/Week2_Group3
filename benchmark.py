import tflite_runtime.interpreter as tflite
import numpy as np
import tensorflow as tf
import time, os

# Load test set
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = (x_test / 255.0).astype(np.float32)

def benchmark_model(model_path, int8=False, n_samples=1000):
    print(f"\n--- Benchmarking {os.path.basename(model_path)} ---")
    intr = tflite.Interpreter(model_path=model_path)
    intr.allocate_tensors()

    in_det  = intr.get_input_details()[0]
    out_det = intr.get_output_details()[0]
    scale, zp = in_det['quantization']

    correct, total_time = 0, 0.0
    for i in range(n_samples):
        x = x_test[i]
        if len(in_det['shape']) == 4:     # CNNs expect (1,28,28,1)
            x = x[..., None]
        if int8:
            xq = np.clip(np.round(x / scale + zp), 0, 255).astype(np.uint8)
            intr.set_tensor(in_det['index'], xq[None, ...])
        else:
            intr.set_tensor(in_det['index'], x[None, ...].astype(np.float32))

        t1 = time.perf_counter_ns()
        intr.invoke()
        t2 = time.perf_counter_ns()

        pred = intr.get_tensor(out_det['index'])[0]
        if np.argmax(pred) == y_test[i]:
            correct += 1
        total_time += (t2 - t1) / 1e6  # convert ns → ms

    acc = correct / n_samples
    avg_ms = total_time / n_samples
    print(f"Accuracy: {acc:.4f},  Avg inference: {avg_ms:.2f} ms\n")
    return acc, avg_ms

# Loop through all .tflite models
models = [f for f in os.listdir("export") if f.endswith(".tflite")]
for m in sorted(models):
    int8 = "_int8" in m
    benchmark_model(os.path.join("export", m), int8=int8)