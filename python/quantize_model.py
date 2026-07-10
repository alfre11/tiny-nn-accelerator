import os
import numpy as np
import torch
import torch.nn as nn


SCALE_BITS = 7
SCALE = 1 << SCALE_BITS  # 128
DATA_DIR = "data"


class TinyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def quantize_int8(x_float):
    """Convert float values to signed int8 fixed-point values."""
    x_q = np.round(x_float * SCALE)
    x_q = np.clip(x_q, -128, 127)
    return x_q.astype(np.int8)


def quantize_bias_int32(b_float):
    """
    Bias is added to the accumulator before shifting.

    input_q * weight_q has scale SCALE^2, so bias must also use SCALE^2.
    """
    b_q = np.round(b_float * SCALE * SCALE)
    return b_q.astype(np.int32)


def fixed_point_inference_single(x_float, params):
    """Run fixed-point inference for one float input sample."""
    W1_q = params["W1_q"]
    b1_q = params["b1_q"]
    W2_q = params["W2_q"]
    b2_q = params["b2_q"]

    x_q = quantize_int8(x_float)

    # Layer 1: 64 -> 16
    hidden_q = np.zeros(16, dtype=np.int32)

    for hidden_idx in range(16):
        acc = np.int32(b1_q[hidden_idx])

        for input_idx in range(64):
            acc += np.int32(x_q[input_idx]) * np.int32(W1_q[hidden_idx, input_idx])

        hidden_q[hidden_idx] = max(acc >> SCALE_BITS, 0)

    # Layer 2: 16 -> 10
    output_q = np.zeros(10, dtype=np.int32)

    for output_idx in range(10):
        acc = np.int32(b2_q[output_idx])

        for hidden_idx in range(16):
            acc += np.int32(hidden_q[hidden_idx]) * np.int32(W2_q[output_idx, hidden_idx])

        output_q[output_idx] = acc >> SCALE_BITS

    prediction = int(np.argmax(output_q))
    return prediction, output_q, hidden_q, x_q


def save_quantized_arrays(params, X_test_q, data_dir=DATA_DIR):
    """Save arrays using the filenames expected by fixed_point_reference.py."""
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "W1_q.npy"), params["W1_q"])
    np.save(os.path.join(data_dir, "b1_q.npy"), params["b1_q"])
    np.save(os.path.join(data_dir, "W2_q.npy"), params["W2_q"])
    np.save(os.path.join(data_dir, "b2_q.npy"), params["b2_q"])
    np.save(os.path.join(data_dir, "X_test_q.npy"), X_test_q)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load trained floating-point model.
    model = TinyNN()
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "tiny_nn_float.pth")))
    model.eval()

    # Extract floating-point weights and biases.
    state = model.state_dict()
    W1 = state["fc1.weight"].numpy()  # shape: (16, 64)
    b1 = state["fc1.bias"].numpy()    # shape: (16,)
    W2 = state["fc2.weight"].numpy()  # shape: (10, 16)
    b2 = state["fc2.bias"].numpy()    # shape: (10,)

    # Quantize weights and biases.
    params = {
        "W1_q": quantize_int8(W1),
        "b1_q": quantize_bias_int32(b1),
        "W2_q": quantize_int8(W2),
        "b2_q": quantize_bias_int32(b2),
    }

    print("Quantized parameter shapes:")
    print("W1_q:", params["W1_q"].shape, params["W1_q"].dtype)
    print("b1_q:", params["b1_q"].shape, params["b1_q"].dtype)
    print("W2_q:", params["W2_q"].shape, params["W2_q"].dtype)
    print("b2_q:", params["b2_q"].shape, params["b2_q"].dtype)

    # Load test data and quantize inputs.
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(np.int64)
    X_test_q = quantize_int8(X_test)

    # Save files expected by fixed_point_reference.py.
    save_quantized_arrays(params, X_test_q, DATA_DIR)

    print("\nSaved quantized arrays:")
    print("data/W1_q.npy", params["W1_q"].shape, params["W1_q"].dtype)
    print("data/b1_q.npy", params["b1_q"].shape, params["b1_q"].dtype)
    print("data/W2_q.npy", params["W2_q"].shape, params["W2_q"].dtype)
    print("data/b2_q.npy", params["b2_q"].shape, params["b2_q"].dtype)
    print("data/X_test_q.npy", X_test_q.shape, X_test_q.dtype)

    # Optional sanity check: compare floating-point and fixed-point predictions.
    fixed_correct = 0
    float_correct = 0
    same_prediction = 0

    for idx in range(len(X_test)):
        x = X_test[idx]
        y = int(y_test[idx])

        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            logits = model(x_tensor)
            float_pred = int(torch.argmax(logits, dim=1).item())

        fixed_pred, _, _, _ = fixed_point_inference_single(x, params)

        if float_pred == y:
            float_correct += 1
        if fixed_pred == y:
            fixed_correct += 1
        if fixed_pred == float_pred:
            same_prediction += 1

    n = len(X_test)
    print("\nAccuracy comparison:")
    print(f"Floating-point accuracy: {float_correct / n * 100:.2f}%")
    print(f"Fixed-point accuracy:    {fixed_correct / n * 100:.2f}%")
    print(f"Prediction match rate:   {same_prediction / n * 100:.2f}%")

    # Show one example for RTL testing.
    sample_idx = 0
    fixed_pred, output_q, hidden_q, x_q = fixed_point_inference_single(X_test[sample_idx], params)

    print("\nExample test sample:")
    print("Sample index:      ", sample_idx)
    print("True label:        ", int(y_test[sample_idx]))
    print("Fixed prediction:  ", fixed_pred)
    print("Quantized input:   ", x_q)
    print("Hidden values:     ", hidden_q)
    print("Output logits q:   ", output_q)


if __name__ == "__main__":
    main()