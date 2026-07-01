import numpy as np
import torch
import torch.nn as nn


SCALE_BITS = 7
SCALE = 1 << SCALE_BITS  # 128


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
    """
    Convert float values to signed int8 fixed-point values.

    float 0.5 -> round(0.5 * 128) = 64
    """
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


def relu_int(x):
    return np.maximum(x, 0)


def fixed_point_inference_single(x_float, params):
    """
    Run fixed-point inference for one input sample.

    x_float shape: (64,)
    """
    W1_q = params["W1_q"]      # shape: (16, 64), int8
    b1_q = params["b1_q"]      # shape: (16,), int32
    W2_q = params["W2_q"]      # shape: (10, 16), int8
    b2_q = params["b2_q"]      # shape: (10,), int32

    # Quantize input to int8
    x_q = quantize_int8(x_float)  # shape: (64,)

    # -----------------------------
    # Layer 1: 64 -> 16
    # -----------------------------
    hidden_q = np.zeros(16, dtype=np.int32)

    for j in range(16):
        acc = np.int32(b1_q[j])

        for i in range(64):
            acc += np.int32(x_q[i]) * np.int32(W1_q[j, i])

        # Shift back from SCALE^2 to SCALE
        acc_shifted = acc >> SCALE_BITS

        # ReLU
        hidden_q[j] = max(acc_shifted, 0)

    # Optional clipping if you want hidden values to fit into int8 memory.
    # For now, keep hidden_q as int32 for accuracy and simplicity.
    # hidden_q = np.clip(hidden_q, -128, 127).astype(np.int8)

    # -----------------------------
    # Layer 2: 16 -> 10
    # -----------------------------
    output_q = np.zeros(10, dtype=np.int32)

    for k in range(10):
        acc = np.int32(b2_q[k])

        for j in range(16):
            acc += np.int32(hidden_q[j]) * np.int32(W2_q[k, j])

        # Shift back from SCALE^2 to SCALE
        output_q[k] = acc >> SCALE_BITS

    prediction = int(np.argmax(output_q))

    return prediction, output_q, hidden_q, x_q


def main():
    # -----------------------------
    # Load trained PyTorch model
    # -----------------------------
    model = TinyNN()
    model.load_state_dict(torch.load("data/tiny_nn_float.pth"))
    model.eval()

    # -----------------------------
    # Extract floating-point weights
    # -----------------------------
    state = model.state_dict()

    W1 = state["fc1.weight"].numpy()  # shape: (16, 64)
    b1 = state["fc1.bias"].numpy()    # shape: (16,)
    W2 = state["fc2.weight"].numpy()  # shape: (10, 16)
    b2 = state["fc2.bias"].numpy()    # shape: (10,)

    # -----------------------------
    # Quantize parameters
    # -----------------------------
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

    # -----------------------------
    # Load test data
    # -----------------------------
    X_test = np.load("data/X_test.npy").astype(np.float32)
    y_test = np.load("data/y_test.npy").astype(np.int64)
    X_test_q = quantize_int8(X_test)
    np.save("data/X_test_q.npy", X_test_q)
    print("\nSaved quantized inputs:")
    print("data/X_test_q.npy", X_test_q.shape, X_test_q.dtype)

    # -----------------------------
    # Compare floating-point and fixed-point accuracy
    # -----------------------------
    fixed_correct = 0
    float_correct = 0
    same_prediction = 0

    for idx in range(len(X_test)):
        x = X_test[idx]
        y = y_test[idx]

        # Floating-point PyTorch prediction
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            logits = model(x_tensor)
            float_pred = int(torch.argmax(logits, dim=1).item())

        # Fixed-point prediction
        fixed_pred, output_q, hidden_q, x_q = fixed_point_inference_single(x, params)

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

    # -----------------------------
    # Show one example for RTL testing
    # -----------------------------
    sample_idx = 0
    x = X_test[sample_idx]
    y = y_test[sample_idx]

    fixed_pred, output_q, hidden_q, x_q = fixed_point_inference_single(x, params)

    print("\nExample test sample:")
    print("True label:       ", y)
    print("Fixed prediction: ", fixed_pred)
    print("Quantized input:  ", x_q)
    print("Hidden values:    ", hidden_q)
    print("Output logits q:  ", output_q)


if __name__ == "__main__":
    main()
