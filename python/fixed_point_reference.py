import os
import numpy as np


SCALE_BITS = 7
SCALE = 1 << SCALE_BITS  # 128
DATA_DIR = "data"
DEBUG_SAMPLE_IDX = 0


def load_quantized_params(data_dir=DATA_DIR):
    """
    Load quantized weights and biases created by quantize_model.py.

    Expected files:
        W1_q.npy: shape (16, 64), dtype int8
        b1_q.npy: shape (16,), dtype int32
        W2_q.npy: shape (10, 16), dtype int8
        b2_q.npy: shape (10,), dtype int32
    """
    params = {
        "W1_q": np.load(os.path.join(data_dir, "W1_q.npy")).astype(np.int8),
        "b1_q": np.load(os.path.join(data_dir, "b1_q.npy")).astype(np.int32),
        "W2_q": np.load(os.path.join(data_dir, "W2_q.npy")).astype(np.int8),
        "b2_q": np.load(os.path.join(data_dir, "b2_q.npy")).astype(np.int32),
    }

    check_param_shapes(params)
    return params


def check_param_shapes(params):
    """Catch shape mistakes early before running inference."""
    expected = {
        "W1_q": (16, 64),
        "b1_q": (16,),
        "W2_q": (10, 16),
        "b2_q": (10,),
    }

    for name, expected_shape in expected.items():
        actual_shape = params[name].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} has shape {actual_shape}, expected {expected_shape}"
            )


def fixed_point_inference_single_q(x_q, params):
    """
    Run integer-only inference for one already-quantized input sample.

    This function mirrors the planned RTL behavior:
        - int8 input values
        - int8 weights
        - int32 biases and accumulators
        - shift right by SCALE_BITS after each layer dot product
        - ReLU after layer 1 only
        - argmax over final output logits
    """
    W1_q = params["W1_q"]
    b1_q = params["b1_q"]
    W2_q = params["W2_q"]
    b2_q = params["b2_q"]

    x_q = x_q.astype(np.int8)

    # -----------------------------
    # Layer 1: 64 -> 16
    # -----------------------------
    hidden_q = np.zeros(16, dtype=np.int32)

    for hidden_idx in range(16):
        acc = np.int32(b1_q[hidden_idx])

        for input_idx in range(64):
            acc += np.int32(x_q[input_idx]) * np.int32(W1_q[hidden_idx, input_idx])

        # Convert from SCALE^2 back to SCALE, then apply ReLU.
        acc_shifted = acc >> SCALE_BITS
        hidden_q[hidden_idx] = max(acc_shifted, 0)

    # -----------------------------
    # Layer 2: 16 -> 10
    # -----------------------------
    output_q = np.zeros(10, dtype=np.int32)

    for output_idx in range(10):
        acc = np.int32(b2_q[output_idx])

        for hidden_idx in range(16):
            acc += np.int32(hidden_q[hidden_idx]) * np.int32(W2_q[output_idx, hidden_idx])

        # Convert from SCALE^2 back to SCALE.
        output_q[output_idx] = acc >> SCALE_BITS

    prediction = int(np.argmax(output_q))
    return prediction, output_q, hidden_q


def evaluate_test_set(X_test_q, y_test, params):
    """Run fixed-point inference on the full test set and return accuracy."""
    correct = 0
    predictions = []

    for idx in range(len(X_test_q)):
        pred, _, _ = fixed_point_inference_single_q(X_test_q[idx], params)
        predictions.append(pred)

        if pred == int(y_test[idx]):
            correct += 1

    accuracy = correct / len(y_test)
    return accuracy, np.array(predictions, dtype=np.int64)


def print_debug_sample(sample_idx, X_test_q, y_test, params):
    """Print one detailed sample for later RTL comparison."""
    x_q = X_test_q[sample_idx]
    y = int(y_test[sample_idx])

    pred, output_q, hidden_q = fixed_point_inference_single_q(x_q, params)

    print("\nExample test sample:")
    print(f"Sample index:      {sample_idx}")
    print(f"True label:        {y}")
    print(f"Fixed prediction:  {pred}")
    print("Quantized input:  ", x_q)
    print("Hidden values:    ", hidden_q)
    print("Output logits q:  ", output_q)


def main():
    # Load quantized test inputs and labels.
    X_test_q = np.load(os.path.join(DATA_DIR, "X_test_q.npy")).astype(np.int8)
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(np.int64)

    if X_test_q.ndim != 2 or X_test_q.shape[1] != 64:
        raise ValueError(f"X_test_q should have shape (N, 64), got {X_test_q.shape}")

    if len(X_test_q) != len(y_test):
        raise ValueError(
            f"X_test_q has {len(X_test_q)} samples but y_test has {len(y_test)} labels"
        )

    # Load already-quantized weights and biases.
    params = load_quantized_params(DATA_DIR)

    print("Loaded quantized parameter shapes:")
    for name, array in params.items():
        print(f"{name}: shape={array.shape}, dtype={array.dtype}")

    # Validate across the full test set.
    accuracy, predictions = evaluate_test_set(X_test_q, y_test, params)
    print("\nFixed-point validation:")
    print(f"Test samples:          {len(y_test)}")
    print(f"Correct predictions:   {int((predictions == y_test).sum())}")
    print(f"Fixed-point accuracy:  {accuracy * 100:.2f}%")

    # Print one detailed sample for RTL debugging.
    print_debug_sample(DEBUG_SAMPLE_IDX, X_test_q, y_test, params)


if __name__ == "__main__":
    main()