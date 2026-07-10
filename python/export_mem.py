import os
import numpy as np


SCALE_BITS = 7
SCALE = 1 << SCALE_BITS


def int_to_twos_complement_hex(value, bit_width):
    """
    Convert a signed integer to two's complement hex string.

    Example:
    -1 with 8 bits  -> FF
    -244 with 32 bits -> FFFFFF0C
    """
    if value < 0:
        value = (1 << bit_width) + value

    hex_width = bit_width // 4
    return f"{value:0{hex_width}X}"


def write_mem_file(filename, array, bit_width):
    """
    Write a NumPy array to a .mem file, one hex value per line.
    The array is flattened in row-major order.
    """
    flat_array = array.flatten()

    with open(filename, "w") as f:
        for value in flat_array:
            value_int = int(value)
            hex_value = int_to_twos_complement_hex(value_int, bit_width)
            f.write(hex_value + "\n")


def fixed_point_inference_single_q(x_q, params):
    """
    Runs the same fixed-point inference as fixed_point_reference.py.
    This is used to generate expected RTL outputs.
    """
    W1_q = params["W1_q"]
    b1_q = params["b1_q"]
    W2_q = params["W2_q"]
    b2_q = params["b2_q"]

    hidden_q = np.zeros(16, dtype=np.int32)

    # Layer 1: 64 -> 16
    for j in range(16):
        acc = np.int32(b1_q[j])

        for i in range(64):
            acc += np.int32(x_q[i]) * np.int32(W1_q[j, i])

        hidden_q[j] = max(acc >> SCALE_BITS, 0)

    output_q = np.zeros(10, dtype=np.int32)

    # Layer 2: 16 -> 10
    for k in range(10):
        acc = np.int32(b2_q[k])

        for j in range(16):
            acc += np.int32(hidden_q[j]) * np.int32(W2_q[k, j])

        output_q[k] = acc >> SCALE_BITS

    prediction = np.array([int(np.argmax(output_q))], dtype=np.int32)

    return hidden_q, output_q, prediction


def main():
    os.makedirs("data/mem", exist_ok=True)

    # -----------------------------
    # Load quantized model parameters
    # -----------------------------
    W1_q = np.load("data/W1_q.npy").astype(np.int8)
    b1_q = np.load("data/b1_q.npy").astype(np.int32)
    W2_q = np.load("data/W2_q.npy").astype(np.int8)
    b2_q = np.load("data/b2_q.npy").astype(np.int32)

    # -----------------------------
    # Load quantized test input
    # -----------------------------
    X_test_q = np.load("data/X_test_q.npy").astype(np.int8)
    y_test = np.load("data/y_test.npy").astype(np.int64)

    sample_idx = 0
    x_q = X_test_q[sample_idx]
    true_label = np.array([int(y_test[sample_idx])], dtype=np.int32)

    params = {
        "W1_q": W1_q,
        "b1_q": b1_q,
        "W2_q": W2_q,
        "b2_q": b2_q,
    }

    # -----------------------------
    # Generate expected RTL outputs
    # -----------------------------
    hidden_q, output_q, prediction = fixed_point_inference_single_q(x_q, params)

    # -----------------------------
    # Write .mem files for RTL
    # -----------------------------
    write_mem_file("data/mem/input.mem", x_q, 8)

    write_mem_file("data/mem/weights_l1.mem", W1_q, 8)
    write_mem_file("data/mem/biases_l1.mem", b1_q, 32)

    write_mem_file("data/mem/weights_l2.mem", W2_q, 8)
    write_mem_file("data/mem/biases_l2.mem", b2_q, 32)

    write_mem_file("data/mem/expected_hidden.mem", hidden_q, 32)
    write_mem_file("data/mem/expected_output.mem", output_q, 32)
    write_mem_file("data/mem/expected_prediction.mem", prediction, 32)
    write_mem_file("data/mem/true_label.mem", true_label, 32)

    # -----------------------------
    # Print summary
    # -----------------------------
    print("Exported .mem files to data/mem/")
    print()
    print("Sample index:       ", sample_idx)
    print("True label:         ", int(true_label[0]))
    print("Expected prediction:", int(prediction[0]))
    print("Expected hidden:    ", hidden_q)
    print("Expected output:    ", output_q)

    print("\nFiles written:")
    print("data/mem/input.mem")
    print("data/mem/weights_l1.mem")
    print("data/mem/biases_l1.mem")
    print("data/mem/weights_l2.mem")
    print("data/mem/biases_l2.mem")
    print("data/mem/expected_hidden.mem")
    print("data/mem/expected_output.mem")
    print("data/mem/expected_prediction.mem")
    print("data/mem/true_label.mem")


if __name__ == "__main__":
    main()