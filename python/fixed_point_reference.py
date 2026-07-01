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
    x_q = np.round(x_float * SCALE)
    x_q = np.clip(x_q, -128, 127)
    return x_q.astype(np.int8)


def quantize_bias_int32(b_float):
    b_q = np.round(b_float * SCALE * SCALE)
    return b_q.astype(np.int32)

def main():
    X_test_q = np.load("data/X_test_q.npy").astype(np.int8)
    y_test = np.load("data/y_test.npy").astype(np.int64)

    sample_idx = 0
    x_q = X_test_q[sample_idx]   # shape: (64,), dtype int8
    y = y_test[sample_idx]

    model = TinyNN()
    model.load_state_dict(torch.load("data/tiny_nn_float.pth"))
    model.eval()

    state = model.state_dict()

    W1 = state["fc1.weight"].numpy()
    b1 = state["fc1.bias"].numpy()
    W2 = state["fc2.weight"].numpy()
    b2 = state["fc2.bias"].numpy()

    params = {
        "W1_q": quantize_int8(W1),
        "b1_q": quantize_bias_int32(b1),
        "W2_q": quantize_int8(W2),
        "b2_q": quantize_bias_int32(b2),
    }

    # print(x_q)
    # print(y)
    pred, output_q, hidden_q = fixed_point_inference_single_q(x_q, params)
    print(pred)
    print(output_q)
    print(hidden_q)

def fixed_point_inference_single_q(x_q, params):
    W1_q = params["W1_q"]
    b1_q = params["b1_q"]
    W2_q = params["W2_q"]
    b2_q = params["b2_q"]

    hidden_q = np.zeros(16, dtype=np.int32)

    for j in range(16):
        acc = np.int32(b1_q[j])
        for i in range(64):
            acc += np.int32(x_q[i]) * np.int32(W1_q[j, i])
        hidden_q[j] = max(acc >> SCALE_BITS, 0)

    output_q = np.zeros(10, dtype=np.int32)

    for k in range(10):
        acc = np.int32(b2_q[k])
        for j in range(16):
            acc += np.int32(hidden_q[j]) * np.int32(W2_q[k, j])
        output_q[k] = acc >> SCALE_BITS

    prediction = int(np.argmax(output_q))
    return prediction, output_q, hidden_q

if __name__ == "__main__":
    main()