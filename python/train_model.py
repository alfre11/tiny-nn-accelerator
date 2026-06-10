import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Tiny neural network: 64 -> 16 -> 10
# -----------------------------
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


def main():
    # Make results repeatable
    torch.manual_seed(0)
    np.random.seed(0)

    # -----------------------------
    # Load sklearn digits dataset
    # -----------------------------
    digits = load_digits()

    # X shape: (1797, 64)
    # y shape: (1797,)
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    # Normalize pixel values.
    # Digits pixels are originally from 0 to 16.
    X = X / 16.0

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # -----------------------------
    # Create model
    # -----------------------------
    model = TinyNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # -----------------------------
    # Train
    # -----------------------------
    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        logits = model(X_train)
        loss = criterion(logits, y_train)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test)
                predictions = torch.argmax(test_logits, dim=1)
                accuracy = (predictions == y_test).float().mean().item()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Loss: {loss.item():.4f}, "
                f"Test Accuracy: {accuracy * 100:.2f}%"
            )

    # -----------------------------
    # Final accuracy
    # -----------------------------
    model.eval()
    with torch.no_grad():
        train_preds = torch.argmax(model(X_train), dim=1)
        train_acc = (train_preds == y_train).float().mean().item()

        test_preds = torch.argmax(model(X_test), dim=1)
        test_acc = (test_preds == y_test).float().mean().item()

    print("\nFinal Results")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%")

    # -----------------------------
    # Save model weights
    # -----------------------------
    os.makedirs("data", exist_ok=True)

    torch.save(model.state_dict(), "data/tiny_nn_float.pth")

    # Also save test data for later quantization / RTL testing
    np.save("data/X_test.npy", X_test.numpy())
    np.save("data/y_test.npy", y_test.numpy())

    print("\nSaved:")
    print("data/tiny_nn_float.pth")
    print("data/X_test.npy")
    print("data/y_test.npy")


if __name__ == "__main__":
    main()