import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microtorch_py.engine import Tensor
from microtorch_py.loss import binary_cross_entropy
from microtorch_py.nn import MLP
from microtorch_py.optim import SGD


def make_dataset(n_per_class=80, seed=0):
    rng = np.random.default_rng(seed)

    class_a = rng.normal(loc=[-2.0, 0.0], scale=0.8, size=(n_per_class, 2))
    class_b = rng.normal(loc=[2.0, 0.0], scale=0.8, size=(n_per_class, 2))

    X = np.vstack([class_a, class_b]).astype(float)
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=float)
    return X, y


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    Z = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            sample = np.array([xx[i, j], yy[i, j]], dtype=float)
            x_t = Tensor(sample, requires_grad=False)
            Z[i, j] = float(model(x_t).sigmoid().data[0])

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, levels=20, alpha=0.8, cmap="coolwarm")

    class_0 = X[y == 0]
    class_1 = X[y == 1]
    plt.scatter(class_0[:, 0], class_0[:, 1], c="blue", s=20, label="class 0")
    plt.scatter(class_1[:, 0], class_1[:, 1], c="red", s=20, label="class 1")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.title("2D Classification")
    plt.tight_layout()
    plt.savefig("2d_classification_boundary.png")
    print("Saved plot to 2d_classification_boundary.png")


def main():
    X, y = make_dataset()

    print("2D toy classification example")
    print("dataset shape:", X.shape)
    print("labels:", y[:5], "...", y[-5:])

    model = MLP(n_input=2, n_output=1, hiddens=[10, 10])
    optimizer = SGD(model.parameters(), lr=0.05)
    losses = []

    for _ in range(300):
        total_loss = 0.0
        optimizer.zero_grad()

        for sample, target in zip(X, y):
            x_t = Tensor(sample, requires_grad=False)
            target_t = Tensor(target, requires_grad=False)
            logits = model(x_t)
            loss = binary_cross_entropy(logits, target_t)
            total_loss += float(loss.data)
            loss.backward()

        optimizer.step()
        losses.append(total_loss / len(X))

    print("\nFinal loss:", losses[-1])

    print("\nSample predictions after training:")
    for sample, label in zip(X[:5], y[:5]):
        x_t = Tensor(sample, requires_grad=False)
        pred = float(model(x_t).sigmoid().data[0])
        print(f"x={sample}, label={int(label)}, probability={pred:.4f}")

    plot_decision_boundary(model, X, y)


if __name__ == "__main__":
    main()
