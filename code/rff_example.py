import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# -------------------------------------------------
#  Load MNIST
# -------------------------------------------------
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(np.int32)

# -------------------------------------------------
#  Subset: 1k train / 1k test
# -------------------------------------------------
n_train = 1000
n_test  = 1000


rng = np.random.default_rng(911)
idx = rng.permutation(len(X))
train_idx = idx[:n_train]
test_idx  = idx[n_train:n_train + n_test]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

num_classes = len(np.unique(y_train))

# Scale pixels to [0, 1]
X_train = X_train.astype(np.float64) / 255.0
X_test  = X_test.astype(np.float64)  / 255.0

# One-hot targets (multi-class least squares)
Y_train = np.zeros((n_train, num_classes), dtype=np.float64)
Y_train[np.arange(n_train), y_train] = 1.0

Y_test = np.zeros((n_test, num_classes), dtype=np.float64)
Y_test[np.arange(n_test), y_test] = 1.0

# -------------------------------------------------
#  Random Fourier Features (Gaussian kernel)
#    For RBF bandwidth sigma:
#       w ~ N(0, sigma^{-2} I)
#       b ~ Uniform(0, 2pi)
#    z(x) = sqrt(2/N) * cos(xW + b)
# -------------------------------------------------
def sample_rff_params(d, N, sigma, rng):
    W = rng.normal(loc=0.0, scale=1.0 / sigma, size=(d, N))
    b = rng.uniform(low=0.0, high=2.0 * np.pi, size=(N,))
    return W, b

def rff_map(X, W, b):
    N = W.shape[1]
    return np.sqrt(2.0 / N) * np.cos(X @ W + b)

def mse(A, B):
    return float(np.mean((A - B) ** 2))

def acc_from_scores(S, y_true):
    # S: (n, C) scores
    y_pred = np.argmax(S, axis=1)
    return float(np.mean(y_pred == y_true))

# -------------------------------------------------
# Sweep over number of random features N
# -------------------------------------------------
Ns = np.unique(
    np.concatenate([
        np.linspace(50, 800, 15, dtype=int),
        np.linspace(900, 1200, 25, dtype=int),
        np.linspace(1300, 6000, 25, dtype=int),
    ])
)

sigma = 1  # bandwidth in pixel-space (with X scaled to [0,1])

train_sq = []
test_sq  = []
train_err = []
test_err  = []
coef_norm = []

d = X_train.shape[1]

for N in Ns:

    print(f"Running N = {N}")
    # sample (W,b) 
    W, b = sample_rff_params(d, N, sigma, rng)

    Ztr = rff_map(X_train, W, b)     # (n_train, N)
    Zte = rff_map(X_test,  W, b)     # (n_test,  N)

    # Minimum-norm ERM (ridgeless): solve Ztr A ≈ Y_train
    # A_hat: (N, C)
    A_hat, *_ = np.linalg.lstsq(Ztr, Y_train, rcond=None)

    # Scores
    Str = Ztr @ A_hat   # (n_train, C)
    Ste = Zte @ A_hat   # (n_test,  C)

    # Losses
    train_sq.append(mse(Str, Y_train))
    test_sq.append(mse(Ste, Y_test))

    train_err.append(1.0 - acc_from_scores(Str, y_train))
    test_err.append(1.0 - acc_from_scores(Ste, y_test))

    # ||A||_F is the natural multi-output analogue of ||a||_2
    coef_norm.append(float(np.linalg.norm(A_hat, ord="fro")))

x = Ns / 100

fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)

# Test 0-1 loss (log scale)
axes[0, 0].plot(x, np.array(test_err) * 100.0, marker="o", linewidth=1)
axes[0, 0].set_yscale("log")
axes[0, 0].set_title("Zero-one loss")
axes[0, 0].set_ylabel("Test (%)")

# Test squared loss (log scale)
axes[0, 1].plot(x, test_sq, marker="o", linewidth=1)
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("Squared loss")
axes[0, 1].set_ylabel("Test")

# Norm (log scale)
axes[1, 0].plot(x, coef_norm, marker="o", color="darkred", linewidth=1)
axes[1, 0].set_yscale("log")
axes[1, 0].set_ylabel("Norm")

axes[1, 1].plot(x, coef_norm, marker="o", color="darkred", linewidth=1)
axes[1, 1].set_yscale("log")
axes[1, 1].set_ylabel("Norm")

# Training risks
axes[2, 0].plot(x, np.array(train_err) * 100.0, marker="o", color="orange", linewidth=1)
axes[2, 0].set_ylabel("Train (%)")
axes[2, 0].set_xlabel(r"Number of Random Fourier Features ($\times 10^3$) (N)")

axes[2, 1].plot(x, train_sq, marker="o", color="orange", linewidth=1)
axes[2, 1].set_ylabel("Train")
axes[2, 1].set_xlabel(r"Number of Random Fourier Features ($\times 10^3$) (N)")

for ax in axes.flat:
    ax.axvline(n_train / 100, linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
