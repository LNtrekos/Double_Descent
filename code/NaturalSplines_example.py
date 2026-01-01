import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Generate data ---
np.random.seed(911)
n_train = 20
x_train = np.random.uniform(-5, 5, n_train)
x_train.sort()
eps = np.random.normal(0, 0.3, n_train)
y_train = np.sin(x_train) + eps
len(x_train) ; len(y_train)

x_test = np.random.uniform(-5, 5, 10000)
y_true = np.sin(x_test)
len(x_train) ; len(y_train)

d = 20 # df 

natural_spline = SplineTransformer(
    degree = 3, n_knots = d,
    knots="quantile", extrapolation="linear", # THIS makes it a natural spline
    include_bias=False
)

# Design matrices
X_train = natural_spline.fit_transform(x_train.reshape(-1, 1))
X_test  = natural_spline.transform(x_test.reshape(-1, 1))

# Linear regression on spline basis
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_hat = model.predict(X_test)

mse = mean_squared_error(y_true, y_hat)
print("Test MSE:", mse)

b2_norm = sum(model.coef_**2)

# --- Plot ----
plt.figure(figsize=(8,5))

plt.scatter(x_train, y_train, color="black", s=30, label="Training data")
plt.plot(np.sort(x_test), y_hat[np.argsort(x_test)],
    color="orange", lw=2, label="Natural spline fit"
)
plt.plot(np.sort(x_test), np.sin(np.sort(x_test)), 
    color="blue", lw=2, linestyle="dashed",
    label="True sin(x)"
)

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Natural Spline with df={d}")
plt.show()


degress_to_fit = np.concatenate([
    np.arange(2, 21, 2),
    np.arange(21, 30, 1),
    np.arange(35, 101, 5)
])

print(degress_to_fit)

train_mse_counter = [] ; test_mse_counter = [] ; b2_norm_counter = []
for d in degress_to_fit:
    natural_spline = SplineTransformer(
        degree = 3, n_knots = d,
        knots="quantile", extrapolation="linear", # THIS makes it a natural spline
        include_bias=True
    )

    # Design matrices
    X_train = natural_spline.fit_transform(x_train.reshape(-1, 1))
    X_test  = natural_spline.transform(x_test.reshape(-1, 1))

    # Linear regression on spline basis
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    # Predictions
    y_hat_train = model.predict(X_train)
    y_hat = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_hat_train) 
    test_mse = mean_squared_error(y_true, y_hat)
    
    b2_norm = sum(model.coef_**2)

    train_mse_counter.append(train_mse) 
    test_mse_counter.append(test_mse)  
    b2_norm_counter.append(b2_norm)

degrees = degress_to_fit

# --- MSE (train and test) vs degrees of freedom ---
plt.figure(figsize=(7,4))

plt.plot(degrees, test_mse_counter, marker="o", label="Test MSE")
plt.plot(degrees, train_mse_counter, marker="o", color="orange", label="Train MSE")

plt.xlabel("Degrees of freedom (d)")
plt.ylabel("MSE")
plt.title("Train and Test MSE vs Degrees of Freedom")

plt.ylim(-0.1, 2)

plt.xticks([0, 4, 8, 20, 40, 60, 80, 100])

plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --- ||beta||^2 vs degrees of freedom ---
plt.figure(figsize=(7,4))
plt.plot(degrees, b2_norm_counter, marker="o", color="darkred")
plt.xlabel("Degrees of freedom (d)")
plt.ylabel(r"$\sum_j \hat\beta_j^2$")
plt.title("Squared coefficient norm vs Degrees of Freedom")
plt.grid(alpha=0.3)
plt.ylim(0, 100)
plt.xticks([0, 4, 8, 20, 40, 60, 80, 100])
plt.show()
