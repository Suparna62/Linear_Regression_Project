# ==========================================
# MINI PROJECT: LINEAR REGRESSION (METALLURGY)
# Dataset: Data.csv
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------------------
# STEP 1: LOAD DATA
# ------------------------------
data = pd.read_csv("Data.csv")

print("\nColumns in dataset:", data.columns)

# ------------------------------
# STEP 2: CLEAN DATA
# ------------------------------
data['Su'] = pd.to_numeric(data['Su'], errors='coerce')
data['Sy'] = pd.to_numeric(data['Sy'], errors='coerce')

# Drop only rows where Su or Sy is missing
data = data.dropna(subset=['Su', 'Sy'])

print("Number of data points:", len(data))

# ------------------------------
# STEP 3: SELECT VARIABLES
# ------------------------------
X = data[['Su']].values
y = data['Sy'].values

# ------------------------------
# STEP 4: SKLEARN MODEL
# ------------------------------
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

# Equation
m = model.coef_[0]
c = model.intercept_

print("\n=== SKLEARN MODEL ===")
print(f"Equation: y = {m:.2f}x + {c:.2f}")

# ------------------------------
# STEP 5: PLOT RESULT
# ------------------------------
plt.figure()
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Regression Line")
plt.xlabel("Ultimate Strength (Su)")
plt.ylabel("Yield Strength (Sy)")
plt.title("Linear Regression (Sklearn)")
plt.legend()
plt.show()

# ------------------------------
# STEP 6: GRADIENT DESCENT
# ------------------------------
def gradient_descent(X, y, lr, epochs, batch_size):
    m = 0
    c = 0
    n = len(X)

    X = X.flatten()

    for epoch in range(epochs):
        # Shuffle data
        indices = np.arange(n)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Batch training
        for i in range(0, n, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            y_pred = m * X_batch + c

            dm = (-2/len(X_batch)) * np.sum(X_batch * (y_batch - y_pred))
            dc = (-2/len(X_batch)) * np.sum(y_batch - y_pred)

            m = m - lr * dm
            c = c - lr * dc

    return m, c

# ------------------------------
# STEP 7: HYPERPARAMETER TEST
# ------------------------------
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [1, 10, len(X)]
epochs_list = [50, 100]

print("\n=== HYPERPARAMETER RESULTS ===")

for lr in learning_rates:
    for batch in batch_sizes:
        for epochs in epochs_list:
            m_gd, c_gd = gradient_descent(X.copy(), y.copy(), lr, epochs, batch)
            print(f"LR={lr}, Batch={batch}, Epochs={epochs} → y = {m_gd:.2f}x + {c_gd:.2f}")

# ------------------------------
# STEP 8: PLOT COMPARISON (Learning Rate)
# ------------------------------
plt.figure()
plt.scatter(X, y, label="Actual Data")

for lr in learning_rates:
    m_gd, c_gd = gradient_descent(X.copy(), y.copy(), lr, 100, len(X))
    y_gd = m_gd * X.flatten() + c_gd
    plt.plot(X, y_gd, label=f"LR={lr}")

plt.xlabel("Ultimate Strength (Su)")
plt.ylabel("Yield Strength (Sy)")
plt.title("Learning Rate Comparison")
plt.legend()
plt.show()

# ------------------------------
# STEP 9: FINAL CONCLUSION
# ------------------------------
print("\n=== CONCLUSION ===")
print("Linear regression successfully models relationship between Su and Sy.")
print("Learning rate affects convergence speed.")
print("Batch size affects stability of training.")
print("Epochs affect final model accuracy.")