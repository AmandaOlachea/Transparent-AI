"""
linear_regression_example.py

This script demonstrates a transparent linear regression model. It trains a linear regression on a synthetic dataset and explains the contributions of each feature to a prediction.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def train_and_explain():
    np.random.seed(0)
    X = np.random.rand(100, 3)
    true_coefs = np.array([2.0, -3.5, 1.0])
    y = X @ true_coefs + 0.5

    model = LinearRegression()
    model.fit(X, y)

    print("Learned coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    sample = X[0]
    pred = model.predict(sample.reshape(1, -1))[0]
    contributions = sample * model.coef_
    print(f"\nPrediction for first sample: {pred:.3f}")
    print("Contributions:")
    for i, contrib in enumerate(contributions):
        print(f"  Feature {i}: {contrib:.3f}")
    print(f"Bias (intercept): {model.intercept_:.3f}")


if __name__ == "__main__":
    train_and_explain()
