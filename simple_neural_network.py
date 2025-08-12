"""
simple_neural_network.py

This script trains a simple neural network classifier on the iris dataset using scikit-learn's MLPClassifier.
It demonstrates a method for interpreting the model's predictions using integrated gradients. Integrated
gradients is an attribution technique that explains the importance of each input feature by
integrating the gradients of the model's output with respect to the input along a straight-line path
from a baseline input to the actual input.

Transparent AI and explainability:
Unlike many deep learning models that operate as black boxes, transparent AI aims to make model decisions
understandable. Integrated gradients, along with other attribution methods, helps uncover how each input
feature influences the network's decision. This script is intentionally simple to illustrate how even a small
neural network can be interpreted when equipped with an appropriate explanation technique.

References:
- Palo Alto Networks article on explainable AI, which emphasizes that transparency and interpretability are
  critical for building trust and accountability in AI systems.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def predict_prob(model, x):
    """Return predicted class probabilities for a single sample."""
    return model.predict_proba(x.reshape(1, -1))[0]


def gradient(model, x, pred_class, epsilon=1e-4):
    """Approximate the gradient of the predicted class probability with respect to the input features.

    Uses a symmetric finite difference method: f(x + eps) - f(x - eps) / (2*eps) for each dimension.
    """
    grads = np.zeros_like(x)
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = epsilon
        probs_plus = predict_prob(model, x + d)
        probs_minus = predict_prob(model, x - d)
        grads[i] = (probs_plus[pred_class] - probs_minus[pred_class]) / (2 * epsilon)
    return grads


def integrated_gradients(model, input_tensor, baseline=None, steps=50):
    """Compute integrated gradients for a single input example.

    Args:
        model: Trained scikit-learn classifier with predict_proba method.
        input_tensor (np.ndarray): The input features for which attributions are computed.
        baseline (np.ndarray): Baseline input to start from. If None, uses zero baseline.
        steps (int): Number of steps for the Riemann approximation of the integral.

    Returns:
        np.ndarray: Attribution scores for each input feature.
    """
    if baseline is None:
        baseline = np.zeros_like(input_tensor)
    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]
    # Determine the predicted class once
    pred_probs = predict_prob(model, input_tensor)
    pred_class = int(np.argmax(pred_probs))
    grads = []
    for scaled in scaled_inputs:
        grads.append(gradient(model, scaled, pred_class))
    avg_grads = np.mean(np.array(grads), axis=0)
    return (input_tensor - baseline) * avg_grads


def main():
    # Load iris dataset and standardize features
    X, y = load_iris(return_X_y=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a simple neural network classifier
    clf = MLPClassifier(hidden_layer_sizes=(5,), activation="relu", max_iter=500, random_state=0)
    clf.fit(X_scaled, y)

    # Choose the first sample for explanation
    sample = X_scaled[0]
    baseline = np.zeros_like(sample)

    attributions = integrated_gradients(clf, sample, baseline, steps=50)
    pred_class = clf.predict(sample.reshape(1, -1))[0]

    print(f"Predicted class for sample 0: {pred_class}")
    print("Integrated gradients feature attributions:")
    for idx, value in enumerate(attributions):
        print(f"Feature {idx}: {value:.4f}")


if __name__ == "__main__":
    main()
