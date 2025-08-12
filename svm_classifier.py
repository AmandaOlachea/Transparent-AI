"""
svm_classifier.py

This script demonstrates a support vector machine (SVM) classifier with a linear kernel and
how to interpret its predictions. It trains an SVC on the Iris dataset and prints the
coefficients of the separating hyperplanes, the support vectors, and the contribution of each
feature to a sample prediction.

SVMs with linear kernels produce a linear decision boundary similar to linear regression.
The `coef_` attribute gives the weight of each feature for each class, and the support
vectors indicate which training samples define the boundary.
"""

from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np


def train_and_explain():
    # Load data
    X, y = load_iris(return_X_y=True)
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

    # Train SVM with linear kernel
    model = SVC(kernel='linear', decision_function_shape='ovr')
    model.fit(X, y)

    # Print shape of coefficients
    coef = model.coef_
    print(f"Coefficient matrix shape: {coef.shape} (n_classes x n_features)")
    for class_idx, class_coef in enumerate(coef):
        print(f"\nClass {class_idx} coefficients:")
        for name, weight in zip(feature_names, class_coef):
            print(f"  {name}: {weight:.3f}")

    # Show number of support vectors for each class
    print("\nNumber of support vectors for each class:", model.n_support_)

    # Explain a sample
    sample = X[0]
    pred = model.predict([sample])[0]
    print(f"\nPrediction for first sample: {pred}")

    # Compute contributions for the predicted class (one-vs-rest)
    class_coef = coef[pred]
    contributions = sample * class_coef
    print("Contributions of each feature to the decision function:")
    for name, contrib in sorted(zip(feature_names, contributions), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {contrib:.3f}")
    intercept = model.intercept_[pred]
    print(f"Bias term (intercept): {intercept:.3f}")


if __name__ == "__main__":
    train_and_explain()
