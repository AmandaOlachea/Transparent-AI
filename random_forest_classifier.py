"""
random_forest_classifier.py

This script demonstrates a random forest classifier and how to interpret its predictions.
It trains a RandomForestClassifier on the Iris dataset, reports the global feature importances
and approximates the contribution of each feature for a sample prediction.

Random forests are ensemble models made up of many decision trees. While individual trees are
easy to interpret, aggregating them reduces transparency. However, scikit‑learn exposes
`feature_importances_` which estimate how much each feature contributes to predictions.
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def train_and_explain():
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

    # Train random forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Global feature importances
    importances = model.feature_importances_
    print("Global feature importances:")
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {importance:.3f}")

    # Explain a sample prediction
    sample = X[0]
    pred = model.predict([sample])[0]
    print(f"\nPrediction for first sample: {pred}")

    # Approximate per-feature contribution for the sample using feature importances
    # This is a simplistic approach and does not capture interactions.
    contributions = sample * importances
    print("Approximate contributions for the first sample:")
    for name, contrib in sorted(zip(feature_names, contributions), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {contrib:.3f}")

    # Show the decision of each tree for the sample
    leaf_indices = model.apply([sample])
    print("\nLeaf indices for each tree in the forest (first five shown):")
    print(leaf_indices[0][:5])


if __name__ == "__main__":
    train_and_explain()
