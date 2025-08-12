"""
decision_tree_classifier.py

This script demonstrates a transparent decision tree classifier. The model uses a simple dataset and trains a DecisionTreeClassifier. Decision trees are naturally interpretable because each decision is based on a single feature and threshold. The script prints the structure of the tree and the feature importances.
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text


def train_and_explain():
    # Load iris dataset
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    # Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    clf.fit(X, y)

    # Print the tree in a readable text form
    tree_rules = export_text(clf, feature_names=feature_names)
    print("Decision tree rules:")
    print(tree_rules)

    # Show feature importances
    importances = clf.feature_importances_
    print("Feature importances:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.3f}")

    # Predict and explain a sample
    sample = X[0].reshape(1, -1)
    pred = clf.predict(sample)[0]
    print(f"\nPrediction for first sample: {target_names[pred]}")


if __name__ == "__main__":
    train_and_explain()
