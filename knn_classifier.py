"""
knn_classifier.py

This script demonstrates a k-nearest neighbors classifier and how to interpret its predictions.
It trains a KNeighborsClassifier on the Iris dataset and shows the nearest neighbors that
influenced a sample prediction.

K-nearest neighbors is a simple, instance-based learning algorithm. For each input, the
model finds the k closest training samples in the feature space and predicts the most common label.
By examining the distances and labels of the neighbors, you can see why a particular label was chosen.
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


def train_and_explain():
    # Load data
    X, y = load_iris(return_X_y=True)
    # Train KNN classifier with 3 neighbors
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    # Choose a sample to explain
    sample = X[0]
    pred = model.predict([sample])[0]
    print(f"Prediction for first sample: {pred}")

    # Find nearest neighbors and their distances
    distances, indices = model.kneighbors([sample])
    print("Nearest neighbors (distance, index, label):")
    for dist, idx in zip(distances[0], indices[0]):
        print(f"  Distance: {dist:.3f}, Index: {idx}, Label: {y[idx]}")


if __name__ == "__main__":
    train_and_explain()
