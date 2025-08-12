# Transparent-AI

This repository contains a simple example of an explainable text classification model built with Python and scikit‑learn. The aim is to demonstrate how to build a lightweight language model that not only makes predictions but also explains why it predicted a particular label.

## Project Structure

- `explainable_language_model.py` – Python script that trains a logistic regression classifier on a small set of positive and negative sentences and exposes an `explain_prediction` function. Logistic regression models assign a weight (coefficient) to each feature, which makes them highly interpretable. The script uses these coefficients to report the contribution of each word in the input sentence to the final decision.

## Requirements

The script requires Python 3 and scikit‑learn. You can install scikit‑learn via pip:

```bash
pip install scikit-learn
```

## Running the Script

To run the model interactively:

```bash
python explainable_language_model.py
```

The script will train the model on its sample data and then prompt you to enter sentences. For each sentence, it prints the predicted label (either “positive” or “negative”) and a list of words with their contribution scores.

Example usage:

```
> I am very happy with this product
Predicted label: positive
Top contributing words:
  happy: 0.80
  product: 0.40
```

A positive score means the word pushes the model towards the positive class; a negative score pushes it towards the negative class. You can customise the training data in the script by editing the `positive_texts` and `negative_texts` lists.

## Additional Transparent Models

In addition to the explainable text classifier, this repository includes several other transparent models:

- `transparent_language_model.py` – Implements a simple bigram language model. It trains on a small corpus and predicts the next word based on word bigram counts. The `explain_next_word` function prints the probability distribution over possible next words, so you can see exactly how the model chooses its output.

- `decision_tree_classifier.py` – Trains a decision tree classifier on the Iris dataset. The script prints the decision rules and feature importances, providing a transparent view of how the model makes decisions.

- `linear_regression_example.py` – Demonstrates a linear regression model on synthetic data. It prints the learned coefficients and intercept, and shows how each feature weight contributes to a sample prediction.

### Running the Additional Scripts

To run each of these scripts, use the following commands:

```bash
python transparent_language_model.py
```

The bigram language model will train on its built‑in sample text and then prompt you for a prefix. It outputs the probability of each candidate next word and selects the most probable one.

```bash
python decision_tree_classifier.py
```

This script fits a decision tree on the Iris dataset, prints the tree structure and feature importances, and then performs a sample prediction.

```bash
pyt

hon linear_regression_example.py
```

The linear regression script trains on synthetic data, prints the learned coefficients, and calculates the contribution of each feature to a sample prediction.

## About Transparent AI

Transparent AI, also known as explainable AI (XAI), is essential for building trust, ensuring accountability, and meeting regulatory requirements. By using interpretable models like logistic and linear regression, decision trees, and simple language models, we can provide clear explanations of how predictions are made. Understanding which features drive outcomes helps users detect bias, improve fairness, and confidently adopt AI technology.


### Additional Models Continued

- `random_forest_classifier.py` – Builds a random forest classifier on the Iris dataset and reports global feature importances. The script also approximates per‑feature contributions for a sample prediction by averaging the paths through the trees.
- `knn_classifier.py` – Trains a k‑nearest neighbors classifier and, for a test sample, prints the distances, indices and labels of the closest training points so you can see why a particular label was chosen.
- `svm_classifier.py` – Fits a support‑vector machine with a linear kernel and prints the coefficients for each class as well as the support vectors that define the decision boundary. It also shows how each feature weight contributes to a single prediction.
- `simple_neural_network.py` – Trains a small neural network on the Iris dataset using scikit‑learn’s MLPClassifier and explains its predictions using integrated gradients. Integrated gradients compute an attribution score for each feature by integrating the gradient of the output with respect to the input along a path from a baseline to the actual input.

#### Running These Scripts

To run these additional scripts:

```bash
python random_forest_classifier.py
```
This script outputs overall feature importances and approximate per‑feature contributions for a sample prediction.

```bash
python knn_classifier.py
```
The program prints the distances and labels of the three nearest neighbors for the first sample, illustrating how the prediction is determined.

```bash
python svm_classifier.py
```
It displays the weight vector for each class, the number of support vectors and the per‑feature contributions for a sample prediction.

```bash
python simple_neural_network.py
```
This runs the neural network example, prints the predicted class for the first sample and lists the integrated‑gradient attribution values for each input feature.

### Transparency in Complex Models

The AI landscape also includes powerful models like large language models, diffusion models, and other foundation models. These are typically deep neural networks with millions or billions of parameters and thus operate as black boxes. While it isn’t feasible to fully trace every parameter’s influence, researchers use techniques such as attention visualization, saliency maps, integrated gradients and feature‑importance analysis to provide partial explanations of why a model made a particular decision. Even so, the field of explainable AI acknowledges that as models grow in scale and complexity, achieving complete transparency is challenging. These interpretability tools provide insight but cannot match the direct interpretability of simpler models like linear regression or decision trees.

