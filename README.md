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
python linear_regression_example.py
```

The linear regression script trains on synthetic data, prints the learned coefficients, and calculates the contribution of each feature to a sample prediction.

## About Transparent AI

Transparent AI, also known as explainable AI (XAI), is essential for building trust, ensuring accountability, and meeting regulatory requirements. By using interpretable models like logistic and linear regression, decision trees, and simple language models, we can provide clear explanations of how predictions are made. Understanding which features drive outcomes helps users detect bias, improve fairness, and confidently adopt AI technology.
