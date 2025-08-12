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
