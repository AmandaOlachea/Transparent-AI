"""
Explainable Text Classification using Logistic Regression.

This script trains a simple logistic regression classifier to classify text into
positive or negative sentiment. Logistic regression is an interpretable model
where each feature has a weight (coefficient) that indicates how strongly it
contributes to the decision.  The coefficients can be extracted to explain
predictions.

We use these weights to explain predictions by showing how each word in the input
contributes to the predicted label.

Run the script and enter a sentence to see the predicted label and
contribution scores for each word.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Prepare a small training dataset
positive_texts = [
    "I love this product, it works great!",
    "This is an amazing experience",
    "The service was excellent and friendly",
    "Great quality and fantastic support",
    "I am very happy with this purchase",
    "The experience was pleasant and enjoyable",
    "Excellent results and very positive outcome",
    "The movie was fun and entertaining",
    "What a wonderful experience, I had a great time",
    "The team did a great job and I am satisfied",
]

negative_texts = [
    "This product is terrible and broke quickly",
    "I hate this service, it was awful",
    "The experience was bad and unpleasant",
    "Poor quality and unfriendly support",
    "I am disappointed with this purchase",
    "The movie was boring and a waste of time",
    "This was a horrible experience, not good at all",
    "I am unhappy with the results",
    "The team did a bad job and I am dissatisfied",
    "This service is not good and I regret using it",
]

train_texts = positive_texts + negative_texts
train_labels = ["positive"] * len(positive_texts) + ["negative"] * len(negative_texts)

# Initialize the vectorizer and model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

def explain_prediction(text: str):
    """
    Predict the label for the given text and return an explanation.

    Args:
        text: A string containing the text to classify.

    Returns:
        A tuple (predicted_label, contributions) where contributions is a list of
        (word, score) pairs sorted by absolute contribution score in descending order.
        The score is computed as coefficient * count for each word in the input.
    """
    X_test = vectorizer.transform([text])
    predicted_label = model.predict(X_test)[0]

    # Determine which row of coefficients to use based on predicted class
    class_index = list(model.classes_).index(predicted_label)
    coeffs = model.coef_[class_index]
    feature_names = vectorizer.get_feature_names_out()
    counts = X_test.toarray()[0]

    # Compute contribution for each word
    contributions = {}
    for idx, count in enumerate(counts):
        if count > 0:
            word = feature_names[idx]
            contributions[word] = coeffs[idx] * count

    # Sort contributions by absolute value descending
    sorted_contribs = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    return predicted_label, sorted_contribs

if __name__ == "__main__":
    print("Enter a sentence to classify (press enter on an empty line to exit).")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        label, contribs = explain_prediction(line)
        print(f"Predicted label: {label}")
        if contribs:
            print("Top contributing words:")
            for word, score in contribs:
                print(f"  {word}: {score:.3f}")
        else:
            print("No known words from the vocabulary were found in the input.")
        print()
