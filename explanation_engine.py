"""Grounded explanations of the existing sentiment classifier."""
from explainable_language_model import model, vectorizer

def explain(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter text to analyze.")
    if len(text) > 5000:
        raise ValueError("Maximum input length is 5000 characters.")
    x = vectorizer.transform([text])
    classes = [str(v) for v in model.classes_]
    score = float(model.decision_function(x)[0])
    probabilities = model.predict_proba(x)[0]
    label = classes[1] if score >= 0 else classes[0]
    direction = 1 if label == classes[1] else -1
    words = vectorizer.get_feature_names_out()
    counts = x.toarray()[0]
    features = [
        {"word": str(words[i]), "count": int(counts[i]),
         "weight": float(model.coef_[0][i]),
         "contribution": float(model.coef_[0][i] * counts[i])}
        for i in range(len(words)) if counts[i]
    ]
    features.sort(key=lambda f: abs(f["contribution"]), reverse=True)
    intercept = float(model.intercept_[0])
    reconstructed = intercept + sum(f["contribution"] for f in features)
    supporting = [f for f in features if direction * f["contribution"] > 0]
    opposing = [f for f in features if direction * f["contribution"] < 0]
    evidence = ", ".join('"' + f["word"] + '"' for f in supporting[:3])
    if evidence:
        simple = f"The computer noticed {evidence}. In its training examples, those words pushed it toward the {label} label. It guessed {label} for this message, but it could be wrong."
        everyday = f"The model predicted {label}. Its strongest supporting word-level evidence was {evidence}. These associations come from a small training dataset and may not generalize."
    else:
        simple = f"The computer guessed {label}, but it didn't find words that clearly pushed it toward that answer. Its starting preference may have mattered. It could be wrong."
        everyday = f"The model predicted {label}. No recognized words contributed positively toward this label. The intercept and any opposing features determine the result."
    return {
        "input": text, "model": "CountVectorizer + binary LogisticRegression",
        "predicted_label": label, "classes": classes,
        "probabilities": {classes[0]: float(probabilities[0]), classes[1]: float(probabilities[1])},
        "predicted_probability": float(probabilities[1] if direction == 1 else probabilities[0]),
        "decision_score": score, "intercept": intercept,
        "reconstructed_score": reconstructed, "reconstruction_error": abs(score - reconstructed),
        "features": features, "supporting_features": supporting, "opposing_features": opposing,
        "simple": simple, "everyday": everyday,
        "limitations": "These probabilities are model estimates, not validated real-world confidence. Word coefficients explain the model score, not human intent or causality."
    }
