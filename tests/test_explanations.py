"""Tests for the explanations, not just whether the scripts run."""
import unittest
import numpy as np

import explainable_language_model as sentiment
from transparent_language_model import BigramLanguageModel


class ExplainabilityTests(unittest.TestCase):
    def test_sentiment_explains_both_classes(self):
        for phrase in ("excellent amazing great", "terrible awful bad"):
            label, contributions = sentiment.explain_prediction(phrase)
            self.assertIn(label, ("positive", "negative"))
            self.assertGreater(len(contributions), 0)

    def test_sentiment_contributions_reconstruct_decision_score(self):
        for phrase in ("excellent great", "terrible awful"):
            x = sentiment.vectorizer.transform([phrase])
            label, contributions = sentiment.explain_prediction(phrase)
            # The binary model's decision function uses coefficients for classes_[1].
            direction = 1 if label == sentiment.model.classes_[1] else -1
            score = direction * sentiment.model.decision_function(x)[0]
            intercept = direction * sentiment.model.intercept_[0]
            self.assertAlmostEqual(sum(value for _, value in contributions) + intercept, score, places=6)

    def test_bigram_probabilities_sum_to_one(self):
        model = BigramLanguageModel()
        model.train(["hello world", "hello there", "hello world"])
        distribution = model.get_next_word_distribution("hello")
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertAlmostEqual(distribution["world"], 2 / 3)

    def test_bigram_explanation_matches_distribution(self):
        model = BigramLanguageModel()
        model.train(["blue sky", "blue ocean"])
        self.assertEqual(dict(model.explain_next_word("blue")), model.get_next_word_distribution("blue"))


if __name__ == "__main__":
    unittest.main()
