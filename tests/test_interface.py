import unittest
from explanation_engine import explain

class ExplanationEngineTests(unittest.TestCase):
    def test_three_views_same_prediction(self):
        result = explain("terrible awful bad")
        self.assertEqual(result["predicted_label"], "negative")
        self.assertIn("negative", result["simple"])
        self.assertIn("negative", result["everyday"])

    def test_reconstruction(self):
        for phrase in ["great excellent", "terrible awful", "unknownwordonly"]:
            result = explain(phrase)
            self.assertLess(result["reconstruction_error"], 1e-8)
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)

    def test_input_validation(self):
        for phrase in ["", " ", None, "a" * 5001]:
            with self.assertRaises(ValueError):
                explain(phrase)

if __name__ == "__main__":
    unittest.main()
