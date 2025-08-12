"""
transparent_language_model.py

This script implements a simple transparent bigram language model. A bigram language model 
generates text by estimating the probability of the next word based on the previous word. 
Because it uses straightforward word frequency counts, every step of the model is 
transparent and explainable.

During training, the model builds a dictionary of bigram counts from a corpus of sentences. 
Given a previous word, the model computes the probability distribution over possible next words.
To generate text, it repeatedly samples a next word based on these probabilities.

The `explain_next_word` function returns a sorted list of candidate next words along with their 
probabilities, allowing the user to see exactly why the model chose a particular next word.
"""

from collections import defaultdict, Counter
import random
import re
from typing import List, Tuple

class BigramLanguageModel:
    def __init__(self):
        # A mapping from a word to a counter of next words
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess a string into a list of lowercase word tokens.
        Non-alphabetic characters are treated as delimiters.

        Args:
            text: The text to preprocess.

        Returns:
            A list of tokens.
        """
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        return tokens

    def train(self, corpus: List[str]) -> None:
        """
        Train the bigram model on a list of sentences.

        Args:
            corpus: A list of sentences (strings).
        """
        for text in corpus:
            tokens = self.preprocess(text)
            for w1, w2 in zip(tokens[:-1], tokens[1:]):
                self.bigram_counts[w1][w2] += 1
                self.vocab.add(w1)
                self.vocab.add(w2)

    def get_next_word_distribution(self, previous_word: str):
        """
        Get the probability distribution of next words given a previous word.

        If the previous word is unseen, a uniform distribution over the vocabulary is returned.

        Args:
            previous_word: The previous word token.

        Returns:
            A dictionary mapping each candidate next word to its probability.
        """
        counts = self.bigram_counts.get(previous_word, None)
        if not counts:
            # unseen word: assign equal probability to all words in the vocabulary
            return {w: 1/len(self.vocab) for w in self.vocab}

        total = sum(counts.values())
        distribution = {word: count / total for word, count in counts.items()}
        return distribution

    def generate_sentence(self, start_word: str, length: int = 10) -> str:
        """
        Generate a sentence of a given length starting from a given word.

        Args:
            start_word: The first word of the generated sentence.
            length: Number of words in the generated sentence.

        Returns:
            A generated sentence string.
        """
        if not self.vocab:
            raise ValueError("Model has not been trained yet.")

        sentence = [start_word]
        current_word = start_word
        for _ in range(length - 1):
            dist = self.get_next_word_distribution(current_word)
            # sample the next word according to the probability distribution
            words = list(dist.keys())
            probs = list(dist.values())
            next_word = random.choices(words, weights=probs, k=1)[0]
            sentence.append(next_word)
            current_word = next_word
        return " ".join(sentence)

    def explain_next_word(self, previous_word: str) -> List[Tuple[str, float]]:
        """
        Explain the possible next words given a previous word by returning a sorted list of 
        candidate words and their probabilities.

        Args:
            previous_word: The previous word token.

        Returns:
            A list of tuples (word, probability) sorted by probability in descending order.
        """
        dist = self.get_next_word_distribution(previous_word)
        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return sorted_dist

if __name__ == "__main__":
    """
    Example usage:
    Train the model on a small corpus and generate sentences with explanations.
    """

    training_corpus = [
        "Hello world, welcome to transparent AI.",
        "Explainable AI aims to make AI systems transparent and interpretable.",
        "AI models should provide reasons for their decisions.",
        "We are building a simple language model that is easy to understand.",
    ]

    model = BigramLanguageModel()
    model.train(training_corpus)

    # Starting word for generation
    start_word = "ai"
    print(f"Generating a sentence starting with '{start_word}':")
    print(model.generate_sentence(start_word=start_word, length=10))
    print("\nExplanation of next word probabilities after 'ai':")
    explanation = model.explain_next_word(start_word)
    for word, prob in explanation:
        print(f"{word}: {prob:.3f}")
