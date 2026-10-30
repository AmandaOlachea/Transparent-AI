# Transparent AI: Understanding Why a Model Makes a Prediction

AI can produce an answer that looks convincing while giving us very little insight into how it got there. I wanted to explore what happens when we can actually inspect the calculations, features, or examples behind a prediction.

This repository is a collection of small, runnable explainable AI examples. Some models are inherently easier to inspect, while others need attribution methods that only give us part of the picture. I think that distinction matters, especially when AI is being considered for decisions that affect people.

## Run it yourself

You'll need Python 3.10 or newer. From the repository folder:

```bash
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run an example:

```bash
python explainable_language_model.py
```

Enter a sentence to see the predicted sentiment and which known words contributed to that prediction.

Or try the count-based language model:

```bash
python transparent_language_model.py
```

It shows the probabilities of possible next words based on word pairs in its training corpus.

## Interactive explanation interface

The repository includes a local browser interface that shows **three levels of explanation from the same real model output**:

- **Explain like I'm 5:** plain-language explanation based on the actual supporting word features.
- **Everyday:** what evidence influenced the prediction, plus its limitations.
- **Technical:** fitted feature weights, word counts, signed contributions, intercept, decision score, reconstructed score, reconstruction error, estimated class probabilities, and complete JSON evidence.

Start the interface after installing dependencies:

```bash
python app.py
```

Open **http://127.0.0.1:8765** in your browser. Type any sentence and choose an explanation level. The app runs locally and binds only to the loopback interface.

The three views share a single prediction record. The plain-language explanation is derived from actual contributing features rather than a canned explanation for each class. The technical view lets you verify that the word contributions and intercept reconstruct the classifier's decision score.

This interface currently explains the repository's **binary sentiment classifier**. The other model examples remain available as separate scripts; they are not yet wired into the interface. It is not a general-purpose conversational LLM.

To run automated checks, including the explanation-engine tests:

```bash
python -m unittest discover -s tests -v
```

**Privacy:** the local server is intended for demonstrations with synthetic or nonsensitive text. It does not provide authentication or production security controls.

## The models

| File | What you can inspect |
| --- | --- |
| `explainable_language_model.py` | Word contributions to a binary logistic regression decision |
| `transparent_language_model.py` | Bigram counts and next-word probabilities |
| `decision_tree_classifier.py` | Decision rules and feature importance |
| `linear_regression_example.py` | Coefficients and prediction components |
| `random_forest_classifier.py` | Feature importance and approximate contributions |
| `knn_classifier.py` | Nearest training examples and distances |
| `svm_classifier.py` | Linear weights and support vectors |
| `simple_neural_network.py` | Approximate integrated-gradient attributions |

Run any example with `python filename.py`.

## Check the explanations

I added tests because a model producing an explanation doesn't mean that explanation is correct.

```bash
python -m unittest discover -s tests -v
```

The tests check both predicted classes of the sentiment model, verify that its feature contributions reconstruct its decision score, and check bigram probabilities against the observed counts. GitHub Actions runs the same tests on pushes and pull requests.

See [what these explanations mean](docs/what-the-explanations-mean.md) for limitations and how I'd evaluate them.

## The black box problem

A small logistic regression model lets us reconstruct its decision score from coefficients. A bigram model lets us see exactly how the next-word probabilities were counted. Those are useful forms of transparency.

A neural network is more complicated. Integrated gradients and other attribution methods can highlight influential inputs, but they do not reveal every internal mechanism or prove that a model's stated explanation is faithful.

The goal here is to make model behavior more inspectable, test the explanations, and document what the methods can and cannot show. I want other people to be able to run the same examples and challenge the results.

## References

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [NIST Four Principles of Explainable AI](https://www.nist.gov/publications/four-principles-explainable-artificial-intelligence)
