# What these explanations actually tell us

I want people to be able to inspect a prediction, but I also want to be clear about what each method proves.

## Logistic regression

The text classifier uses bag-of-words counts. For binary logistic regression, the decision score is the intercept plus each token count multiplied by its coefficient. The explanation shows those token contributions. The scores are in log-odds units, not probabilities. Unknown words do not appear in the model's vocabulary, and correlated features can make individual weights misleading.

The tests check whether the contributions plus the intercept reconstruct the model's decision score.

## Bigram language model

The next-word probabilities come directly from observed word-pair counts in a small corpus. Those counts are inspectable. This example does not explain the internal behavior of a transformer-based LLM.

## Decision trees, neighbors, and linear models

Tree paths, nearest neighbors, and linear coefficients expose useful parts of their decision procedures. Each has limitations. For example, the nearest neighbors depend on the distance metric and feature scaling.

## Random forests and neural networks

Feature importance and integrated gradients are attribution methods. They can help investigate behavior, but they are not proof of causation or complete access to the model's internal decision process. Explanations should be tested for stability and fidelity.

## What I would measure next

- Prediction quality on held-out data
- Explanation reconstruction or fidelity where mathematically possible
- Stability under small input changes
- Sensitivity to feature correlations and baselines
- Performance differences across meaningful subgroups when appropriate data is available

References: [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) and [NIST Explainable AI principles](https://www.nist.gov/publications/four-principles-explainable-artificial-intelligence).
