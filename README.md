# Classification Task as Contextual Multi-Arms Bandits

The project explores framing classification tasks as contextual multi-armed bandit problems, comparing linear, Bayesian, and neural exploration strategies, and extends the framework to a Markov Decision Process (MDP) for active feature acquisition.

## Repository Contents

- `report.pdf`: Full report detailing the theoretical formulations, experimental results, and the MDP extension.
- `Bandits_Classification.ipynb`: Jupyter notebook containing the algorithms' implementations and experimental evaluations.

## Problem & Datasets

Classification is mapped to a contextual bandit setting where each class corresponds to an arm, feature vectors represent contexts, and the agent receives a binary reward (1 for a correct prediction, 0 otherwise). Experiments are conducted on two datasets:

- **Breast Cancer**: Binary classification, 30 features.
- **Handwritten Digits**: 10 classes, 64 features.

## Implemented Algorithms

- **LinUCB**: Disjoint linear models with UCB-based exploration.
- **Thompson Sampling**: Bayesian exploration assuming linear payoffs with Gaussian priors.
- **NeuralUCB**: Off-policy PyTorch implementation using a fully connected neural network with UCB exploration, diagonal matrix approximation, and context normalization.
- **Baselines**: Uniform random, Memory-based, and Logistic Regression.

## Summary of Results

- **Breast Cancer**: Linear methods (LinUCB and Thompson Sampling) achieve strong performance, with Thompson Sampling matching Logistic Regression (2.1% test error) at a low computational cost.
- **Digits**: Under partial feedback and higher task complexity, NeuralUCB outperforms all baselines (2.4% test error), though at a significantly higher computational and memory cost than Thompson Sampling (5.1% test error).
- **MDP Extension**: The report also formulates a Fitted-Q Iteration approach to solve the problem as an MDP where the agent learns to classify contexts while minimizing the number of revealed features.

## Conclusion
Model performance strongly depends on task complexity. While LinUCB performs well on simple binary tasks, its linear assumptions cause it to degrade on more complex reward structures. NeuralUCB overcomes this limitation through richer reward representation and achieves the best results on complex tasks, but its instability, sensitivity to hyperparameters, and high computational and memory costs make it unsuitable for simpler problems. Overall, Thompson Sampling stands out as the most balanced and reliable choice across both datasets, leveraging Bayesian exploration to deliver strong cumulative reward and regret with significantly lower resource requirements.

## Requirements

To run the notebook, you will need Python 3 along with standard data science and deep learning libraries:

```bash
pip install numpy scikit-learn matplotlib torch jupyter
```

## Author

Gwenn Garrigues-Journaud
