# decision-tree-wine-quality
Decision tree classification from scratch using entropy and information gain to predict wine quality.

## Project Overview

This project implements a decision tree classifier from scratch using entropy and information gain to predict wine quality.

The original wine quality scores range from 3 to 8. For this project, the target variable was transformed into a binary classification problem:

- **1** — wine quality score ≥ 6
- **0** — wine quality score < 6

The decision tree was implemented without using a pre-built machine learning classification library. The project explores how different tree depths affect classification performance on unseen data.

## Dataset

The project uses the **Wine Quality – Red Wine** dataset, which contains physicochemical measurements of red wine samples together with their quality scores.

The dataset includes variables such as:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality

The original quality score ranges from 3 to 8. The target variable was converted into a binary classification outcome, with scores of 6 or above classified as higher quality.

## Methodology

The decision tree classifier was implemented from scratch using Python, pandas and NumPy.

The main steps were:

1. Load and preprocess the wine quality dataset.
2. Convert the original quality score into a binary classification target.
3. Split the dataset into **80% training data and 20% test data**.
4. Calculate **entropy** to measure the impurity of each node.
5. Calculate **information gain** to identify the best feature and threshold for each split.
6. Recursively build the decision tree.
7. Apply different stopping depths of **2, 3 and 4**.
8. Generate predictions on the test dataset.
9. Evaluate model performance using classification accuracy.

10. ## Results

Three decision trees were trained using different stopping depths to compare how model complexity affected performance.

| Stopping Depth | Test Accuracy |
|----------------|---------------|
| 2 | 73.125% |
| 3 | 73.4375% |
| 4 | 76.875% |

The model achieved its highest test accuracy at a stopping depth of 4, with an accuracy of **76.875%**.

As the tree depth increased from 2 to 4, test accuracy also increased. This suggests that the deeper trees were able to capture more detailed patterns in the wine characteristics while still maintaining reasonable performance on unseen data.

However, test accuracy alone is not sufficient to determine whether a model is overfitting or underfitting. Comparing training and test performance would provide a more complete assessment of model generalisation.

## Key Takeaways

- Implemented a decision tree classifier from scratch using entropy and information gain.
- Converted wine quality prediction into a binary classification problem.
- Used an 80/20 train-test split with a fixed random state for reproducibility.
- Compared decision tree depths of 2, 3 and 4.
- Achieved a test accuracy of **76.875%** at depth 4.
- Explored the relationship between model complexity and generalisation performance.

## Tools & Technologies

- Python
- pandas
- NumPy
- Machine Learning
- Decision Trees
- Entropy
- Information Gain
- Classification
