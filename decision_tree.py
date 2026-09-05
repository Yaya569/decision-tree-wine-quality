import math
import numpy as np
import pandas as pd


PREDICT_LABEL = "quality"


class Node:
    def __init__(
        self,
        in_cat=None,
        in_subcat=None,
        in_label=None,
        in_left=None,
        in_right=None,
    ):
        self.cat = in_cat
        self.subcat = in_subcat
        self.label = in_label
        self.left = in_left
        self.right = in_right

    def node_return(self):
        return self.label is not None


def load_data(filename):
    """Load the Wine Quality dataset."""
    return pd.read_csv(filename, sep=";")


def entropy(labels):
    """Calculate entropy for a set of class labels."""
    total = len(labels)

    if total == 0:
        return 0

    counts = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    entropy_value = 0

    for count in counts.values():
        probability = count / total
        entropy_value -= probability * math.log2(probability)

    return entropy_value


def information_gain(parent_labels, left_labels, right_labels):
    """Calculate information gain for a potential split."""
    parent_entropy = entropy(parent_labels)

    total = len(parent_labels)
    left_size = len(left_labels)
    right_size = len(right_labels)

    if total == 0:
        return 0

    weighted_entropy = (
        (left_size / total) * entropy(left_labels)
        + (right_size / total) * entropy(right_labels)
    )

    return parent_entropy - weighted_entropy


def build_tree(data, depth, stopping_depth):
    """Recursively build a decision tree using information gain."""
    labels = data[PREDICT_LABEL].tolist()

    # Stop if the maximum depth is reached
    # or all observations belong to the same class.
    if depth >= stopping_depth or len(set(labels)) <= 1:
        majority = max(set(labels), key=labels.count)
        return Node(in_label=majority)

    best_gain = 0
    best_feature = None
    best_threshold = None
    best_left = None
    best_right = None

    features = [
        column for column in data.columns
        if column != PREDICT_LABEL
    ]

    for feature in features:
        values = data[feature]
        thresholds = set(values)

        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = ~left_mask

            left = data[left_mask]
            right = data[right_mask]

            if len(left) == 0 or len(right) == 0:
                continue

            gain = information_gain(
                labels,
                left[PREDICT_LABEL].tolist(),
                right[PREDICT_LABEL].tolist(),
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_left = left
                best_right = right

    # If no useful split is found, return the majority class.
    if best_gain <= 0:
        majority = max(set(labels), key=labels.count)
        return Node(in_label=majority)

    left_subtree = build_tree(
        best_left,
        depth + 1,
        stopping_depth,
    )

    right_subtree = build_tree(
        best_right,
        depth + 1,
        stopping_depth,
    )

    return Node(
        in_cat=best_feature,
        in_subcat=best_threshold,
        in_left=left_subtree,
        in_right=right_subtree,
    )


def predict(tree, row):
    """Predict the class for a single observation."""
    if tree.node_return():
        return tree.label

    feature_value = row[tree.cat]

    if feature_value <= tree.subcat:
        return predict(tree.left, row)

    return predict(tree.right, row)


def get_predictions(tree, test_df):
    """Generate predictions for every row in a test dataset."""
    return [
        predict(tree, row)
        for _, row in test_df.iterrows()
    ]


def accuracy(predictions, true_labels):
    """Calculate classification accuracy."""
    correct = sum(
        prediction == true_label
        for prediction, true_label
        in zip(predictions, true_labels)
    )

    return correct / len(true_labels)
