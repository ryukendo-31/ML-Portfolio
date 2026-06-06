import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_idx = feature_idx  # Changed to feature_idx to match your traversal
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None
    
class decisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_feat, best_thresh = self._best_split(X, y, n_features)

        if best_feat is None:
            leaf_value = self._most_common_label(y) # Fixed: Added (y) and underscore
            return Node(value=leaf_value)
            
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        
        left_child = self.grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self.grow_tree(X[right_idxs, :], y[right_idxs], depth + 1) # Fixed: Removed underscore
        return Node(best_feat, best_thresh, left_child, right_child)

    def _best_split(self, X, y, n_features):
        best_gini = float('inf')
        split_idx, split_thresh = None, None
        
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for thr in thresholds:
                gini = self._evaluate_split(X_column, y, thr)
                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_thresh = thr
                    
        return split_idx, split_thresh

    def _evaluate_split(self, X_column, y, split_thresh):
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return float('inf')
            
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        
        weighted_gini = (n_l / n) * g_l + (n_r / n) * g_r
        return weighted_gini

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1.0 - np.sum(probabilities ** 2)
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node): # Fixed: Changed parameters to (x, node)
        if node.is_leaf_node():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate our custom Tree
clf = decisionTree(max_depth=5)
clf.fit(X_train, y_train)

# Predict and test
predictions = clf.predict(X_test)
print(f"Custom Tree Accuracy: {accuracy_score(y_test, predictions):.4f}")