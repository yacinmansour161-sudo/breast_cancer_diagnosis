import numpy as np
from utils import sigmoid, euclidean_dist, information_gain, entropy
from metrics import accuracy,confusion_matrix,precision,recall,f1_score
from collections import Counter
import matplotlib.pyplot as plt

class BaseModel:
    def fit(self,X,y):
        raise NotImplementedError
    def predict(self,X):
        raise NotImplementedError

class LogisticRegression(BaseModel):
    def __init__(self,lr=0.01,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights = None
        self.bias=None
        self.losses = []
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias=0
        for _ in range(self.n_iters):
            linear_model = np.dot(X,self.weights)+self.bias
            y_pred = sigmoid(linear_model)
            dw = (1/n_samples)*np.dot(X.T,(y_pred - y))
            db = (1/n_samples)*np.sum(y_pred - y)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            loss = -np.mean(y*np.log(y_pred + 1e-6) + (1-y)*np.log(1-y_pred + 1e-6))
            self.losses.append(loss)

    def predict(self,X):
        linear_model = np.dot(X,self.weights)+self.bias
        y_pred = sigmoid(linear_model)
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.savefig("../figures/loss_curves.png")
        return np.where(y_pred >=0.5,1,0)
    def evaluate(self,X,y):
        y_pred = self.predict(X)
        return {
            'model': 'Logistic Regression',
            'accuracy': accuracy(y,y_pred),
            'confusion_matrix': confusion_matrix(y,y_pred),
            'precision': precision(y,y_pred),
            'recall': recall(y,y_pred),
            'f1_score': f1_score(y,y_pred)
        }

class KNN(BaseModel):
    def __init__(self,k=3):
        self.k=k
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    def predict(self,X):
        predictions=[]
        for x in X:
            d = [euclidean_dist(x, x_train)for x_train in self.X_train]
#for a single test point x , compute distance to every training point
            k_indices = np.argsort(d)[:self.k] 
#sort indices by distance an take first k smallest distances
            k_labels  = self.y_train[k_indices]
            values,counts = np.unique (k_labels,return_counts=True)
#extract labels of the nearest neighbors and then majority votings win !
            predictions.append(values[np.argmax(counts)])
        return np.array(predictions)
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            "model": "KNN",
            "accuracy": accuracy(y, y_pred),
            "precision": precision(y, y_pred),
            "recall": recall(y, y_pred),
            "f1_score": f1_score(y, y_pred),
            "confusion_matrix": confusion_matrix(y, y_pred)
        }

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None
    def best_split(self, X, y):
        best_gain = -1
        split_idx, split_val = None, None
        n_samples, n_features = X.shape
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                gain = information_gain(y,y[left_mask],y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_val = threshold
        return split_idx, split_val
    def build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_labels = len(np.unique(y))
        if (depth >= self.max_depth or n_labels == 1 or n_samples < 2):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        feature, threshold = self.best_split(X, y)
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feature=feature,threshold=threshold,left=left_child,right=right_child)
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {"model": "Decision Tree",
                "accuracy": accuracy(y, y_pred),
                "precision": precision(y, y_pred),
                "recall": recall(y, y_pred),
                "f1_score": f1_score(y, y_pred),
                "confusion_matrix": confusion_matrix(y, y_pred)
            }
                