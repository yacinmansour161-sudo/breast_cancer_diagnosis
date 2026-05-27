import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
#transforms raw linear output into probabilities between 0 and 1
#perfect for binary classification

def euclidean_dist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
#heart of KNN, comparing distances between samples
# (closest points influence prediction)

def plot_loss_curve(losses):
    plt.plot(losses)
    plt.title("loss curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
#measures impurity of a dataset, used in decision trees to find best splits
#0 means pure (all samples same class), 
#higher values mean more mixed classes

def information_gain(y, left_y, right_y):
    p = len(left_y) / len(y)
    return entropy(y) - (p * entropy(left_y) + (1 - p) * entropy(right_y))
#measures how much a split reduces impurity, 
#used to find best splits in decision trees