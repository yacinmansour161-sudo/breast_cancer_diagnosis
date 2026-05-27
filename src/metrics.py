import numpy as np

def accuracy(y , y_hat):
    if len(y) == 0:
        return 0
    return np.sum(y == y_hat)/len(y)

def confusion_matrix(y,y_hat):
    tp = np.sum((y==1)&(y_hat==1))#true positive (malignant predicted malignant)
    tn = np.sum((y==0)&(y_hat==0))#true negative (benign predicted benign)
    fp = np.sum((y==0)&(y_hat==1))#false positive (benign predicted malignant)
    fn = np.sum((y==1)&(y_hat==0))#false negative (malignant predicted benign)
    return np.array([[tp,fp],
                     [fn,tn]])
#The size of a confusion matrix is determined entirely 
#by the number of classes (categories) you are trying to predict. 
#Specifically, if you have N classes, your confusion matrix will be an NxN matrix.

def precision(y,y_hat):
    tp = np.sum((y==1)&(y_hat==1))
    fp = np.sum((y==0)&(y_hat==1))
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)
#Among predicted malignant tumors: how many were actually malignant?

def recall(y,y_hat):
    tp = np.sum((y==1)&(y_hat==1))
    fn = np.sum((y==1)&(y_hat==0))
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)
#Among actual malignant tumors: how many were correctly predicted?

def f1_score(y,y_hat):
    p = precision(y,y_hat)
    r = recall(y,y_hat)
    if (p + r) == 0:
        return 0
    return 2*p*r/(p+r)
#F1-score provides a balanced evaluation between precision and recall, 
#making it particularly relevant for medical diagnosis tasks 
#where both false positives and false negatives are important.