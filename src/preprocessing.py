import pandas as pd
import numpy as np
def split_data(X,y,test_size=0.2):
    n=len(X) #number of samples in the dataset
    indices= np.random.permutation(n) #generate shuffled indices from 0 to n-1
    test_samples = int(n * test_size)#calculate number of samples for test set 
    test_indices= indices[:test_samples]#select the first 'test_samples' indices for the test set
    train_indices= indices[test_samples:]#select the remaining indices for the training set
    X_train, y_train = X[train_indices], y[train_indices]#select the training data and labels using the training indices
    X_test, y_test = X[test_indices], y[test_indices]#select the test data and labels using the test indices
    return X_train, y_train, X_test, y_test

def standardize_data(X_train,X_test):
    mean=np.mean(X_train,axis=0) #calculate mean of each feature in training data
    std = np.std(X_train, axis=0)#calculate standard deviation of each feature in training data
    std[std==0]=1 #to avoid division by zero, if std is zero, set it to 1
    X_train_scaled = (X_train - mean)/std #standardize training data
    X_test_scaled = (X_test - mean)/std #standardize test data using mean and std from training data
    return X_train_scaled,X_test_scaled

def load_preprocess_data(path):
    df = pd.read_csv(path)
    df=df.drop(columns=['id',"Unnamed: 32"])
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}) 
    X = df.drop(columns=['diagnosis']).values
    y = df["diagnosis"].values
    X_train, y_train, X_test, y_test = split_data(X,y)
    X_train,X_test = standardize_data(X_train,X_test)
    return X_train, y_train, X_test, y_test 
