import numpy as np
from sklearn.model_selection import train_test_split

def train_test_val_split(X,y,train_size,stratify=None):
    """generates a stratified train_test split

    Args:
        X (np.ndarray): features to split
        y (np.array): grounf truth to split
        train_size (float or int): fraction or absolute value of the training split compared to global data
        stratify (np.array, optional): labels to stratify. Defaults to None.

    Returns:
        tuple: tuples of train and test ground truth and feature values
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size,test_size=1-train_size, stratify=stratify)
    return X_train,X_test,y_train,y_test