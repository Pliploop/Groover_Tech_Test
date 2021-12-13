import numpy as np
from sklearn.model_selection import train_test_split

def train_test_val_split(X,y,train_size,stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size,test_size=1-train_size, stratify=stratify)
    return X_train,X_test,y_train,y_test