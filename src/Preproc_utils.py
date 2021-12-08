import numpy as np
from sklearn.model_selection import train_test_split

def train_test_val_split(X,y,train_size):
    X_train, X_temp, y_train, y_temp = train_test_split(X,y,train_size=train_size,test_size=1-train_size, stratify=y) 
    X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,train_size=.5,test_size=.5, stratify=y_temp) 
    return X_train,X_test,X_val,y_train,y_test,y_val