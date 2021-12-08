import sklearn
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score

def score_classifier(classifier,X,y):
    preds = classifier.predict(X)
    return accuracy_score(preds,y), precision_score(preds,y,average='macro'), recall_score(preds,y,average="macro"), f1_score(preds,y,average='macro')