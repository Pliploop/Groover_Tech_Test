from numpy import average
from numpy.lib.function_base import _average_dispatcher
import sklearn
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def score_classifier(classifier,X,y):
    preds = classifier.predict(X)
    return accuracy_score(preds,y), precision_score(preds,y,average='macro'), recall_score(preds,y,average="macro"), f1_score(preds,y,average='macro')

def microscore_classifier(classifier,X,y,name):
    preds = classifier.predict(X)
    return precision_score(preds,y,average=None), recall_score(preds,y,average=None),f1_score(preds,y,average=None),[name for k in range(len(precision_score(preds,y,average=None)))]


def plot_confusion_matrix(classifier,X,y,classes_dic,ax):
    ConfusionMatrixDisplay.from_predictions(y_true=y,y_pred=classifier.predict(X),ax=ax,cmap='viridis',colorbar=False)
    ax.set_xticklabels(classes_dic)
    ax.set_yticklabels(classes_dic)   
    
