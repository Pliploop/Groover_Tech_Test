from numpy import average
from numpy.lib.function_base import _average_dispatcher
import sklearn
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def score_classifier(classifier,X,y):
    """scores a classifier using macroaverage metrics for multiclass classification"

    Args:
        classifier (sklearn.Estimator): classifier to score
        X (np.ndarray): feature array to score the estimator on
        y (np.array): ground truth array to score the estimator on

    Returns:
        tuple: scores for each metric for the classifier
    """
    preds = classifier.predict(X)
    return accuracy_score(preds,y), precision_score(preds,y,average='macro'), recall_score(preds,y,average="macro"), f1_score(preds,y,average='macro')

def microscore_classifier(classifier,X,y,name):
    """generates a micro-score array for the given classifier 

    Args:
        classifier (sklearn.Estimator): classifier to score
        X (np.ndarray): feature array to score the estimator on
        y (np.array): ground truth array to score the estimator on

    Returns:
        tuple: scores for each metric for the classifier and classifier name
    """
    preds = classifier.predict(X)
    return precision_score(preds,y,average=None), recall_score(preds,y,average=None),f1_score(preds,y,average=None),[name for k in range(len(precision_score(preds,y,average=None)))]


def plot_confusion_matrix(classifier,X,y,classes_dic,ax):
    """plots the confusion matrix for the given classifier

    Args:
        classifier (skelarn.Estimator): classifier to plot the matrix from
        X (np.ndarray): array to score the classifier on
        y (np.array): ground truth to score the classifier on
        classes_dic (dict): contains label-name correspondences for each class
        ax (plt.axis): axis to plot on
    """
    ConfusionMatrixDisplay.from_predictions(y_true=y,y_pred=classifier.predict(X),ax=ax,cmap='viridis',colorbar=False)
    ax.set_xticklabels(classes_dic)
    ax.set_yticklabels(classes_dic)   
    
