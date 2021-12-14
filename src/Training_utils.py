
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(


    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=5,
    n_jobs=None
):
    """generates a learning curve plot for a given estimator on given data

    Args:
        estimator (sklearn.Estimator): estimator to plot the learning curve from
        title (str): title of the plot
        X (np.ndarray): features to score the estimator on
        y (np.ndarray): ground truth to score the classifier on
        axes ( optional):  Defaults to None.
        ylim (float, optional): Defaults to None.
        cv (int, optional): K-fold cross-validation split amount of folds. Defaults to 5.

    Returns:
        plt.plot: plot of estimator learning curve
    """
    if axes is None:
        axes = plt.subplots(figsize=(20, 5))

    axes.set_title(title)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores,= learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.1,1.0,10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="b",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="k",
    )
    axes.plot(
        train_sizes, train_scores_mean, "*-", color="b", label="Training score"
    )
    axes.plot(
        train_sizes, test_scores_mean, "*-", color="k", label="Cross-validation score"
    )
    axes.legend(loc="best")

    return plt
