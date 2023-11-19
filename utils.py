from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import warnings
import numpy as np
import datetime
from typing import List
PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(PATH,"HW1_datasets_public")


def plot_confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray, labels_name:List[str], model_name:str,accuracy:float,plot_save_path:str,plot_show:bool,plot_save:bool):
    """
    Plot the confusion matrix for a classification model.

    Args:
    y_true(np.ndarray): Ground truth labels.
    y_pred(np.ndarray): Predicted labels.
    labels_name (List[str]): List of label names.
    model_name (str): Name of the model for display purposes.
    accuracy (float): Accuracy score of the model.
    plot_save_path (str): Path to save the confusion matrix plot.
    plot_show (bool): Whether to display the confusion matrix plot.
    plot_save (bool): Whether to save the confusion matrix plot.

    Raises:
    ValueError: If plot_save is set to True but plot_save_path is not specified.
    ValueError: If y_pred or y_true are not NumPy array.

    Example:
    >>> labels = ["class1", "class2", "class3"]
    >>> y_true = ["class1", "class2", "class3", "class1", "class2", "class3"]
    >>> y_pred = ["class1", "class2", "class3", "class1", "class2", "class2"]
    >>> model_name = "MyClassifier"
    >>> accuracy = 0.75
    >>> plot_confusion_matrix(y_true, y_pred, labels, model_name, accuracy, "path/to/save", True, True)
    """
    if not isinstance(y_pred, np.ndarray):
            raise TypeError("y_pred must be a NumPy array.")
    if not isinstance(y_true, np.ndarray):
            raise TypeError("y_true must be a NumPy array.")
    
    if plot_show == None and plot_save == None:
        warnings.warn("Called 'plot_confusion_matrix' but plot_show and plot_save are None",RuntimeWarning)
        return
    if plot_save and plot_save_path is None:
        raise ValueError("If plot_save is set to True, plot_save_path must be specified.")
    if plot_save is None and plot_save_path:
        warnings.warn("If plot_save is set to False, plot_save_path should not be specified.",RuntimeWarning)
    
    
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
    
    # Set the figure size in inches for 720p resolution
    fig, ax = plt.subplots(figsize=(1280/100, 720/100), dpi=100)
    
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set(
        title='Confusion Matrix ' + model_name,
        xlabel='Predicted',
        ylabel='Actual '
    )
    plt.xticks(rotation=45)
    plt.grid(False)
    
    # Save the figure in the current working directory with 720p resolution
    if plot_save:
        plt.savefig(os.path.join(plot_save_path, "confusion_matrix" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")) + " $ " +str(str(accuracy).split(".")[1])), dpi=100)
    if plot_show:
        plt.show()

def plot_grid_search(grid_search):
    results = grid_search.cv_results_
    params = results['params']
    mean_test_scores = results['mean_test_score']

    # Create a DataFrame for easy plotting


    df = pd.DataFrame(params)
    df['mean_test_score'] = mean_test_scores

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Plot for linear kernel
    sns.lineplot(x='svc__C', y='mean_test_score', data=df[df['svc__kernel'] == 'linear'], marker='o', label='Linear Kernel')

    ## Plot for linear kernel
    #sns.lineplot(x='C', y='mean_test_score', hue='gamma', data=df[df['kernel'] == 'linear'], marker='o', label='Linear Kernel')

    ## Plot for rbf kernel
    #sns.lineplot(x='C', y='mean_test_score', hue='gamma', data=df[df['kernel'] == 'rbf'], marker='o', label='RBF Kernel')

    plt.title('Grid Search Results for SVM')
    plt.xlabel('C')
    plt.ylabel('Mean Test Score (Accuracy)')
    plt.xscale('log')  # Use a logarithmic scale for better visualization
    plt.legend(title='Kernel')  # Add a title for the legend
    plt.show()

