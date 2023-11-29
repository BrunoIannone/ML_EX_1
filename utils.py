from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import warnings
import numpy as np
import datetime
from typing import List
from sklearn.decomposition import PCA
from sklearn import preprocessing

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


def plot_decision_boundary(X, y, clf, title, legend_labels=None):
    h = .02  # Step size in the meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the class for each point in the meshgrid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundaries
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    
    # Add legend if legend_labels are provided
    if legend_labels:
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.show()


def plot_points_with_pca(samples,labels):
    
    pca = PCA(n_components=2,random_state=0)
    data_2d = pca.fit_transform(preprocessing.StandardScaler().fit_transform(samples))

    # Colora i punti in base alle classi
    classes = set(labels)
    scatter = []
    for class_label in classes:
        indices = (labels == class_label)
        scatter.append(plt.scatter(data_2d[indices, 0], data_2d[indices, 1], label=f'Classe {class_label}',cmap=plt.cm.Paired,edgecolors='k'))

    # Plotta la legenda
    plt.legend(handles=scatter, title='Classi', loc='best')

    # Resto del codice per il plot
    plt.title('PCA: Riduzione della dimensionalità a 2D')
    plt.xlabel('Prima componente principale')
    plt.ylabel('Seconda componente principale')
    plt.show()
from sklearn.manifold import TSNE

def plot_points_with_pca2(samples,labels,legend_labels = None):
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(samples)

    # Plot the data
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], cmap='Paired', c=labels, edgecolors='k')

    # Add legend if legend_labels are provided
    if legend_labels:
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)

    plt.title('PCA Visualization of 100-Dimensional Data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
