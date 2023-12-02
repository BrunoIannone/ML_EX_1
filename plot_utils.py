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
import plotly.express as px
import plotly.graph_objects as go




def plot_confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray, model_name:str,accuracy:float,plot_save_path:str,plot_show:bool,plot_save:bool,labels_name:List[str]=None):
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
    plt.clf()
    plt.close()

def plot_3d_decision_boundaries_gpu(X, y, model, mesh_size=0.02):

    coefficients = model.coef_
    intercepts = model.intercept_

    # Crea una griglia 3D
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Crea la figura 3D
    fig = go.Figure()

    # Aggiungi i punti del dataset con colori diversi per ogni classe
    for class_idx in range(len(model.classes_)):
        # Calcola i valori previsti per ogni combinazione di x e y per la classe corrente
        z_grid = (coefficients[class_idx, 0] * x_grid +
                coefficients[class_idx, 1] * y_grid +
                intercepts[class_idx])

        # Aggiungi una superficie 3D per rappresentare il piano di decisione della classe corrente
        fig.add_trace(go.Surface(x=x_grid, y=y_grid, z=z_grid, opacity=0.2, colorscale="Viridis", name=f'Class {class_idx}'))

        # Aggiungi i punti del dataset per la classe corrente
        class_points = X[y == class_idx]
        fig.add_trace(go.Scatter3d(x=class_points[:, 0], y=class_points[:, 1], z=class_points[:, 2],
                                mode='markers', marker=dict(size=4), name=f'Class {class_idx}'))

    # Aggiungi etichette degli assi
    fig.update_layout(scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'))

    # Mostra il grafico
    fig.show()

def plot_3d_decision_boundaries_gpu2(X, y, model, mesh_size=0.02):
    # Estrai i coefficienti del modello
    coefficients = model.coef_
    intercepts = model.intercept_

    # Calcola i punti di intersezione tra i piani di decisione delle diverse classi
    intersection_points = []
    for i in range(len(model.classes_)):
        for j in range(i + 1, len(model.classes_)):
            coefficients_diff = coefficients[i] - coefficients[j]
            intercepts_diff = intercepts[i] - intercepts[j]
            intersection_x = -intercepts_diff / coefficients_diff[0]
            intersection_y = -intercepts_diff / coefficients_diff[1]
            intersection_z = -intercepts_diff / coefficients_diff[2]
            intersection_points.append([intersection_x, intersection_y, intersection_z])

    intersection_points = np.array(intersection_points)

    # Crea la figura 3D
    fig = go.Figure()

    # Aggiungi la retta di intersezione
    fig.add_trace(go.Scatter3d(x=intersection_points[:, 0], y=intersection_points[:, 1], z=intersection_points[:, 2],
                            mode='lines', line=dict(width=5, color='red'), name='Intersection Line'))

    # Aggiungi i punti del dataset con colori diversi per ogni classe
    for class_idx in range(len(model.classes_)):
        class_points = X[y == class_idx]
        fig.add_trace(go.Scatter3d(x=class_points[:, 0], y=class_points[:, 1], z=class_points[:, 2],
                                mode='markers', marker=dict(size=4), name=f'Class {class_idx}'))

    # Aggiungi etichette degli assi
    fig.update_layout(scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'))

    # Mostra il grafico
    fig.show()

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
        legend_handles = scatter.legend_elements()[0]
        plt.legend(handles=legend_handles, labels=legend_labels)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.show()







def plot_samples_3d_gpu(X, y,normalization_method, title, marker_size=5,show_plot = True, save_plot = False):
    
    
    pca = PCA(n_components=3)
    X = pca.fit_transform(normalization_method.fit_transform(X))#x_train 

    y = np.reshape(y,(y.shape[0],1))
    combined_data = np.concatenate([X, y], axis=1)

    # Create a pandas DataFrame
    column_names = ['feature_1', 'feature_2', 'feature_3', 'label']
    df = pd.DataFrame(combined_data, columns=column_names)
    df["label"] = df["label"].astype(str)
    fig = px.scatter_3d(df, x='feature_1', y='feature_2', z='feature_3',
              color='label')
    fig.update_traces(marker=dict(size=3,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    
    fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 30),itemsizing = 'constant'))

    if show_plot:
        fig.show()
    if save_plot:
        fig.write_html('interactive_plot.html')

def plot_points_with_pca2(samples,labels,legend_labels = None):
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(samples)

    # Plot the data
    fig = plt.figure(figsize=(300, 300))

    ##
    ax = fig.add_subplot(111,projection = '3d')
    ##
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],pca_result[:,2], cmap='Paired', c=labels, edgecolors='k')

    # Add legend if legend_labels are provided
    if legend_labels:
        legend_handles = scatter.legend_elements()[0]
        plt.legend(handles=legend_handles, labels=legend_labels)
    plt.title('PCA Visualization of 100-Dimensional Data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
