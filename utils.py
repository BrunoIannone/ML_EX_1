from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import warnings
import datetime
PATH = os.path.dirname(__file__)

def plot_confusion_matrix(y_true, y_pred, labels_name, model_name,accuracy,plot_save_path,plot_show,plot_save):
    
    if plot_show == None and plot_save == None:
        warnings.warn("Called 'plot_confusion_matrix' but plot_show and plot_save are None",RuntimeWarning)
        return
    elif plot_save and plot_save_path is None:
        raise ValueError("If plot_save is set to True, plot_save_path must be specified.")
    elif plot_save is None and plot_save_path:
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

