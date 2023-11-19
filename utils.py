from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def plot_confusion_matrix(y_true,y_pred,labels_name, model_name):
    cm = confusion_matrix(y_true, y_pred,normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels_name)
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set(
    title='Confusion Matrix ' + model_name, 
    xlabel='Predicted', 
    ylabel='Actual ')
    plt.xticks(rotation = 45)
    plt.grid(False)
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

