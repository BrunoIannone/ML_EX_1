import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn import tree,svm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import seaborn as sns
import pandas as pd
from sklearn.model_selection import GridSearchCV



#model = MultinomialNB().fit(X_train, y_train)
#
PATH = os.path.dirname(__file__)
MODEL = {"svm": svm.SVC(kernel='linear', C=1),"tree": tree.DecisionTreeClassifier()}

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

def fit_model(model,normalization_method,x_train,y_train,x_eval,y_eval,labels_name_str,model_name):
    pipe = make_pipeline(normalization_method, model) #make_pipeline( model) 
    pipe.fit(x_train, y_train)
    # if isinstance(model, tree.DecisionTreeClassifier):
    #     tree.plot_tree(model,fontsize=12)
    #     plt.rcParams["figure.figsize"] = (10,20)
    #     plt.show()

    y_pred = pipe.predict(x_eval)
    acc = pipe.score(x_eval, y_eval)
    print("Accuracy %.5f" %acc)
    print(classification_report(y_eval, y_pred, labels=None, target_names=labels_name_str, digits=3))
    plot_confusion_matrix(y_eval,y_pred,labels_name_str,model_name)
    return pipe, y_pred

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


def fit_model_with_grid_search(model,normalization_method,x_train,y_train,x_eval,y_eval,labels_name_str,model_name,grid_search):
    
    grid_search.fit(x_train,y_train)
    best_estmiator = grid_search.best_estimator_
    y_pred = best_estmiator.predict(x_eval)
    print(accuracy_score(y_eval,y_pred))
    print(classification_report(y_eval, y_pred, labels=None, target_names=labels_name_str, digits=3))
    plot_confusion_matrix(y_eval,y_pred,labels_name_str,model_name)
    plot_grid_search(grid_search)