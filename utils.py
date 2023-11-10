import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn import tree,svm


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
    print("Accuracy %.3f" %acc)
    print(classification_report(y_eval, y_pred, labels=None, target_names=labels_name_str, digits=3))
    plot_confusion_matrix(y_eval,y_pred,labels_name_str,model_name)
    return pipe, y_pred