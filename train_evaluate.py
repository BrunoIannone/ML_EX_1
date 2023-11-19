import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn import tree,svm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import seaborn as sns
import pandas as pd
from sklearn.model_selection import GridSearchCV
from utils import plot_confusion_matrix


#model = MultinomialNB().fit(X_train, y_train)
#
MODEL = {"svm": svm.SVC(kernel='linear', C=1),"tree": tree.DecisionTreeClassifier()}
class TrainEvaluate():

    def __init__(self,x_train,y_train,x_eval,y_eval) -> None:

        self.x_train = x_train
        self.y_train = y_train
        self.x_eval = x_eval
        self.y_eval = y_eval
        


    def fit_and_evaluate_model(self,model,normalization_method,labels_name_str,model_name,plot,plot_save,plot_path,plot_show):
        
        if normalization_method is not None:
            print("Chosen normalization is: ",normalization_method )
            x_train = normalization_method.fit_transform(self.x_train)
            x_eval = normalization_method.transform(self.x_eval)

        else:
            x_train = self.x_train
            x_eval = self.x_eval
            print("No normalization chosen")            
        model.fit(x_train, self.y_train)
        
        y_pred = model.predict(x_eval)
        acc = model.score(x_eval, self.y_eval)
        print("Accuracy %.5f" %acc)
        print(classification_report(self.y_eval, y_pred, labels=None, target_names=labels_name_str, digits=3))
        if plot:
            plot_confusion_matrix(self.y_eval,y_pred,labels_name_str,model_name,acc,plot_path,plot_show,plot_save)
        return model,acc

    def fit_and_evaluate_with_grid_search(self,grid_search,normalization_method,labels_name_str,model_name,plot,plot_save,plot_path,plot_show):
        
        if normalization_method is not None:
            print("Chosen normalization is: ",normalization_method )
            x_train = normalization_method.fit_transform(self.x_train)
            x_eval = normalization_method.transform(self.x_eval)

        else:
            x_train = self.x_train
            x_eval = self.x_eval
            print("No normalization chosen")
        grid_search.fit(x_train,self.y_train)
        best_estimator = grid_search.best_estimator_
        print(grid_search.best_params_)
        y_pred = best_estimator.predict(x_eval)
        acc = accuracy_score(self.y_eval,y_pred)
        print(acc)
        print(classification_report(self.y_eval, y_pred, labels=None, digits=3))
        if plot:
            plot_confusion_matrix(self.y_eval,y_pred,labels_name_str,model_name,acc,plot_path,plot_show,plot_save)
        #plot_grid_search(grid_search)
        return  best_estimator, acc
    