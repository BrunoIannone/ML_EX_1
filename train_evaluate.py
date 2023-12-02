import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn import tree,svm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import seaborn as sns
import pandas as pd
from sklearn.model_selection import GridSearchCV
from plot_utils import plot_confusion_matrix
import numpy as np
from typing import List
from sklearn.base import BaseEstimator
from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn.decomposition import PCA

class TrainEvaluate():
    """
        Class that implements some methods for training and evaluation

    """    
    

    def __init__(self,x_train:np.ndarray,y_train:np.ndarray,x_eval:np.ndarray,y_eval:np.ndarray) -> None:
        """
    Init function for TrainEvaluate class.

    Attributes:
    x_train (numpy.ndarray): Training samples.
    y_train (numpy.ndarray): Training label.
    x_eval (numpy.ndarray): Evaluation samples.
    y_eval (numpy.ndarray): Evaluation label.
    """
        if not isinstance(x_train, np.ndarray):
            raise TypeError("x_train must be a NumPy array.")
        if not isinstance(y_train, np.ndarray):
            raise TypeError("y_train must be a NumPy array.")
        if not isinstance(x_eval, np.ndarray):
            raise TypeError("x_eval must be a NumPy array.")
        if not isinstance(y_eval, np.ndarray):
            raise TypeError("y_eval must be a NumPy array.")
        self.x_train = x_train
        self.y_train = y_train
        self.x_eval = x_eval
        self.y_eval = y_eval
        


    def fit_and_evaluate_model(self,model:BaseEstimator,model_name:str,plot:bool,plot_save:bool,plot_path:str,plot_show:bool,normalization_method:Normalizer=None,labels_name_str:List[str]=None):
        """
        Fit and evaluate a scikit-learn model with optional normalization. This function also implements plotting choices

        Args:
        model (BaseEstimator): The scikit-learn model to be fitted and evaluated.
        normalization_method (Normalizer): The normalization method from scikit-learn to preprocess the input data.
        labels_name_str (List[str]): List of label names for classification report. Default = None
        model_name (str): Name of the model for display purposes.
        plot (bool): Whether to generate a confusion matrix plot.
        plot_save (bool): Whether to save  on a file the confusion matrix plot.
        plot_path (str): The path to save the confusion matrix plot.
        plot_show (bool): Whether to display the confusion matrix plot.

        Returns:
        tuple: model, accuracy N.B. the returned model is the fitted one

        
       """
        # Check if model is a valid scikit-learn model

        

        if normalization_method is not None:
            print("Chosen normalization is: ", normalization_method)
            x_train = normalization_method.fit_transform(self.x_train)
            x_eval = normalization_method.transform(self.x_eval)
            
        else:
            print("No normalization chosen")            

            x_train = self.x_train
            x_eval = self.x_eval
        
        model.fit(x_train, self.y_train)
        
        y_pred = model.predict(x_eval)
        accuracy = accuracy_score(self.y_eval, y_pred)
        print("Accuracy %.5f" %accuracy)
        print(classification_report(self.y_eval, y_pred, labels=None, target_names=labels_name_str, digits=3))
        if plot:
            plot_confusion_matrix(self.y_eval,y_pred,model_name,accuracy,plot_path,plot_show,plot_save,labels_name_str)
        return model,accuracy

    def fit_and_evaluate_with_grid_search(self,grid_search:GridSearchCV,labels_name_str:List[str],model_name:str,plot:bool,plot_save:str,plot_path:str,plot_show:bool,normalization_method:Normalizer=None):
        """
        Fit and evaluate a scikit-learn model using grid search with optional normalization.This function also implements plotting choices

        Args:
        grid_search (GridSearchCV): The grid search object with a scikit-learn model and hyperparameter grid. N.B. This code only works if parameter refit = True
        normalization_method (BaseEstimator): The normalization method from scikit-learn to preprocess the input data.
        labels_name_str (List[str]): List of label names for classification report.
        model_name (str): Name of the model for display purposes.
        plot (bool): Whether to generate a confusion matrix plot.
        plot_save (bool): Whether to save the confusion matrix plot.
        plot_path (str): The path to save the confusion matrix plot.
        plot_show (bool): Whether to display the confusion matrix plot.

        Returns:
        tuple: model, accuracy N.B. the returned model is the best chosen model.

        
        """
        
        if normalization_method is not None:
            print("Chosen normalization is: ",normalization_method)
            x_train = normalization_method.fit_transform(self.x_train)
            x_eval = normalization_method.transform(self.x_eval)
            

        else:
            print("No normalization chosen")
            x_train = self.x_train
            x_eval = self.x_eval

        grid_search.fit(x_train,self.y_train)
        best_estimator = grid_search.best_estimator_
        print("Best model parameters: ", grid_search.best_params_)
        y_pred = best_estimator.predict(x_eval)
        accuracy = accuracy_score(self.y_eval,y_pred)
        print("Accuracy %.5f" %accuracy)
        print(classification_report(self.y_eval, y_pred, digits=3))
        if plot:
            plot_confusion_matrix(y_true=self.y_eval,y_pred=y_pred,model_name=model_name,accuracy=accuracy,plot_save_path=plot_path,plot_show=plot_show,plot_save=plot_save,labels_name=labels_name_str)
        return  best_estimator, accuracy
    

    