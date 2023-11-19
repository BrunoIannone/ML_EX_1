import load_data
import train_evaluate
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV,RandomizedSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
import matplotlib.pyplot as plt
from sklearn import svm,tree
from sklearn.svm import LinearSVC 
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from data_processor import DataProcessor
NORMALIZATION = False

DATA_PATH = os.path.join(train_evaluate.PATH,"HW1_datasets_public")

print("Processing data")
data_processor = DataProcessor(train_data_path=DATA_PATH +"/dataset1.csv",test_data_path=DATA_PATH +"/blind_test1.csv",test_size=0.333, random_state=0)
print("Processing done")
x_train = data_processor.x_train
y_train = data_processor.y_train

x_eval = data_processor.x_eval
y_eval = data_processor.y_eval

cnt = data_processor.count_labels()
print("Labels are: ", data_processor.labels_name,"\nLabels distribution is:\n", cnt)


print("Creating TrainEvaluate object")
train_eval = train_evaluate.TrainEvaluate(x_train,y_train,x_eval,y_eval)
print("Creating TrainEvaluate object done")
print()
print("Creating and fitting the model\n")
model = LogisticRegression(random_state=0, n_jobs=-1)
normalization_method = None
train_eval.fit_and_evaluate_model(model,normalization_method,data_processor.labels_name,"Logistic Regression",False)
print("Fine programma")

param_grid2 = {

   
    'C': np.arange(0.1, 0.5, 0.1).tolist(),
    'max_iter':np.arange(300, 400, 100).tolist(),#np.arange(0.8, 2, 0.1).tolist(),#uniform(loc=0, scale=4),
    'penalty': ['l2'],
    'solver': ["lbfgs"]
}
grid_search = GridSearchCV(model,param_grid2,scoring='accuracy',n_jobs=-1,verbose=3)
best_logreg= train_eval.fit_and_evaluate_with_grid_search(grid_search,normalization_method,data_processor.labels_name,"Logistic Regression",False)