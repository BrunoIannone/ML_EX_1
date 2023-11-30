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
from sklearn.linear_model import LogisticRegression,Perceptron
from data_processor import DataProcessor
import utils
from sklearn.decomposition import PCA

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB,BernoulliNB


print("Processing data")
data_processor = DataProcessor(train_data_path=utils.DATA_PATH + "/dataset2.csv",test_data_path=utils.DATA_PATH + "/blind_test2.csv",test_size=0.333, random_state=0)
print("Processing done")
x_train = data_processor.x_train
y_train = data_processor.y_train

x_eval = data_processor.x_eval
y_eval = data_processor.y_eval

#cnt = data_processor.count_labels()
#print("Labels are: ", sorted(data_processor.labels_name),"\nLabels distribution is:\n", cnt)

normalization_method = preprocessing.StandardScaler()

#utils.plot_points_with_pca2(normalization_method.fit_transform(data_processor.train_samples),data_processor.train_labels,sorted(data_processor.labels_name))


print("Creating TrainEvaluate object")
train_eval = train_evaluate.TrainEvaluate(x_train,y_train,x_eval,y_eval)
print("Creating TrainEvaluate object done")
print()
print("Creating and fitting the model")


## MODEL SECTION

#model = LogisticRegression(random_state=0, n_jobs=-1)
#model = svm.SVC(random_state=0)
#model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(max_depth=64),n_estimators=100,random_state=0)
#model = RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1, bootstrap=True,max_depth=20,max_leaf_nodes=20)
#normalization_method = preprocessing.MinMaxScaler()
model = LogisticRegression(random_state=0,max_iter=999999)

#log_reg_model,_=train_eval.fit_and_evaluate_model(model = model,normalization_method=normalization_method,labels_name_str=data_processor.labels_name,model_name=str(model),plot=True,plot_path=os.path.join(utils.PATH,"plots/"),plot_save=True,plot_show=True)

#  GRID


# param_grid_logreg = {

   
#     'C':np.arange(0.1, 2, 0.1).tolist(),
#     'max_iter':np.arange(300, 600, 100).tolist(),#np.arange(0.8, 2, 0.1).tolist(),#uniform(loc=0, scale=4),
#     'penalty': ['l2'],
#     'solver': ["sag","saga","lbfgs"]
# }
# param_grid_svm = {

   
#     'C': np.arange(0.1, 2, 0.1).tolist(),
#     'max_iter':[-1],#np.arange(300, 600, 100).tolist(),#np.arange(0.8, 2, 0.1).tolist(),#uniform(loc=0, scale=4),
#     'kernel': ["linear", "rbf"] ,
#     'gamma': ["scale","auto"],
    
#     "random_state": [0]
# }
#clf = AdaBoostClassifier(estimator=LogisticRegression(C=0.1, max_iter=300,solver='saga',verbose=False,n_jobs=-1),random_state=0)
# param_grid_random_forest = {
#     'n_estimators': np.arange(500, 1100, 100).tolist(),
#     'max_depth': np.arange(10, 100, 30).tolist(),
#     'min_samples_split': np.arange(10, 100, 30).tolist(),
#     'min_samples_leaf': np.arange(10, 100, 30).tolist(),
#     'max_features': ['sqrt'],
#     'criterion': ['gini', 'entropy']
# }
# param_grid_random_forest = {
#     'n_estimators': [500, 800, 1100],
#     'max_depth': [10, 40, 70],
#     'min_samples_split': [10, 40, 70],
#     'min_samples_leaf': [10, 40, 70],
#     'max_features': ['sqrt'],
#     'criterion': ['entropy']  # or 'gini'
# }
#grid_search = GridSearchCV(model,param_grid_random_forest,scoring='accuracy',n_jobs=-1,verbose=3)#,random_state=0,n_iter=200)
#best_logreg= train_eval.fit_and_evaluate_with_grid_search(grid_search=grid_search,normalization_method=normalization_method,labels_name_str=data_processor.labels_name,model_name="SVM",plot=True,plot_path=os.path.join(utils.PATH,"plots/"),plot_save=True,plot_show=False)



#utils.plot_points_with_pca(x_train,y_train)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_eval)

print("Apply PCA for dimensionality reduction to 2 components")
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# # Fit a classifier (Support Vector Machine in this example)
# clf = Perceptron(penalty='l2',n_jobs=-1,random_state=0)#LogisticRegression(max_iter = 500,n_jobs=-1,verbose=3)#svm.SVC(kernel='linear',verbose=3,max_iter=100)
model.fit(X_train_pca, y_train)


# # Plot decision boundaries on the training set
# utils.plot_decision_boundary(X_train_pca, y_train, clf, 'Decision Boundaries on Training Set')

print("Plot decision boundaries on the testing set")

#utils.plot_samples_3d_gpu(X_test_pca, y_eval,'Decision Boundaries on Testing Set')#,sorted(data_processor.labels_name))
utils.plot_3d_decision_boundaries_gpu(X_test_pca,y_eval, model)#,'Decision Boundaries on Testing Set')#,sorted(data_processor.labels_name))
