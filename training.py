import load_data
import utils
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
import LinearClassifier
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from data_processor import DataProcessor
NORMALIZATION = False

DATA_PATH = os.path.join(utils.PATH,"HW1_datasets_public")
data_preprocessor = DataProcessor(train_data_path=DATA_PATH +"/dataset1.csv",test_data_path=DATA_PATH +"/blind_test1.csv",test_size=0.333, random_state=0)


cnt = data_preprocessor.count_labels()
print(data_preprocessor.labels_name, "\n", cnt)




if NORMALIZATION:
    print("Scaling enabled")
    normalization_method = preprocessing.RobustScaler()#preprocessing.MinMaxScaler()#preprocessing.StandardScaler()

    x_train = normalization_method.fit_transform(x_train)
    x_eval = normalization_method.transform(x_eval)
clf = LogisticRegression(random_state=0,n_jobs=-1,max_iter=100)
#clf.fit(x_train, y_train)
#print("default LOGISTIC EVAL ACCURACY:", accuracy_score(y_eval, clf.predict(x_eval)))

param_grid = {

   
    'C': np.arange(0.1, 2, 0.1).tolist(),
    'max_iter':np.arange(300, 1000, 100).tolist(),#np.arange(0.8, 2, 0.1).tolist(),#uniform(loc=0, scale=4),
    'penalty': ['l2'],
    'solver': ["lbfgs"]
}#param_grid = {'C': [0.1, 1, 10], 'loss': ['hinge', 'squared_hinge']}
# grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1,verbose=3,refit=True)#,random_state=0)
# grid_search.fit(x_train,y_train)
# print(accuracy_score(y_eval, grid_search.best_estimator_.predict(x_eval)))
# print("BEST LOGISTIC: ",grid_search.best_params_,grid_search.best_score_)
# print("Finito")








#model = svm.SVC()#LinearSVC(dual='auto',max_iter=9999)
# #model = utils.MODEL["svm"]

# #model = BernoulliNB()
# #model = LinearClassifier.LeastSquare()
#model.fit(x_train,y_train)
#Print("DEFAULT: ",accuracy_score(y_eval,model.predict(x_eval)))
# #pipe,y_pred = utils.fit_model(model,preprocessing.StandardScaler(),x_train,y_train,x_eval,y_eval,labels_name_str,"Bernoulli")
# #print("DEFAULT: ",accuracy_score(y_eval,pipe.predict(x_eval)))

# param_grid = {
#     'C': np.arange(0.1, 5, 0.1).tolist(),#np.arange(0.8, 2, 0.1).tolist(),#uniform(loc=0, scale=4),
#     'kernel': ["rbf","poly"],

#     'gamma': ['scale']
# }#param_grid = {'C': [0.1, 1, 10], 'loss': ['hinge', 'squared_hinge']}


# print("Defined params")
# print("Starting fitting")

# #pipe = Pipeline(steps=[("scaler", normalization_method), ("svc", model)]) #make_pipeline( model) 
# #grid_search = RandomizedSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1,verbose=3,refit=True,random_state=0)

# #best_estimator = utils.fit_model_with_grid_search(pipe,normalization_method,x_train,y_train,x_eval,y_eval,labels_name_str,"SVM",grid_search)

# #print("MIO STIMATORE: ", accuracy_score(y_eval, best_estimator.predict(x_eval)))
# #model2 = svm.SVC(C = 0.1, kernel='linear')
# #pipe2,y_pred = utils.fit_model(model2,preprocessing.StandardScaler(),x_train,y_train,x_eval,y_eval,labels_name_str,"Bernoulli")
# model2 = svm.SVC()
# print("Finito")
# grid_search = GridSearchCV(model2, param_grid, cv=5, scoring='accuracy', n_jobs=-1,verbose=3,refit=True)#,random_state=0)
# grid_search.fit(x_train,y_train)
# print(grid_search.best_params_,grid_search.best_score_)
# print(accuracy_score(y_eval,grid_search.best_estimator_.predict(x_eval)))