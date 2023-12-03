import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
import os

PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(PATH,"HW1_datasets_public")

MODEL = "LogisticRegression" #choose between "LogisticRegression", "SVM", "AdaBoost", "RandomForest"
NORMALIZATION_METHOD = MinMaxScaler() #choose between None, StandardScaler(),MinMaxScaler()
RANDOM_STATE = 0
N_JOBS = -1
GRID_SEARCH = True

#### LR ####
param_logreg = {

    'C':1,
    'max_iter':100,
    'penalty': 'l2',
    'solver': "lbfgs",
    'random_state' : RANDOM_STATE,
    'n_jobs' : N_JOBS
}

param_grid_logreg = {

    'C':[1],#np.arange(0.1, 2, 0.1).tolist(),
    'max_iter':[100],#np.arange(300, 700, 100).tolist(),
    'penalty': ['l2'],
    'solver': ["lbfgs"]#,"sag","saga"]
}
#############

#### SVM ####
param_svm = {

    'C':1,
    'max_iter':-1,
    'kernel': "linear",
    'random_state' : RANDOM_STATE,
    'gamma': 'scale',
    'verbose': 3
}

param_grid_svm = {

    'C': np.arange(0.1, 2, 0.1).tolist(),
    'max_iter':np.arange(300, 600, 100).tolist(),
    'kernel': ["rbf","linear"],
    'gamma': ["scale","auto"],
    "random_state": [0]
}
#############

#### Random Forest ####

param_random_forest = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'criterion': 'gini',
    'gamma': 'scale',
    'verbose': 3
}

param_grid_random_forest = {
    'n_estimators': np.arange(500, 1100, 100).tolist(),
    'max_depth': np.arange(10, 100, 30).tolist(),
    'min_samples_split': np.arange(10, 100, 30).tolist(),
    'min_samples_leaf': np.arange(10, 100, 30).tolist(),
    'max_features': ['sqrt'],
    'criterion': ['gini', 'entropy']
}
#############

#### CUSTOM MODEL ####
param_custom_model = {
    "n_estimators":100
}

param_grig_custom_model = {
    "n_estimators" : 100
}
CUSTOM_MODEL = RandomForestClassifier(**param_custom_model) #declare here any model that implements sklearn interface
#############