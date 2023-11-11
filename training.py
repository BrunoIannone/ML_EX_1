import load_data
import utils
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
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


DATA_PATH = os.path.join(utils.PATH,"HW1_datasets_public")

labels_name_str = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
labels_name_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

train_data = load_data.load_data(DATA_PATH +"/dataset1.csv") # samples =  (50000, 100) labels = (50000,)
assert train_data is not None, "train_data is None"

test_data = load_data.load_data(DATA_PATH +"/blind_test1.csv") # samples =  (50000, 100) labels = (50000,)
assert test_data is not None, "test_data is None"

train_samples= train_data[0]
train_labels = train_data[1]
cnt = {}
for label in train_labels:
    if label not in cnt:
        cnt[label] = 1
    else:
        cnt[label] = cnt[label] + 1
print(cnt)
assert train_labels is not None, "train_labels is None, wrong file?"
assert train_samples.shape[0] == train_labels.shape[0], "train_samples and train_labels dimension mismatch"

test_samples = test_data[0]

x_train, x_eval, y_train, y_eval = train_test_split(train_samples, train_labels, test_size=0.333,
                                                    random_state=1127)



model = svm.SVC()#LinearSVC(dual='auto',max_iter=9999)
#model = utils.MODEL["svm"]

#model = BernoulliNB()
#model = LinearClassifier.LeastSquare()

#pipe,y_pred = utils.fit_model(model,preprocessing.StandardScaler(),x_train,y_train,x_eval,y_eval,labels_name_str,"Bernoulli")

param_grid = {
    'svc__C': np.arange(0.1, 2, 0.1).tolist(),
    'svc__kernel': ['linear'],
    #'gamma': np.arange(0.1, 0.3, 0.1).tolist()
}#param_grid = {'C': [0.1, 1, 10], 'loss': ['hinge', 'squared_hinge']}

print("Defined params")
print("Starting fitting")

normalization_method = preprocessing.StandardScaler()
pipe = Pipeline(steps=[("scaler", normalization_method), ("svc", model)]) #make_pipeline( model) 
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1,verbose=3)

utils.fit_model_with_grid_search(model,normalization_method,x_train,y_train,x_eval,y_eval,labels_name_str,"SVM",grid_search)
print("Finito")
