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
import utils
from sklearn.decomposition import PCA

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier



print("Processing data")
data_processor = DataProcessor(train_data_path=utils.DATA_PATH + "/dataset1.csv",test_data_path=utils.DATA_PATH + "/blind_test2.csv",test_size=0.333, random_state=0)
print("Processing done")
x_train = data_processor.x_train
y_train = data_processor.y_train

x_eval = data_processor.x_eval
y_eval = data_processor.y_eval

cnt = data_processor.count_labels()
print("Labels are: ", sorted(data_processor.labels_name),"\nLabels distribution is:\n", cnt)


# print("Creating TrainEvaluate object")
train_eval = train_evaluate.TrainEvaluate(x_train,y_train,x_eval,y_eval)
print("Creating TrainEvaluate object done")
print()
print("Creating and fitting the model")
#model = LogisticRegression(random_state=0, n_jobs=-1)
#model = svm.SVC(random_state=0)
#model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(max_depth=64),n_estimators=100,random_state=0)
#model = RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1, bootstrap=True,max_depth=20,max_leaf_nodes=20)
#normalization_method = preprocessing.MinMaxScaler()
#log_reg_model,_=train_eval.fit_and_evaluate_model(model = model,normalization_method=normalization_method,labels_name_str=data_processor.labels_name,model_name="Logistic Regression",plot=True,plot_path=os.path.join(utils.PATH,"plots/"),plot_save=True,plot_show=False)
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

# clf = AdaBoostClassifier(estimator=LogisticRegression(C=0.1, max_iter=300,solver='saga',verbose=3,n_jobs=-1),n_estimators=100,random_state=0)
# x_train = normalization_method.fit_transform(x_train)
# x_eval = normalization_method.transform(x_eval)
# clf.fit(x_train,y_train)
# print(accuracy_score(y_eval,clf.predict(x_eval)))



scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(train_eval.x_train)
X_test_scaled = scaler.transform(train_eval.x_eval)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Fit logistic regression on the reduced-dimensional data
logreg = LogisticRegression()
logreg.fit(X_train_pca, y_train)

# Plot decision boundaries with 'jet' colormap
h = .02  # step size in the mesh

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a custom 'jet' colormap
cmap_jet =plt.cm.get_cmap('Paired', 10)

# Scatter plot of the data points
plt.figure(figsize=(8, 6))
# Plot decision boundaries first
plt.contourf(xx, yy, Z, alpha=0.8,cmap = cmap_jet)
# Scatter plot of the data points with legend
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train,edgecolors='k', cmap=cmap_jet)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision Boundaries with PCA and Logistic Regression')

# Create a legend
legend_labels = ['Label ' + str(i) for i in range(10)]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Labels')

plt.show()

