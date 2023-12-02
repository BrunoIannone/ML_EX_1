from data_processor import DataProcessor
import train_evaluate
import plot_utils
import os
from utils import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import svm,tree
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.model_selection import  GridSearchCV,RandomizedSearchCV
from termcolor import colored

print(colored("Starting data processor",'yellow'))
data_processor = DataProcessor(train_data_path=DATA_PATH + "/dataset1.csv",test_data_path=DATA_PATH + "/blind_test2.csv",test_size=0.333, random_state=0)
print(colored("Finished data processing",'green'))
print()

x_train = data_processor.x_train
y_train = data_processor.y_train

x_eval = data_processor.x_eval
y_eval = data_processor.y_eval

cnt = data_processor.count_labels(data_processor.train_labels)
print(colored("Labels are: ","cyan"), sorted(data_processor.labels_name),colored("\nLabels distribution is:\n","cyan"), cnt)
print()



#utils.plot_points_with_pca2(normalization_method.fit_transform(data_processor.train_samples),data_processor.train_labels,sorted(data_processor.labels_name))


print(colored("Creating TrainEvaluate object",'yellow'))
train_eval = train_evaluate.TrainEvaluate(x_train,y_train,x_eval,y_eval)
print(colored("TrainEvaluate object created","green"))

if MODEL == "LogisticRegression":
    print(colored("Fitting ","yellow") + colored(MODEL,"red"))
    model = LogisticRegression(**param_logreg)
    if GRID_SEARCH:
        print(colored("Starting Grid Search","yellow"))
        grid_search = GridSearchCV(model,param_grid_logreg,scoring='accuracy',n_jobs=-1,verbose=3)#,random_state=0,n_iter=200)
        best_logreg= train_eval.fit_and_evaluate_with_grid_search(grid_search=grid_search,normalization_method = NORMALIZATION_METHOD,labels_name_str=data_processor.labels_name,model_name=model.__class__.__name__,plot=True,plot_path=os.path.join(PATH,"plots/"),plot_save=True,plot_show=False)
        print(colored("Grid Search finished","green"))
    else:
        print(colored("Starting Normal fit","yellow"))
        log_reg_model,_=train_eval.fit_and_evaluate_model(model = model,normalization_method = NORMALIZATION_METHOD,labels_name_str=sorted(data_processor.labels_name),model_name=model.__class__.__name__,plot=True,plot_path=os.path.join(PATH,"plots/"),plot_save=True,plot_show=True)
        print(colored("Normal fit finished","green"))

elif MODEL == "SVM":
    print(colored("Fitting ","yellow") + colored(MODEL,"red"))
    model = svm.SVC(**param_svm)
    if GRID_SEARCH:
        print(colored("Starting Grid Search","yellow"))
        grid_search = GridSearchCV(model,param_grid_svm,scoring='accuracy',n_jobs=-1,verbose=3)#,random_state=0,n_iter=200)
        best_logreg= train_eval.fit_and_evaluate_with_grid_search(grid_search=grid_search,normalization_method = NORMALIZATION_METHOD,labels_name_str=data_processor.labels_name,model_name=model.__class__.__name__,plot=True,plot_path=os.path.join(PATH,"plots/"),plot_save=True,plot_show=False)
        print(colored("Grid Search finished","green"))
    else:
        print(colored("Starting Normal fit","yellow"))
        log_reg_model,_=train_eval.fit_and_evaluate_model(model = model,normalization_method = NORMALIZATION_METHOD,labels_name_str=sorted(data_processor.labels_name),model_name=model.__class__.__name__,plot=True,plot_path=os.path.join(PATH,"plots/"),plot_save=True,plot_show=True)
        print(colored("Normal fit finished","green"))

elif MODEL == "RandomForest":
    
    print(colored("Fitting ","yellow") + colored(MODEL,"red"))
    model = RandomForestClassifier(**param_random_forest)

    if GRID_SEARCH:
        print(colored("Starting Grid Search","yellow"))
        grid_search = GridSearchCV(model,param_grid_random_forest,scoring='accuracy',n_jobs=-1,verbose=3)#,random_state=0,n_iter=200)
        best_logreg= train_eval.fit_and_evaluate_with_grid_search(grid_search=grid_search,normalization_method = NORMALIZATION_METHOD,labels_name_str=data_processor.labels_name,model_name=model.__class__.__name__,plot=True,plot_path=os.path.join(PATH,"plots/"),plot_save=True,plot_show=False)
        print(colored("Grid Search finished","green"))
    else:
        print(colored("Starting Normal fit","yellow"))
        log_reg_model,_=train_eval.fit_and_evaluate_model(model = model,normalization_method = NORMALIZATION_METHOD,labels_name_str=sorted(data_processor.labels_name),model_name=model.__class__.__name__,plot=True,plot_path=os.path.join(PATH,"plots/"),plot_save=True,plot_show=True)
        print(colored("Normal fit finished","green"))


else:
    print(colored("Fitting ","yellow") + colored(CUSTOM_MODEL.__class__.__name__,"red"))
    model = CUSTOM_MODEL
    if GRID_SEARCH:
        print(colored("Starting Grid Search","yellow"))
        grid_search = GridSearchCV(model,param_grig_custom_model,scoring='accuracy',n_jobs=-1,verbose=3)#,random_state=0,n_iter=200)
        best_logreg= train_eval.fit_and_evaluate_with_grid_search(grid_search=grid_search,normalization_method = NORMALIZATION_METHOD,labels_name_str=data_processor.labels_name,model_name=model.__class__.__name__,plot=True,plot_path=os.path.join(PATH,"plots/"),plot_save=True,plot_show=False)
        print(colored("Grid Search finished","green"))
    else:
        print(colored("Starting Normal fit","yellow"))
        log_reg_model,_=train_eval.fit_and_evaluate_model(model = model,normalization_method = NORMALIZATION_METHOD,labels_name_str=sorted(data_processor.labels_name),model_name=model.__class__.__name__,plot=True,plot_path=os.path.join(PATH,"plots/"),plot_save=True,plot_show=True)
        print(colored("Normal fit finished","green"))


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
##grid_search = GridSearchCV(model,param_,scoring='accuracy',n_jobs=-1,verbose=3)#,random_state=0,n_iter=200)
#best_logreg= train_eval.fit_and_evaluate_with_grid_search(grid_search=grid_search,normalization_method=normalization_method,labels_name_str=data_processor.labels_name,model_name=model.__class__.__name__,plot=True,plot_path=os.path.join(utils.PATH,"plots/"),plot_save=True,plot_show=False)



#utils.plot_points_with_pca(x_train,y_train)
#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(x_eval)#x_train 
#X_test = scaler.transform(x_eval)

#print("Apply PCA for dimensionality reduction to 2 components")
#pca = PCA(n_components=3)
#X_train_pca = pca.fit_transform(X_train)
#X_test_pca = pca.transform(X_test)

# # Fit a classifier (Support Vector Machine in this example)
# clf = Perceptron(penalty='l2',n_jobs=-1,random_state=0)#LogisticRegression(max_iter = 500,n_jobs=-1,verbose=3)#svm.SVC(kernel='linear',verbose=3,max_iter=100)
#model.fit(X_train_pca, y_train)


# # Plot decision boundaries on the training set
# utils.plot_decision_boundary(X_train_pca, y_train, clf, 'Decision Boundaries on Training Set')

#print("Plot decision boundaries on the testing set")

#utils.plot_samples_3d_gpu(X_train_pca, data_processor.train_labels,'Decision Boundaries on Testing Set',show_plot = True, save_plot = False)#,sorted(data_processor.labels_name))
#utils.plot_3d_decision_boundaries_gpu(X_test_pca,y_eval, model)#,'Decision Boundaries on Testing Set')#,sorted(data_processor.labels_name))
#utils.plot_points_with_pca2(X_test,y_eval,data_processor.labels_name)