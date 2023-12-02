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

print(colored("Creating TrainEvaluate object",'yellow'))
train_eval = train_evaluate.TrainEvaluate(x_train,y_train,x_eval,y_eval)
print(colored("TrainEvaluate object created","green"))
print()

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

print(colored("Program finished, exiting...", "green"))

