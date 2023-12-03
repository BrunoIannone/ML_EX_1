from data_processor import DataProcessor
import train_evaluate
import os
from utils import *
from sklearn.model_selection import  GridSearchCV
from termcolor import colored
import csv


print(colored("Starting data processor",'yellow'))
data_processor = DataProcessor(train_data_path=DATA_PATH + "/dataset1.csv",test_data_path=DATA_PATH + "/blind_test1.csv",test_size=0.333, random_state=RANDOM_STATE)
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


print(colored("Fitting ","yellow") + colored(MODEL.__class__.__name__,"red"))
model = MODEL
best_model = None
if GRID_SEARCH:
    print(colored("Starting Grid Search","yellow"))
    grid_search = GridSearchCV(model,PARAM_GRID,scoring='accuracy',n_jobs=N_JOBS,verbose=3)
    best_model= train_eval.fit_and_evaluate_with_grid_search(grid_search=grid_search,normalization_method = NORMALIZATION_METHOD,labels_name_str=data_processor.labels_name,model_name=model.__class__.__name__,plot=True,plot_path=os.path.join(PATH,"plots/"),plot_save=PLOT_SAVE,plot_show=PLOT_SHOW)
    print(colored("Grid Search finished","green"))
else:
    print(colored("Starting Normal fit","yellow"))
    best_model=train_eval.fit_and_evaluate_model(model = model,normalization_method = NORMALIZATION_METHOD,labels_name_str=sorted(data_processor.labels_name),model_name=model.__class__.__name__,plot_cm=True,plot_path=os.path.join(PATH,"plots/"),plot_save=PLOT_SAVE,plot_show=PLOT_SHOW)
    print(colored("Normal fit finished","green"))



csv_file_path = 'predictions2.csv'
predictions = best_model.predict(data_processor.test_samples)
# Open the CSV file in write mode
with open(csv_file_path, 'w', newline='') as csv_file:
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)

    # Write the header if needed (optional)
    # csv_writer.writerow(['Prediction'])

    # Write the predictions to the CSV file
    csv_writer.writerow(predictions)

print(f'Predictions have been written to {csv_file_path}.')





print(colored("Program finished, exiting...", "green"))
