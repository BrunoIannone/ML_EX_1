import plot_utils
from termcolor import colored
from data_processor import DataProcessor
from utils import *

print(colored("Starting data processor",'yellow'))
data_processor = DataProcessor(train_data_path=DATA_PATH + "/dataset1.csv",test_data_path=DATA_PATH + "/blind_test2.csv",test_size=0.333, random_state=0)
print()

print(colored("Apply PCA for 3D plot with GPU","light_cyan"))
plot_utils.plot_samples_3d_gpu(data_processor.train_samples, data_processor.train_labels,StandardScaler(),plot_save_path='interactive_plot.html',plot_show=  True, plot_save=  False)

print(colored("Program finished, exiting...", "green"))
