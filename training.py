import load_data
import utils
import os
DATA_PATH = os.path.join(utils.PATH,"HW1_datasets_public")
print(DATA_PATH)
data = load_data.load_data(DATA_PATH +"/dataset1.csv") # samples =  (50000, 100) labels = (50000,)

samples = data[0]
labels = data[1]
print(samples.shape, labels.shape)