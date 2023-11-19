from load_data import load_data
from sklearn.model_selection import train_test_split


class DataProcessor():
    
    def __init__(self,train_data_path, test_data_path, test_size,random_state) -> None:
        self.train_data = load_data(train_data_path) # samples =  (50000, 100) labels = (50000,)
        assert self.train_data is not None, "train_data is None"

        self.test_data = load_data(test_data_path) # samples =  (50000, 100) labels = (50000,)
        assert self.test_data is not None, "test_data is None"
        
        self.x_train,self.y_train,self.x_eval,self.y_eval, self.train_samples,self.train_labels,self.test_samples= self.data_load_and_process_data(test_size,random_state)
        
        self.labels_name = self.extract_labels()
    
    def count_labels(self):
        cnt = {}
        for label in self.train_labels:
            if label not in cnt:
                cnt[label] = 1
            else:
                cnt[label] = cnt[label] + 1
        
        return cnt
    
    def extract_labels(self):
        labels_set = set()
    
        for label in self.train_labels:
            labels_set.add(str(label))
            
        return labels_set
    

    def data_load_and_process_data(self,test_size, random_state):
        

        train_samples = self.train_data[0]
        train_labels  = self.train_data[1]

        assert train_labels is not None, "train_labels is None, wrong file?"
        assert train_samples.shape[0] == train_labels.shape[0], "train_samples and train_labels dimension mismatch"

        test_samples = self.test_data[0]

        x_train, x_eval, y_train, y_eval = train_test_split(train_samples, train_labels, test_size=test_size,
                                                        random_state=random_state)

        return x_train,y_train,x_eval,y_eval, train_samples,train_labels,test_samples