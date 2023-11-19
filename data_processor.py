from load_data import load_data
from sklearn.model_selection import train_test_split


class DataProcessor():
    
    def __init__(self,train_data_path:str, test_data_path:str,test_size:float,random_state:int) -> None:
        """Data Processor init function. This class implements some methods for processing data


        Args:
            train_data_path (str): Path to train data
            test_data_path (str): Path to test data
            test_size (float): percentage of data to split to evaluation set
            random_state (int): seed for random operations
        """
        
        self.train_data = load_data(train_data_path)
        if self.train_data is None:
            raise ValueError("Training data is None")

        self.test_data = load_data(test_data_path)
        if self.test_data is None:
            raise ValueError("Test data is None")

        self.x_train, self.y_train, self.x_eval, self.y_eval, self.train_samples, self.train_labels, self.test_samples = self._data_load_and_process_data(test_size, random_state)

        
        self.labels_name = self._extract_labels()
    
    def count_labels(self):
        """
    Count the occurrences of each label in the training data.

    Returns:
    dict: A dictionary where keys are unique labels and values are the count of each label in the training data.

    Example:
    If self.train_labels = [1, 2, 1, 3, 2, 1], the function will return {1: 3, 2: 2, 3: 1}.
    """
        
        cnt = {}
        for label in self.train_labels:
            if label not in cnt:
                cnt[label] = 1
            else:
                cnt[label] = cnt[label] + 1
        
        return cnt
    
    def _extract_labels(self):
        """
    Extract unique labels from the training data.

    Returns:
    set: A set containing unique labels converted to strings.

    Example:
    If self.train_labels = [1, 2, 1, 3, 2, 1], the function will return {'1', '2', '3'}.
    """
        labels_set = set()
    
        for label in self.train_labels:
            labels_set.add(str(label))
            
        return labels_set
    

    def _data_load_and_process_data(self, test_size:float, random_state:int):
        """

        Process training and test data.

        Args:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator for reproducibility.

        Returns:
        tuple: A tuple containing four numpy arrays - x_train, y_train, x_eval, y_eval - representing the training and evaluation sets,
            as well as the original training samples, training labels, and test samples.

        Raises:
        ValueError: If train_labels is None or if the dimensions of train_samples and train_labels do not match.

        Example:
        x_train, y_train, x_eval, y_eval, train_samples, train_labels, test_samples = data_load_and_process_data(0.2, 42)
    
        """

        train_samples = self.train_data[0]
        train_labels = self.train_data[1]

        if train_labels is None:
            raise ValueError("train_labels is None, wrong file?")
        if train_samples.shape[0] != train_labels.shape[0]:
            raise ValueError("train_samples and train_labels dimension mismatch")

        test_samples = self.test_data[0]

        x_train, x_eval, y_train, y_eval = train_test_split(train_samples, train_labels, test_size=test_size,
                                                            random_state=random_state)

        return x_train, y_train, x_eval, y_eval, train_samples, train_labels, test_samples
