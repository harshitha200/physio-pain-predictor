import argparse
import warnings
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
from statistics import mean, variance
import csv

# Ignore the data conversion warning
warnings.filterwarnings("ignore")

def preprocess_data(data_type: str, input_file_path: str):
    """
    Preprocesses the input data and returns numpy arrays for features (X) and targets (y).

    Args:
        data_type (str): Type of data to process ('dia', 'sys', 'eda', 'res', or 'all').
        input_file_path (str): Path to the input CSV file.

    Returns:
        numpy array, numpy array: Processed features (X) and targets (y).
    """
    data_types = {
        'dia': 'BP Dia_mmHg',
        'sys': 'LA Systolic BP_mmHg',
        'eda': 'EDA_microsiemens',
        'res': 'Respiration'
    }
    target_dict = {'No Pain': 0, 'Pain': 1}

    data, target = [], []
    with open(input_file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if data_type != 'all':
                data_type_key = data_types[data_type]
                if data_type_key in row[1]:
                    res = [eval(i) for i in row[3:]]
                    data.append([mean(res), variance(res), min(res), max(res)])
                    target.append(target_dict[row[2]])
            else:
                for data_type_key, data_type_value in data_types.items():
                    if data_type_value in row[1]:
                        res = [eval(i) for i in row[3:]]
                        data.append([mean(res), variance(res), min(res), max(res)])
                        target.append(target_dict[row[2]])

    return np.array(data), np.array(target)

def perform_cross_validation(X, y):
    """
    Performs k-fold cross-validation using Support Vector Machine classifier.

    Args:
        X (numpy array): Features for training.
        y (numpy array): Targets for training.

    Returns:
        list, list: Predictions for each fold, indices of test samples for each fold.
    """
    clf = svm.SVC()
    predictions = []
    test_indices = []
    kf = KFold(n_splits=10)
    for _, (train_index, test_index) in enumerate(kf.split(X)):
        clf.fit(X[train_index], y[train_index])
        predictions.append(clf.predict(X[test_index]))
        test_indices.append(test_index)
    return predictions, test_indices

def print_evaluation_metrics(predictions, test_indices, y_true):
    """
    Prints evaluation metrics including confusion matrix, precision, recall, and accuracy.

    Args:
        predictions (list of arrays): Predicted values for each fold.
        test_indices (list of arrays): Indices of test samples for each fold.
        y_true (array): True labels of data samples.

    Returns:
        None
    """
    sum_conf_matrix = np.zeros((2, 2))
    for i, _ in enumerate(predictions):
        ground_truth_fold = y_true[test_indices[i]]
        pred_fold = predictions[i]
        conf_matrix_fold = confusion_matrix(ground_truth_fold, pred_fold)
        sum_conf_matrix += conf_matrix_fold
    
    avg_conf_matrix = sum_conf_matrix / len(predictions)
    print("Average confusion matrix:\n", avg_conf_matrix)
    
    ground_truth = np.concatenate([y_true[test_indices[i]] for i in range(len(predictions))])
    final_predictions = np.concatenate(predictions)
    avg_precision = precision_score(ground_truth, final_predictions, average='macro')
    avg_recall = recall_score(ground_truth, final_predictions, average='macro')
    avg_accuracy = accuracy_score(ground_truth, final_predictions)
    
    print("Average precision: {:.4f}".format(avg_precision))
    print("Average recall: {:.4f}".format(avg_recall))
    print("Average accuracy: {:.4f}".format(avg_accuracy))

def main():
    """
    Main function to execute data preprocessing, cross-validation, and evaluation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_type")
    parser.add_argument("input_file_path")
    args = parser.parse_args()
    
    X, y = preprocess_data(args.data_type, args.input_file_path)
    predictions, test_indices = perform_cross_validation(X, y)
    print_evaluation_metrics(predictions, test_indices, y)

if __name__ == "__main__":
    main()