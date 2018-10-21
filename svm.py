'''
This file runs Support Vector Machine algorithm on the scraped data
'''
import copy
import math
import statistics
from random import randrange

import numpy
import pandas as pd
import numpy as np

# =============================================================================#
# Calculate the mean and std dev of all the columns of a dataset
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Calculate the mean and standard deviation of columns
def mstd(ds):
    mean = []
    std_dev = []
    for i in range(len(ds[0])):
        col_values = [row[i] for row in ds]
        mean.append(statistics.mean(col_values))
        std_dev.append(statistics.stdev(col_values))
    return mean, std_dev


# =============================================================================#
# Z-normalize the dataset i.e value = (value - mean)/stdev
def z_normalize(ds, mean, std_dev):
    for row in ds:
        for i in range(len(row)):
            if row[i] != None:
                row[i] = ((row[i]) - mean[i]) / std_dev[i]


# =============================================================================#
# Split the dataset into training and cross validation set
def cv_split(ds, n_folds):
    data_set_split = []
    data = list(ds)
    fold_size = int(len(ds) / n_folds)
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(data))
            # print(index)
            fold.append(data.pop(index))
        data_set_split.append(fold)
    return data_set_split


# =================================================================#
def create_train_data(folds, index):
    # Create train data
    train_data = list(folds)
    train_data = numpy.array(train_data)
    train_data = numpy.delete(train_data, (index), axis=0)
    index += 1
    train_data = train_data.tolist()
    train_data = sum(train_data, [])
    return train_data


# =================================================================#
def create_y_train(train_data):
    Y_train = []
    for i in range(len(train_data)):
        Y_train.append(train_data[i][0])
    for row in train_data:
        del row[0]

    Y_train = [int(row) for row in Y_train]
    return Y_train


# =================================================================#
def create_test_data(fold):
    test_data = []
    for row in fold:
        row_copy = list(row)
        test_data.append(row_copy)
        row_copy[0] = None
    for row in test_data:
        del row[0]
    return test_data


# =================================================================#
def evaluation_metrics(Y_train, Y_test, preds_train, preds_test,
                       accuracy_list_train, precision_list_train,
                       recall_list_train, accuracy_list_test,
                       precision_list_test, recall_list_test):
    accuracy_train = accuracy_score(Y_train, preds_train)
    accuracy_test = accuracy_score(Y_test, preds_test)
    precision_train = precision_score(Y_train, preds_train)
    precision_test = precision_score(Y_test, preds_test)
    recall_train = recall_score(Y_train, preds_train)
    recall_test = recall_score(Y_test, preds_test)

    accuracy_list_train.append(accuracy_train)
    precision_list_train.append(precision_train)
    recall_list_train.append(recall_train)
    accuracy_list_test.append(accuracy_test)
    precision_list_test.append(precision_test)
    recall_list_test.append(recall_test)

    return accuracy_list_train, accuracy_list_test, precision_list_train, \
           precision_list_test, recall_list_train, recall_list_test

# =================================================================#
def svm_algorithm(dota_data, n_folds, gamma, C):
    # Split the dataset into training and cross validation set
    folds = cv_split(dota_data, n_folds)
    # Initialize the lists
    accuracy_list_train = []
    precision_list_train = []
    recall_list_train = []
    accuracy_list_test = []
    precision_list_test = []
    recall_list_test = []
    C_list = []
    G_list = []
    index = 0
    for fold in folds:
        # create train data
        train_data = create_train_data(folds, index)

        # Create train labels
        Y_train = create_y_train()

        # Normalize the train data
        mean, stdev = mstd(train_data)
        z_normalize(train_data, mean, stdev)

        # Create test data
        test_data = create_test_data(fold)

        # Normalize the test data
        z_normalize(test_data, mean, stdev)

        # Create target labels for test data
        Y_test = [row[0] for row in fold]
        Y_test = [int(row) for row in Y_test]

        # SVM Algorithm
        model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        model.fit(train_data, Y_train)
        preds_train = model.predict(train_data)
        preds_test = model.predict(test_data)

        accuracy_list_train, accuracy_list_test, precision_list_train, \
        precision_list_test, recall_list_train, recall_list_test = \
            evaluation_metrics(Y_train, Y_test, preds_train, preds_test,
                           accuracy_list_train, precision_list_train,
                           recall_list_train, accuracy_list_test,
                           precision_list_test, recall_list_test)

        C_list.append(C)
        G_list.append(gamma)

    return accuracy_list_train, precision_list_train, recall_list_train \
        , accuracy_list_test, precision_list_test, recall_list_test, C_list, \
           G_list

# =================================================================#
def main():
    dota_data = pd.read_csv('sample_features.csv').values
    n_folds = 10

    C = np.arange(-5, 11, 3)
    C = np.array([math.pow(2, x) for x in C])

    gamma = np.arange(-15, 6, 3)
    gamma = np.array([math.pow(2, x) for x in gamma])

    for c in C:
        for g in gamma:
            # Main SVM function
            accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test, c_value, g_value \
                = svm_algorithm(dota_data, n_folds, g, c)

            print("C values: ", c)
            print("Gamma values:", g)

            print("Training Accuracy : ", end='')
            print(accuracy_train)
            print("Mean Training Accuracy : ", end='')
            print(statistics.mean(accuracy_train))
            # print("Std deviation: ", end='')
            # print(statistics.stdev(accuracy_train))
            print("Testing Accuracy : ", end='')
            print(accuracy_test)
            print("Mean Testing Accuracy : ", end='')
            print(statistics.mean(accuracy_test))
            # print("Std deviation: ", end='')
            # print(statistics.stdev(accuracy_test))

            print("Training Precision : ", end='')
            print(precision_train)
            print("Mean Training Precision : ", end='')
            print(statistics.mean(precision_train))
            # print("Std deviation: ", end='')
            # print(statistics.stdev(precision_train))
            print("Testing Precision : ", end='')
            print(precision_test)
            print("Mean Testing Precision : ", end='')
            print(statistics.mean(precision_test))
            # print("Std deviation: ", end='')
            # print(statistics.stdev(precision_test))

            print("Training Recall : ", end='')
            print(recall_train)
            print("Mean Training Recall : ", end='')
            print(statistics.mean(recall_train))
            # print("Std deviation: ", end='')
            # print(statistics.stdev(recall_train))
            print("Testing Recall : ", end='')
            print(recall_test)
            print("Mean Testing Recall : ", end='')
            print(statistics.mean(recall_test))
            # print("Std deviation: ", end='')
            # print(statistics.stdev(recall_test))

            print('-' * 50)


# =================================================================#
if __name__ == '__main__':
    main()
