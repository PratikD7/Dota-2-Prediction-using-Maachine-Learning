"""
This file provides the following pre-processing on the data:
1) Z- score normalization
"""

import statistics
import pandas as pd


# Load a CSV file
def load_csv(filename):
    data = pd.read_csv(filename)
    dataset = data.values
    return dataset


# Calculate mean and standard deviation of columns
def mstd(i, data):
    col_values = [row[i] for row in data]
    mean_value = statistics.mean(col_values)
    std_dev_value = statistics.stdev(col_values)
    return mean_value, std_dev_value


# Apply z-normalization on data
def z_norm():
    data = load_csv('FEATURES.csv')
    list_of_columnids = []
    for i in list_of_columnids:
        mstd(i, data)


z_norm()
