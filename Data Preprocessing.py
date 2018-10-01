import statistics
import pandas as pd


# Load a CSV file
def load_csv(filename):
    data = pd.read_csv(filename)
    dataset = data.values
    return dataset


def mstd(i,data):
    col_values = [row[i] for row in data]
    mean_value = statistics.mean(col_values)
    std_dev_value = statistics.stdev(col_values)
    return mean_value, std_dev_value

data = load_csv('features.csv')
list_of_columnids = []
for i in list_of_columnids:
    mstd(i, data)
