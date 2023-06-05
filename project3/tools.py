import torch
import pandas as pd
from collections import Counter
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.act = torch.nn.ReLU()
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer
        self.act_out = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.hidden(x))  # activation function for hidden layer
        x = self.act_out(self.out(x))
        return x


# codes below refers to:
# https://github.com/UBC-MDS/Income-Predictors-for-US-Adults/blob/master/src/load_data.py
COL_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
             'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
NUMERICAL_COL_NAMES = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']


def read_csv(csv_file_name):
    # input data
    census_data = pd.read_csv(filepath_or_buffer=csv_file_name, skiprows=1, header=None, names=COL_NAMES)
    # remove 'fnlwgt'
    census_data = census_data.drop(columns='fnlwgt', axis=1)
    return census_data


def read_label(label_file_name):
    # input label
    labels = []
    with open(file=label_file_name) as tlf:
        for label in tlf:
            labels.append(int(label))
    return labels


def make_dict(census_data):
    # remove lines with question marks
    for names in COL_NAMES:
        if names != 'fnlwgt':
            census_data = census_data.query(names + ' != "?"')

    # for string arguments, sum up their contribution
    label_one_data = census_data.query('label == 1').drop(columns='label', axis=1)
    all_data = census_data.drop(columns='label', axis=1)

    # drop numerical cols
    for names in NUMERICAL_COL_NAMES:
        label_one_data = label_one_data.drop(columns=names, axis=1)
        all_data = all_data.drop(columns=names, axis=1)

    # count and store in dict
    counter_one = Counter(label_one_data.values.flatten()).items()
    counter_all = Counter(all_data.values.flatten()).items()
    string_argument_label1_cnt = {}
    string_argument_dict = {}
    for name, cnt in counter_one:
        string_argument_label1_cnt[name] = cnt
    for name, cnt in counter_all:
        string_argument_dict[name] = (
                string_argument_label1_cnt[name] if name in string_argument_label1_cnt.keys() else 0
            ) / cnt * 100

    # save
    np.save('string_argument_dict', string_argument_dict)
    return string_argument_dict


def process_columns(census_data, string_argument_dict):
    for names in COL_NAMES:
        if names in NUMERICAL_COL_NAMES or names == 'fnlwgt':
            if names == 'capital_loss' or names == 'capital_gain':
                census_data[names] = census_data[names].map(lambda x: np.log(x+1))
        else:
            census_data[names] = census_data[names].map(
                    lambda x: string_argument_dict[x] if x in string_argument_dict.keys() else 0
                )
    return census_data


if __name__ == "__main__":
    pass

