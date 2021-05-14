import pandas as pd
import numpy as np
import re
import os
import config
import utils
from sklearn.preprocessing import LabelEncoder

def load_dataset(path):
    return pd.read_csv(path)

def remove_nulls(dataframe):
    dataframe['travel_with'] = dataframe['travel_with'].fillna('Alone')
    dataframe['total_female'] = dataframe['total_female'].fillna(0)
    dataframe['total_male'] = dataframe['total_male'].fillna(0)
    dataframe['most_impressing'] = dataframe['most_impressing'].fillna('No comments')

def encode_categorical(dataframe, cat_column_list):
    dataframe[cat_column_list] = dataframe[cat_column_list].apply(LabelEncoder().fit_transform)

def create_train_test_sets(dataframe, train_files_no):
    train = dataframe[:train_files_no].copy()
    test = dataframe[train_files_no:].copy()
    return train, test

def next_output_file_name(path):
    
    if len(os.walk(path).__next__()[2]) > 0:
        next_file = len(os.walk(path).__next__()[2]) + 1
    else:
        next_file = 1
    next_file_name = "submission_" + str(next_file) + ".csv"

    return next_file_name
