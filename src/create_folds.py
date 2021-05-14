import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os

import config
import utils

# Read training files
train_set = utils.load_dataset(config.TRAIN_PATH)
test_set = utils.load_dataset(config.TEST_PATH)

# Get the number of training samples
train_sample = train_set.shape[0]

# Concatenate datasets
data = pd.concat([train_set, test_set], axis=0)

# Remove null values
utils.remove_nulls(data)

# Encode categorical values
utils.encode_categorical(data, config.CAT_COLUMNS)

# create the train and test sets
train, test = utils.create_train_test_sets(data, train_sample)
# print(train.shape, test.shape)

train["kfold"] = -1
  
train = train.sample(frac=1, random_state=42).reset_index(drop=True)
y = train.total_cost.values

kf = KFold(n_splits=config.SPLITS)
    
for folds, (train_, valid_) in enumerate(kf.split(X=train, y=y)):
  train.loc[valid_, 'kfold'] = folds

train.to_csv(os.path.join(config.INPUT_PATH, "train_folds.csv"), index=False)
test.to_csv(os.path.join(config.INPUT_PATH, 'test_cleaned.csv'), index=False)
