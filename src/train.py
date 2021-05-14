import pandas as pd
import numpy as np
import config
import utils
import pickle
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn import metrics
import inference

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    train_folds = pd.read_csv("../input/train_folds.csv")

    mae_list = []

    for fold_ in range(config.SPLITS):
        
        # temporary dataframes for train and test
        train_df = train_folds[train_folds.kfold != fold_].reset_index(drop=True)
        test_df = train_folds[train_folds.kfold == fold_].reset_index(drop=True)

        model = LGBMRegressor(random_state = 42, 
                                num_leaves= 25, 
                                max_depth=8, 
                                subsample=0.95, 
                                boosting_type='dart', 
                                num_iterations=200)
        
        # fit the model on training
        model.fit(train_df.drop(columns=['ID', 'kfold', 'total_cost']), train_df.total_cost)
            
        preds = model.predict(test_df.drop(columns=['ID', 'kfold', 'total_cost']))

        # calculate mean absolute error
        mae = metrics.mean_absolute_error(test_df.total_cost, preds)
        mae_list.append(mae)
        print(f"Fold: {fold_}")
        print(f"MAE = {mae}")

        print("")    
    print(f"MAE Summary: {np.sum(mae_list)/len(mae_list)}")    

pickle.dump(model, open(config.MODEL_PATH + config.LIGHTGBM_MODEL, 'wb'))
inference.predict()