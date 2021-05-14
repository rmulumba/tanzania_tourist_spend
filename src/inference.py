import config
import utils
import os
import pickle

def predict():

    submission_file_name = utils.next_output_file_name(config.OUTPUT_PATH)
    model = pickle.load(open(config.MODEL_PATH + config.LIGHTGBM_MODEL, 'rb'))

    test = utils.load_dataset(config.CLEAN_TEST_PATH)

    test_preds = model.predict(test.drop(columns=['ID', 'total_cost']))
    test['total_cost'] = test_preds

    # Create submission file

    sub_file = test[['ID', 'total_cost']]
    sub_file.columns = ['test_id', 'total_cost']
    
    sub_file.to_csv(os.path.join(config.OUTPUT_PATH, submission_file_name), index=False)

    if os.path.exists (os.path.join(config.OUTPUT_PATH, submission_file_name)):
        print(f"File : {submission_file_name} created successfully.")
    else:
        print(f"Error creating file {submission_file_name}")