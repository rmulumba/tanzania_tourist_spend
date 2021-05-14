SPLITS = 10
RANDOM_STATE = 22
TRAIN_PATH = "../input/Train.csv"
TEST_PATH = "../input/Test.csv"
CLEAN_TEST_PATH = "../input/test_cleaned.csv"
MODEL_PATH = "../models/"
LIGHTGBM_MODEL = "lightgbm_model.sav"
INPUT_PATH = "../input/"
OUTPUT_PATH = "../output/"
CAT_COLUMNS = ['country', 'age_group', 'travel_with', 'purpose', 'main_activity', 'info_source',
                'tour_arrangement', 'package_transport_int', 'package_accomodation', 'package_food',
                'package_transport_tz', 'package_sightseeing', 'package_guided_tour', 'package_insurance',
                'night_mainland', 'night_zanzibar', 'payment_mode', 'first_trip_tz', 'most_impressing'
                ]