from glocal_settings import LOCAL

LOCAL_DATA_PATH = "data/"

FOX_DATA_PATH = "/fp/projects01/ec30/factkg/"

SAVE_DATAFOLDER = "data/"
SIMPLE_FOLDER = "simple/"
FULL_FOLDER = "full/"
DPBEDIA_FOLDER = "dbpedia/"
DBPEDIA_LIGHT_FILENAME = "dbpedia_2015_undirected_light.pickle"
TRAIN_FILENAME = "train.csv"
VAL_FILENAME = "val.csv"
TEST_FILENAME = "test.csv"

if LOCAL:
    DATA_PATH = LOCAL_DATA_PATH
else:
    DATA_PATH = FOX_DATA_PATH
