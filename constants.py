from glocal_settings import LOCAL
from glocal_settings import ML_NODES


LOCAL_DATA_PATH = "data/"

FOX_DATA_PATH = "/fp/projects01/ec30/factkg/"

SAVE_DATAFOLDER = "data/"
SUBGRAPH_FOLDER = "subgraphs/"
SIMPLE_FOLDER = "simple/"
FULL_FOLDER = "full/"
DBPEDIA_FOLDER = "dbpedia/"
DBPEDIA_LIGHT_FILENAME = "dbpedia_2015_undirected_light.pickle"
SAVED_MODEL_FOLDER = "models/"
RESULTS_FOLDER = "results/"
CHAT_GPT_FOLDER = "chatgpt"
EMBEDDINGS_FILENAME = "embeddings.pkl"
TRAIN_FILENAME = "train.csv"
VAL_FILENAME = "val.csv"
TEST_FILENAME = "test.csv"
DATA_SPLIT_FILENAMES = {"train": TRAIN_FILENAME, "val": VAL_FILENAME, "test": TEST_FILENAME}

N_EARLY_STOP_DEFAULT = 3
BERT_LAST_LAYER_DIM = [768]

if LOCAL or ML_NODES:
    DATA_PATH = LOCAL_DATA_PATH
else:
    DATA_PATH = FOX_DATA_PATH
