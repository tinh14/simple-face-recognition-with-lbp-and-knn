import os

CURRENT_FILE_PATH = os.path.abspath(__file__)
BASE_PATH = os.path.dirname(CURRENT_FILE_PATH)

DATASET_PATH = BASE_PATH + "/dataset"
PROCESSED_DATASET_PATH = BASE_PATH + "/processed_dataset"
TEST_IMAGE_PATH = BASE_PATH + "/image/mtp.jpg"
FEATURES_PATH = BASE_PATH + "/trained_data/features.npy"
LABELS_PATH = BASE_PATH + "/trained_data/labels.npy"

IMAGE_SIZE = 200
K = 5