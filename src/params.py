GPU = True
SET_THREADS_NUMBER = True
THREADS_NUMBER = 2
DETERMINISM = True
SEED = 42

# External cross-validation parameters
FOLDS_NUMBER = 13

# Model definition
INPUT_DIMENSION = 45 # Input: superiors triangles of distances matrices of different binding sites
CENTRAL_HIDDEN_DIMENSION = 45

# Training parameters
EPOCHS_NUMBER = 10
BATCH_DIMENSION = 100
BIAS = True
ACTIVATION_FUNCTION = 'LeakyReLU'
LOSS = 'mse'

OPTIMIZER = 'sgd'
LEARNING_RATE = 0.001
MOMENTUM = 0.5
WEIGHT_DECAY = 0.1

DATASET_PATH = './dataset/' # Path to train and test files
STANDARDIZE_DATA = True
TRAIN_FILE = 'train' # Name of the .dat train file in the dataset dir

TEST = True
TEST_FILE = 'test'   # Name of the .dat test file in the dataset dir

# Working dirs
RESULTS_DIR = './results/'

PRINT_MODEL_PARAMETERS = False
PARAMETERS_DIR = './results/network_parameters'

PRINT_ENCODING = False
PRINT_NUMBER = 10
ENCODING_DIR = './results/encoding/'
