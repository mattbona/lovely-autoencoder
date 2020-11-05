import torch
import math
import src. params as params
import src.ae_utils as util

def run(central_hidden_dim, activation_func, bias, loss, opt,
        learning_rate, momentum, weight_decay, folds_number, epochs_number, intermediate_hidden_layer=False,
        batch_dim=128, parameters=False, encoding=False, gpu=False, nprint=100, standardize_data=False, fix_permutation=False):

    util.check_dirs()

    train_file_path = params.DATASET_PATH+params.TRAIN_FILE+'.dat'
    test_file_path = params.DATASET_PATH+params.TEST_FILE+'.dat'
    x_train, x_test = util.get_data_tensor_from_file(train_file_path, test_file_path, standardize_data=standardize_data)

    if fix_permutation == True:
        x_train = util.get_reconstructed_matrix_tensor_from_flattened_triangular(x_train, 10)
        x_test = util.get_reconstructed_matrix_tensor_from_flattened_triangular(x_test, 10)
        x_train = util.get_flattened_permuted_matrix_tensor(x_train)
        x_test = util.get_flattened_permuted_matrix_tensor(x_test)

    ae_net = util.Autoencoder(x_train.shape[1], x_train.shape[1], intermediate_hidden_layer)
    ae_net.compile(loss, opt, activation_func, learning_rate, momentum, weight_decay, bias)
    ae_net.summary()

    losses_history = ae_net.train_with_external_crossvalidation(x_train, folds_number, epochs_number, testset=x_test,
                                                                batch_dim=batch_dim, encoding=encoding, gpu=gpu, nprint=nprint)

    util.write_on_file_losses_average_stdev(losses_history, params.RESULTS_DIR+'epoch_loss.dat')

if __name__ == '__main__':

    run(params.CENTRAL_HIDDEN_DIMENSION, params.ACTIVATION_FUNCTION,
        params.BIAS, params.LOSS, params.OPTIMIZER, params.LEARNING_RATE, params.MOMENTUM, params.WEIGHT_DECAY,
        params.FOLDS_NUMBER, params.EPOCHS_NUMBER, params.INTERMEDIATE_HIDDEN_LAYERS, params.BATCH_DIMENSION,
        params.PRINT_MODEL_PARAMETERS, params.PRINT_ENCODING, params.GPU, params.PRINT_NUMBER, params.STANDARDIZE_DATA, params.FIX_PERMUTATION)
