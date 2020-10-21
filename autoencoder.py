import src. params as params
import src.variables as variables
import src.ae_utils as util

def run(input_dim, central_hidden_dim, activation_func, bias, loss, opt,
        learning_rate, momentum, weight_decay, folds_number, epochs_number, batch_dim):

    model = util.get_autoencoder(input_dim,central_hidden_dim,activation_func,bias)
    loss_fn = util.get_loss_function(loss)
    optimizer = util.get_optimizer(model, opt, learning_rate, momentum, weight_decay)

    util.train_model_with_external_cross_val(model, loss_fn, optimizer, folds_number, epochs_number, batch_dim)

    util.write_on_file_average_stddev_losses(params.RESULTS_DIR+'epoch_loss.dat')

    if params.PRINT_ENCODING == True:
        print('Printing encoding plots...')
        util.print_encoding_plot()
    if params.PRINT_MODEL_PARAMETERS == True:
        print('Printing network parameters...')
        util.print_model_parameters(model)

if __name__ == '__main__':

    util.check_dirs()
    run(params.INPUT_DIMENSION, params.CENTRAL_HIDDEN_DIMENSION, params.ACTIVATION_FUNCTION,
        params.BIAS, params.LOSS, params.OPTIMIZER, params.LEARNING_RATE, params.MOMENTUM, params.WEIGHT_DECAY,
        params.FOLDS_NUMBER, params.EPOCHS_NUMBER, params.BATCH_DIMENSION)
