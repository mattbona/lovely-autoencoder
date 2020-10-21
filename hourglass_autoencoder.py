import src. params as params
import src.variables as variables
import src.ae_utils as util

if __name__ == '__main__':

    util.check_dirs()

    model = util.get_hourglass_autoencoder(params.INPUT_DIMENSION,params.CENTRAL_HIDDEN_DIMENSION,params.ACTIVATION_FUNCTION,params.BIAS)
    loss_fn = util.get_loss_function(params.LOSS)
    optimizer = util.get_optimizer(model, params.OPTIMIZER, params.LEARNING_RATE, params.MOMENTUM, params.WEIGHT_DECAY)

    util.train_model_with_external_cross_val(model, loss_fn, optimizer, params.FOLDS_NUMBER, params.EPOCHS_NUMBER, params.BATCH_DIMENSION)

    util.write_on_file_average_stddev_losses(params.RESULTS_DIR+'epoch_loss.dat')

    if params.PRINT_ENCODING == True:
        print('Printing encoding plots...')
        util.print_encoding_plot()
    if params.PRINT_MODEL_PARAMETERS == True:
        print('Printing network parameters...')
        util.print_model_parameters(model)
