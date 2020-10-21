import src. params as params
import src.variables as variables
import src.ae_utils as util

if __name__ == '__main__':

    util.check_dirs()

    model = util.get_hourglass_autoencoder(params.INPUT_DIMENSION,params.CENTRAL_HIDDEN_DIMENSION,params.ACTIVATION_FUNCTION,params.BIAS)
    loss_fn = util.get_loss_function(params.LOSS)
    optimizer = util.get_optimizer(model, params.OPTIMIZER, params.LEARNING_RATE, params.MOMENTUM, params.WEIGHT_DECAY)

    ### MAYBE LOAD PREVIOUSLY TRAINED PARAMETERS
    """
    #param_file = sys.argv[0][:-3]
    #param_file = "params_"+param_file+".pt"
    model.load_state_dict(torch.load(param_file))
    """

    util.external_cross_val_train(model, loss_fn, optimizer, params.FOLDS_NUMBER, params.EPOCHS_NUMBER, params.BATCH_DIMENSION)

    util.write_on_file_average_stddev_losses(params.RESULTS_DIR+'epoch_loss.dat', write_test=is_test)
    if params.PRINT_ENCODING == True:
        print('Printing encoding plots...')
        util.print_encoding_plot()

    ### PRINT PARAMETERS ###########################################################
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            #print (name, param.data)
            np.savetxt(name+".txt", param.data, delimiter='\n ')
    """
    ################################################################################

    ### SAVE TRAINED PARAMETERS IN A FILE ##########################################
    param_file = 'model_parameters'
    torch.save(model.state_dict(), param_file)

    ################################################################################
