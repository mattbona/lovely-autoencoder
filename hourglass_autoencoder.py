import torch
from torch.utils.data import TensorDataset, DataLoader
import csv
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

torch.set_num_threads(2)
torch.manual_seed(42) # for determinism

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

    # load training and test (if present) data
    util.append_data_in_lists(variables.train_patterns_list, variables.test_patterns_list)
    test_patterns = torch.FloatTensor(variables.test_patterns_list)
    if len(test_patterns) != 0:
        is_test = True

    ### CREATION AND GROUPING OF THE DIFFERENT FOLDS ###############################
    print("\nStart creation, grouping and training of/on the different folds...")

    for fold in range(params.FOLDS_NUMBER):
        print("\n### Grouping of folds number %d ###" % (fold+1))

        train_patterns, validation_patterns = util.return_fold_train_valid_sets(variables.train_patterns_list, fold)

        model.apply(util.initialize_models_weights)	# weights initialization
        for epoch in range(params.EPOCHS_NUMBER):  # loop over the dataset multiple times

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y_pred = model(train_patterns)
            loss = loss_fn(y_pred, train_patterns)
            loss.backward()
            optimizer.step()

            util.cumulate_loss(fold, epoch, model, loss_fn, train_patterns, validation_patterns, test_patterns)

            if params.PRINT_ENCODING == True:
                if (epoch+1) % params.PRINT_NUMBER == 0:
                    save_encoding(fold, epoch, model, train_patterns)

        print('\nTraining completed.')

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
