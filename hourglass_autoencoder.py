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

            # Print encoding plot
            if epoch % 100 == 0:
                patterns = train_patterns.type(torch.FloatTensor)

                linear_layer = model[0](patterns)
                hidden_layer1 = model[1](linear_layer)
                hidden_layer2 = model[2](hidden_layer1)

                plt.scatter(hidden_layer2.data[:, 0], hidden_layer2.data[:, 1], s=0.5)
                plt.title("fold: {:3d}, epoch: {:3d}".format(fold, epoch))
                #plt.xlim((0,1))
                #plt.ylim((0,1))
                plt.xlabel('H1 value')
                plt.ylabel('H2 value')
                plt.savefig(params.ENCODING_DIR+"encoding_plot_fold{:03d}_ep{:03d}.png".format(fold,epoch))
                plt.clf()  # Clear the figure for the next loop

        print('\nTraining completed.')

    ### COMPUTING THA ACCURACY OF THE SINGLE FOLD ##################################

    outputfile1 = open('epoch_loss.dat', 'w')

    for i in range(N_epochs):
    	# loss
        tr_mean=train_sum[i]/folds_number
        if folds_number > 1:
            tr_dev_std=math.sqrt((train_sum2[i]/folds_number-tr_mean*tr_mean)/(folds_number-1))
        else:
            tr_dev_std=0

        val_mean=val_sum[i]/folds_number
        if folds_number > 1:
            val_dev_std=math.sqrt((val_sum2[i]/folds_number-val_mean*val_mean)/(folds_number-1))
        else:
            val_dev_std=0

        test_mean=test_sum[i]/folds_number
        if folds_number > 1:
            test_dev_std=math.sqrt((test_sum2[i]/folds_number-test_mean*test_mean)/(folds_number-1))
        else:
            val_dev_std=0

        outputfile1.write( str(i)+" ")
        outputfile1.write( str(tr_mean)+" ")
        outputfile1.write( str(tr_dev_std)+" ")
        outputfile1.write( str(val_mean)+" ")
        outputfile1.write( str(val_dev_std)+" ")
        outputfile1.write( str(test_mean)+" ")
        outputfile1.write( str(test_dev_std)+"\n")

    outputfile1.close()

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
