import os
import sys
import csv

import torch
from torch.utils.data import TensorDataset, DataLoader

import src.params as params
import src.variables as variables

def check_dirs():
    if not os.path.exists(params.RESULTS_DIR):
        try:
            os.mkdir(params.RESULTS_DIR)
        except:
            sys.exit('ERROR: Cannot create directory for results.')

    if params.PRINT_ENCODING == True:
        if not os.path.exists(params.ENCODING_DIR):
            try:
                os.mkdir(params.ENCODING_DIR)
            except:
                sys.exit('ERROR: Cannot create directory for the encoding of the binding sites.')

def get_hourglass_autoencoder(input_dim, central_hidden_dim, activation_func, is_bias=True):
    first_hidden_dim = int(input_dim*1.1)

    model = torch.nn.Sequential()
    model.add_module('input_linear', torch.nn.Linear(input_dim, first_hidden_dim, bias=is_bias))
    if activation_func == 'LeakyReLU':
        model.add_module('leakyrelu', torch.nn.LeakyReLU())
    if activation_func == 'ReLU':
        model.add_module('relu', torch.nn.ReLU())
    model.add_module('hidden_linear1', torch.nn.Linear(first_hidden_dim, central_hidden_dim, bias=is_bias))
    if activation_func == 'LeakyReLU':
        model.add_module('encode', torch.nn.LeakyReLU())
    if activation_func == 'ReLU':
        model.add_module('encode', torch.nn.ReLU())
    model.add_module('hidden_linear2', torch.nn.Linear(central_hidden_dim, first_hidden_dim, bias=is_bias))
    if activation_func == 'LeakyReLU':
        model.add_module('decode', torch.nn.LeakyReLU())
    if activation_func == 'ReLU':
        model.add_module('decode', torch.nn.ReLU())
    model.add_module('last_hidden_linear', torch.nn.Linear(first_hidden_dim, input_dim, bias=is_bias))
    if activation_func == 'LeakyReLU':
        model.add_module('leakyrelu', torch.nn.LeakyReLU())
    if activation_func == 'ReLU':
        model.add_module('relu', torch.nn.ReLU())

    return model

def get_loss_function(loss):
    if loss == 'mse':
        return torch.nn.MSELoss(reduction='mean')

def get_optimizer(model, opt, learning_rate=0.001, moment=0.5, wd=1E-5):
    if opt == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=moment, weight_decay=wd)
    if opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

def initialize_models_weights(layer):
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

def append_data_in_lists(train_data_list, test_data_list):
    train_file_path = params.DATASET_PATH + params.TRAIN_FILE + '.dat'
    if os.path.isfile(train_file_path) == True:
        with open(train_file_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=' ')
            for i,row in enumerate(readCSV):
                row = list(map(float,row[:]))
                train_data_list.append(row)
    else:
        sys.exit("ERROR: train file not found! Check the path.")

    test_file_path = params.DATASET_PATH + params.TEST_FILE + '.dat'
    if os.path.isfile(test_file_path) == True:
        with open(test_file_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=' ')
            for i,row in enumerate(readCSV):
                row = list(map(float,row[:]))
                test_data_list.append(row)
    else:
        print("WARNING: no test file detected. Proceding without test.")

def return_fold_train_valid_sets(train_data_list, ifold):
    if params.FOLDS_NUMBER <= 1:
        sys.exit("ERROR: you have to select more than 1 fold to have the external cross validation to work!")
    if len(train_data_list)%params.FOLDS_NUMBER != 0:
        sys.exit("ERROR: the total number of data is not divisible by the number of folds!")

    fold_train_patterns_list = train_data_list.copy()

    start = int( ifold*len(train_data_list)/params.FOLDS_NUMBER )
    end = int( (ifold+1)*len(train_data_list)/params.FOLDS_NUMBER )

    fold_val_patterns_list = fold_train_patterns_list[start:end]

    del fold_train_patterns_list[start:end]

    fold_train_patterns = torch.FloatTensor(fold_train_patterns_list)
    fold_val_patterns = torch.FloatTensor(fold_val_patterns_list)

    return fold_train_patterns, fold_val_patterns

def cumulate_loss(fold, epoch, model, loss_fn, train_patterns, validation_patterns, test_patterns):

    train_prediction = model(train_patterns)
    val_prediction = model(validation_patterns)
    if len(test_patterns) != 0:
        test_prediction = model(test_patterns)

    loss_train = loss_fn(train_prediction, train_patterns)
    loss_val = loss_fn(val_prediction, validation_patterns)
    if len(test_patterns) != 0:
        loss_test = loss_fn(test_prediction, test_patterns)

    if len(test_patterns) != 0:
        print('[folds-group: %d, epoch: %d]\t train loss: %.3f\t val loss: %.3f\t test loss: %.3f' % (fold+1, epoch+1, loss_train.item(), loss_val.item(), loss_test.item() ))
    else:
        print('[folds-group: %d, epoch: %d]\t train loss: %.3f\t val loss: %.3f' % (fold+1, epoch+1, loss_train.item(), loss_val.item() ))

    variables.train_sum[epoch] += (loss_train.item())
    variables.train_sum2[epoch] += (loss_train.item())**2
    variables.val_sum[epoch] += (loss_val.item())
    variables.val_sum2[epoch] += (loss_val.item())**2
    if len(test_patterns) != 0:
        variables.test_sum[epoch] += (loss_train.item())
        variables.test_sum2[epoch] += (loss_train.item())**2
