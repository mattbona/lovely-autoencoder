import os
import sys
import csv
import math
import matplotlib.pyplot as plt
import numpy as np

import src.params as params
import src.variables as variables

import torch
from torch.utils.data import TensorDataset, DataLoader

if params.SET_THREADS_NUMBER == True: torch.set_num_threads(params.THREADS_NUMBER)
if params.DETERMINISM ==  True: torch.manual_seed(params.SEED) # for determinism

def load_model_parameters(model, param_file):
    model.load_state_dict(torch.load(param_file))

def save_model_parameters(model, param_file):
    torch.save(model.state_dict(), param_file)

def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            np.savetxt(params.PARAMETERS_DIR+"/module_"+name+"_parameters.txt", param.data, delimiter='\n ')

def check_dirs():
    if not os.path.exists(params.RESULTS_DIR):
        try:
            os.mkdir(params.RESULTS_DIR)
        except:
            sys.exit('ERROR: Cannot create directory for results.')

    if params.PRINT_MODEL_PARAMETERS == True:
        if not os.path.exists(params.PARAMETERS_DIR):
            try:
                os.mkdir(params.PARAMETERS_DIR)
            except:
                sys.exit('ERROR: Cannot create directory for the parameters of the model.')

    if params.PRINT_ENCODING == True:
        if not os.path.exists(params.ENCODING_DIR):
            try:
                os.mkdir(params.ENCODING_DIR)
            except:
                sys.exit('ERROR: Cannot create directory for the encoding of the binding sites.')

def get_autoencoder(input_dim=0, hidden_dim=0, activation_func='LeakyReLU', is_bias=False):
    model = torch.nn.Sequential()

    model.add_module('input_linear', torch.nn.Linear(input_dim, hidden_dim, bias=is_bias))
    if activation_func == 'LeakyReLU':
        model.add_module('encode', torch.nn.LeakyReLU())
    if activation_func == 'ReLU':
        model.add_module('encode', torch.nn.ReLU())
    model.add_module('hidden_linear', torch.nn.Linear(hidden_dim, input_dim, bias=is_bias))
    if activation_func == 'LeakyReLU':
        model.add_module('decode', torch.nn.LeakyReLU())
    if activation_func == 'ReLU':
        model.add_module('decode',torch.nn.ReLU())

    return model

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

    if params.TEST == True:
        test_file_path = params.DATASET_PATH + params.TEST_FILE + '.dat'
        if os.path.isfile(test_file_path) == True:
            with open(test_file_path) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=' ')
                for i,row in enumerate(readCSV):
                    row = list(map(float,row[:]))
                    test_data_list.append(row)
        else:
            params.TEST = False
            print("WARNING: no test file detected. Proceding without test.")

def get_standardized_tensor(tensor):
    tensor_means = tensor.mean(dim=1, keepdim=True)
    tensor_stds = tensor.std(dim=1, keepdim=True)
    standardized_tensor = (tensor - tensor_means) / tensor_stds

    return standardized_tensor

def return_fold_train_valid_sets(train_data_list, ifold):
    if params.FOLDS_NUMBER <= 1:
        sys.exit("ERROR: you have to select more than 1 fold to have the external cross validation to work!")
    if len(train_data_list)%params.FOLDS_NUMBER != 0:
        sys.exit("ERROR: the total number of data ("+str(len(train_data_list))+") is not divisible by the number of folds ("+str(params.FOLDS_NUMBER)+")!")

    fold_train_patterns_list = train_data_list.copy()

    start = int( ifold*len(train_data_list)/params.FOLDS_NUMBER )
    end = int( (ifold+1)*len(train_data_list)/params.FOLDS_NUMBER )

    fold_val_patterns_list = fold_train_patterns_list[start:end]

    del fold_train_patterns_list[start:end]

    fold_train_patterns = torch.FloatTensor(fold_train_patterns_list)
    fold_val_patterns = torch.FloatTensor(fold_val_patterns_list)

    if params.STANDARDIZE_DATA == True:
        fold_train_patterns = get_standardized_tensor(fold_train_patterns)
        fold_val_patterns = get_standardized_tensor(fold_val_patterns)

    if params.OPTIMIZER == 'sgd':
        fold_trainset = TensorDataset(fold_train_patterns, fold_train_patterns)
        return fold_trainset, fold_val_patterns
    else:
        return fold_train_patterns, fold_val_patterns

def cumulate_loss(model, loss_fn, fold, epoch, train_patterns, validation_patterns, test_patterns):

    train_prediction = model(train_patterns)
    val_prediction = model(validation_patterns)
    if len(test_patterns) != 0 and params.TEST == True:
        test_prediction = model(test_patterns)

    loss_train = loss_fn(train_prediction, train_patterns)
    loss_val = loss_fn(val_prediction, validation_patterns)
    if len(test_patterns) != 0 and params.TEST == True:
        loss_test = loss_fn(test_prediction, test_patterns)

    if len(test_patterns) != 0 and params.TEST == True:
        print('[folds-group: %d, epoch: %d]\t train loss: %.3f\t val loss: %.3f\t test loss: %.3f' % (fold+1, epoch+1, loss_train.item(), loss_val.item(), loss_test.item() ))
    else:
        print('[folds-group: %d, epoch: %d]\t train loss: %.3f\t val loss: %.3f' % (fold+1, epoch+1, loss_train.item(), loss_val.item() ))

    variables.train_sum[epoch] += (loss_train.item())
    variables.train_sum2[epoch] += (loss_train.item())**2
    variables.val_sum[epoch] += (loss_val.item())
    variables.val_sum2[epoch] += (loss_val.item())**2
    if len(test_patterns) != 0 and params.TEST == True:
        variables.test_sum[epoch] += (loss_test.item())
        variables.test_sum2[epoch] += (loss_test.item())**2

def write_on_file_average_stddev_losses(file_path):
    outputfile = open(file_path, 'w')

    for i in range(params.EPOCHS_NUMBER):
        train_mean = variables.train_sum[i]/params.FOLDS_NUMBER
        train_dev_std = math.sqrt((variables.train_sum2[i]/params.FOLDS_NUMBER-train_mean*train_mean)/(params.FOLDS_NUMBER-1))

        val_mean = variables.val_sum[i]/params.FOLDS_NUMBER
        val_dev_std = math.sqrt((variables.val_sum2[i]/params.FOLDS_NUMBER-val_mean*val_mean)/(params.FOLDS_NUMBER-1))

        if params.TEST == True:
            test_mean = variables.test_sum[i]/params.FOLDS_NUMBER
            test_dev_std = math.sqrt((variables.test_sum2[i]/params.FOLDS_NUMBER-test_mean*test_mean)/(params.FOLDS_NUMBER-1))

        outputfile.write( str(i)+" ")
        outputfile.write( str(train_mean)+" ")
        outputfile.write( str(train_dev_std)+" ")
        outputfile.write( str(val_mean)+" ")

        if params.TEST == True:
            outputfile.write( str(val_dev_std)+" ")
            outputfile.write( str(test_mean)+" ")
            outputfile.write( str(test_dev_std)+"\n")
        else:
            outputfile.write( str(val_dev_std)+"\n")

    outputfile.close()

def save_encoding(model, fold, epoch, train_patterns):
    x = train_patterns
    for module_name, module in model.named_children():
        y = module(x)
        x = y
        if module_name == 'encode':
            variables.encoding.append({'fold': fold, 'epoch': epoch, 'h1': y.data[:, 0], 'h2': y.data[:, 0]})

def print_encoding_plot():
    for i, encode in enumerate(variables.encoding):
        plt.scatter(encode['h1'], encode['h2'], s=0.5)

        plt.title("fold: {:3d}, epoch: {:3d}".format((encode['fold']+1),(encode['epoch']+1)))
        #plt.xlim((0,1))
        #plt.ylim((0,1))
        plt.xlabel('H1 value')
        plt.ylabel('H2 value')
        plt.savefig(params.ENCODING_DIR+"encoding_plot_fold{:03d}_ep{:03d}.png".format((encode['fold']+1),(encode['epoch']+1)))
        plt.clf()  # Clear the figure for the next loop

def train_model_with_external_cross_val(model, loss_fn, optimizer, folds_number, epochs_number, batch_dimension=0):

    dev = "cpu"
    if params.GPU == True:
        if torch.cuda.is_available():
            print("Using GPU to enhance computation...")
            dev = "cuda:0"
        else:
            print("WARNING: no GPU detected! Utilizing CPU instead.")
            dev = "cpu"
    device = torch.device(dev)
    model.to(device) # sendig model to cpu or gpu

    append_data_in_lists(variables.train_patterns_list, variables.test_patterns_list)

    test_patterns = torch.FloatTensor(variables.test_patterns_list)
    test_patterns = test_patterns.to(device)
    if params.TEST == True:
        if len(test_patterns) != 0:
            params.TEST = True
            if params.STANDARDIZE_DATA == True:
                test_patterns = get_standardized_tensor(test_patterns)
        else:
            params.TEST = False

    for fold in range(folds_number):
        print("\n### Grouping of folds number %d ###"%(fold+1))

        if params.OPTIMIZER == 'sgd':
            trainset, validation_patterns = return_fold_train_valid_sets(variables.train_patterns_list, fold)
            trainloader = DataLoader(trainset, batch_size = batch_dimension, shuffle=True)
            train_patterns = trainset[:][0].to(device)
            validation_patterns = validation_patterns.to(device)
        else:
            train_patterns, validation_patterns = return_fold_train_valid_sets(variables.train_patterns_list, fold)
            train_patterns = train_patterns.to(device)
            validation_patterns = validation_patterns.to(device)

        model.apply(initialize_models_weights)
        for epoch in range(epochs_number):

            if params.OPTIMIZER == 'sgd':
                for idata, data in enumerate(trainloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device) 
                    optimizer.zero_grad()
                    y_pred = model(inputs)
                    loss = loss_fn(y_pred, labels)
                    loss.backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                predicted_patterns = model(train_patterns)
                loss = loss_fn(predicted_patterns, train_patterns)
                loss.backward()
                optimizer.step()

            cumulate_loss(model, loss_fn, fold, epoch, train_patterns, validation_patterns, test_patterns)

            if params.PRINT_ENCODING == True:
                if (epoch+1) % params.PRINT_NUMBER == 0:
                    save_encoding(model, fold, epoch, train_patterns)

        print('\nTraining completed.')
