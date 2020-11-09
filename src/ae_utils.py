import os
import sys
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time

import src.params as params

import torch
from torch.utils.data import TensorDataset, DataLoader

if params.SET_THREADS_NUMBER == True: torch.set_num_threads(params.THREADS_NUMBER)
if params.DETERMINISM ==  True:
    torch.manual_seed(params.SEED) # for determinism
    random.seed(params.SEED)

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
def get_standardized_tensor(tensor):
    tensor_means = tensor.mean(dim=1, keepdim=True)
    tensor_stds = tensor.std(dim=1, keepdim=True)
    standardized_tensor = (tensor - tensor_means) / tensor_stds

    return standardized_tensor
def get_data_tensor_from_file(train_file_path, test_file_path='', standardize_data=False):
    train_patterns_list = []
    test_patterns_list = []

    if os.path.isfile(train_file_path) == True:
        with open(train_file_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=' ')
            for i,row in enumerate(readCSV):
                row = list(map(float,row[:]))
                train_patterns_list.append(row)
    else:
        sys.exit("ERROR: train file not found! Check the path.")

    random.shuffle(train_patterns_list) # shuffle train data
    train_patterns = torch.FloatTensor(train_patterns_list)
    if standardize_data == True:
        train_patterns = get_standardized_tensor(train_patterns)

    if test_file_path:
        if os.path.isfile(test_file_path) == True:
            with open(test_file_path) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=' ')
                for i,row in enumerate(readCSV):
                    row = list(map(float,row[:]))
                    test_patterns_list.append(row)
        else:
            sys.exit("ERROR: test file not found! Check the path.")

        test_patterns = torch.FloatTensor(test_patterns_list)
        if standardize_data == True:
            test_patterns = get_standardized_tensor(test_patterns)

        return train_patterns, test_patterns

    return train_patterns
def get_reconstructed_matrix_tensor_from_flattened_triangular(tensor, dim):
    triu_i = np.triu_indices(dim,1)
    invert_triu_i = (triu_i[1],triu_i[0])
    matrix = np.zeros((dim,dim))

    reconstruted_matrix_tensor = torch.tensor(()).new_zeros(tensor.shape[0],dim,dim)
    for ifmatrix, flattened_matrix in enumerate(tensor):
        matrix[triu_i] = flattened_matrix
        matrix[invert_triu_i] = matrix[triu_i]
        reconstruted_matrix_tensor[ifmatrix] = torch.from_numpy(matrix)

    return reconstruted_matrix_tensor
def get_permuted_matrix_tensor_by_column_sum(tensor):
    permuted_matrix_tensor = torch.tensor(()).new_zeros(tensor.shape[0],tensor.shape[1],tensor.shape[1])
    for imatrix, matrix in enumerate(tensor):
        sumCol = matrix.sum(axis=0)
        perm = np.argsort(sumCol)
        permuted_matrix_tensor[imatrix] = matrix[:,perm]
        permuted_matrix_tensor[imatrix] = permuted_matrix_tensor[imatrix][perm]

    return permuted_matrix_tensor
def get_flattened_permuted_matrix_tensor(tensor):
    tensor = get_permuted_matrix_tensor_by_column_sum(tensor)
    triu_i = np.triu_indices(tensor.shape[1],1)
    tensor = tensor[:,triu_i[0],triu_i[1]]

    return tensor

def print_encoding_plot(encoding, encoding_dir):
    for i, encode in enumerate(encoding):
        plt.scatter(encode['h1'], encode['h2'], s=0.5)

        plt.title("fold: {:3d}, epoch: {:3d}".format((encode['fold']+1),(encode['epoch']+1)))
        #plt.xlim((0,1))
        #plt.ylim((0,1))
        plt.xlabel('H1 value')
        plt.ylabel('H2 value')
        plt.savefig(encoding_dir+"encoding_plot_fold{:03d}_ep{:03d}.png".format((encode['fold']+1),(encode['epoch']+1)))
        plt.clf()  # Clear the figure for the next loop
def write_on_file_losses_average_stdev(history, file_path):

    train_mean = history['train_loss'].sum(axis=1)/history['train_loss'].shape[1]
    train_squared_mean = np.square(history['train_loss']).sum(axis=1)/history['train_loss'].shape[1]
    train_var = (train_squared_mean - np.square(train_mean))/(history['train_loss'].shape[1]-1)
    if train_var.all() > 0:
        train_dev_std = np.sqrt(train_var)
    else:
        train_dev_std = 0

    val_mean = history['val_loss'].sum(axis=1)/history['val_loss'].shape[1]
    val_squared_mean = np.square(history['val_loss']).sum(axis=1)/history['val_loss'].shape[1]
    val_var = (val_squared_mean - np.square(val_mean))/(history['val_loss'].shape[1]-1)
    if val_var.all() > 0:
        val_dev_std = np.sqrt(val_var)
    else:
        val_dev_std = 0

    if 'test_loss' in history:
        test_mean = history['test_loss'].sum(axis=1)/history['test_loss'].shape[1]
        test_squared_mean = np.square(history['test_loss']).sum(axis=1)/history['test_loss'].shape[1]
        test_var = (test_squared_mean - np.square(test_mean))/(history['test_loss'].shape[1]-1)
        if test_var.all() > 0:
            test_dev_std = np.sqrt(test_var)
        else:
            test_dev_std = 0

        means_stdevs = np.stack((train_mean, train_dev_std, val_mean, val_dev_std, test_mean, test_dev_std), axis=1)
    else:
        means_stdevs = np.stack((train_mean, train_dev_std, val_mean, val_dev_std), axis=1)

    np.savetxt(file_path, means_stdevs, fmt='%.3f')

class Autoencoder:

    def __init__(self, input_dim, central_hidden_dim=2, intermediate_hidden_layer=False):

        self.model = torch.nn.Sequential()
        self.input_dim = input_dim
        self.central_hidden_dim = central_hidden_dim
        self.intermediate_hidden_layer = intermediate_hidden_layer
        self.activation_func = ''
        self.bias = 0
        self.loss = ''
        self.optimizer = ''
        self.learning_rate = 0
        self.momentum = 0
        self.weight_decay = 0
        self.losses_history = {}
        self.encoding_history = []
        self.device = torch.device('cpu')

    def summary(self):
        print("Model:")
        print(self.model)
    def load_model_parameters(self, param_file):
        self.model.load_state_dict(torch.load(param_file))
    def save_model_parameters(self, param_file):
        torch.save(self.model.state_dict(), param_file)
    def print_model_parameters(self, file_dir):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                np.savetxt(file_dir+"/module_"+name+"_parameters.txt", param.data, delimiter='\n ')
    def compile(self, loss='mse', opt='sgd', activation_func='LeakyReLU',
                learning_rate=0.001, momentum=0.5, weight_decay=1E-5, bias=True):

        self.activation_func = activation_func
        self.bias = bias

        if self.intermediate_hidden_layer == False:
            self.model.add_module('input_linear', torch.nn.Linear(self.input_dim, self.central_hidden_dim, bias=self.bias))
            if self.activation_func == 'LeakyReLU':
                self.model.add_module('encode', torch.nn.LeakyReLU())
            if self.activation_func == 'ReLU':
                self.model.add_module('encode', torch.nn.ReLU())
            if self.activation_func == 'Sigmoid':
                self.model.add_module('encode', torch.nn.Sigmoid())
            self.model.add_module('hidden_linear', torch.nn.Linear(self.central_hidden_dim, self.input_dim, bias=self.bias))
        else:
            intermediate_hidden_dim = int(self.input_dim*1.1)
            self.model.add_module('input_linear', torch.nn.Linear(self.input_dim, intermediate_hidden_dim, bias=self.bias))
            if self.activation_func == 'LeakyReLU':
                self.model.add_module('leakyrelu', torch.nn.LeakyReLU())
            if self.activation_func == 'ReLU':
                self.model.add_module('relu', torch.nn.ReLU())
            if self.activation_func == 'Sigmoid':
                self.model.add_module('sigmoid', torch.nn.Sigmoid())
            self.model.add_module('hidden_linear1', torch.nn.Linear(intermediate_hidden_dim, self.central_hidden_dim, bias=self.bias))
            if self.activation_func == 'LeakyReLU':
                self.model.add_module('encode', torch.nn.LeakyReLU())
            if self.activation_func == 'ReLU':
                self.model.add_module('encode', torch.nn.ReLU())
            if self.activation_func == 'Sigmoid':
                self.model.add_module('encode', torch.nn.Sigmoid())
            self.model.add_module('hidden_linear2', torch.nn.Linear(self.central_hidden_dim, intermediate_hidden_dim, bias=self.bias))
            if self.activation_func == 'LeakyReLU':
                self.model.add_module('decode', torch.nn.LeakyReLU())
            if self.activation_func == 'ReLU':
                self.model.add_module('decode', torch.nn.ReLU())
            if self.activation_func == 'Sigmoid':
                self.model.add_module('decode', torch.nn.Sigmoid())
            self.model.add_module('last_hidden_linear', torch.nn.Linear(intermediate_hidden_dim, self.input_dim, bias=self.bias))

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        if loss == 'mse':
            self.loss = torch.nn.MSELoss(reduction='mean')
        if opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                             momentum=self.momentum, weight_decay=self.weight_decay)
        if opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
    def get_train_valid_sets(self, train_tensor, folds_number, ifold):
        if folds_number <= 1:
            sys.exit("ERROR: you have to select more than 1 fold to have the external cross validation to work!")
        if train_tensor.shape[0]%folds_number != 0:
            sys.exit("ERROR: the total number of data ("+str(train_tensor.shape[0])+") is not divisible by the number of folds ("+str(folds_number)+")!")

        fold_train_patterns = train_tensor.clone().detach()

        start = int( ifold*train_tensor.shape[0]/folds_number )
        end = int( (ifold+1)*train_tensor.shape[0]/folds_number )

        fold_val_patterns = fold_train_patterns[start:end].clone().detach()
        fold_train_patterns = torch.cat([fold_train_patterns[:start],fold_train_patterns[end:]])

        if self.optimizer == 'sgd':
            fold_trainset = TensorDataset(fold_train_patterns, fold_train_patterns)
            return fold_trainset, fold_val_patterns
        else:
            return fold_train_patterns, fold_val_patterns
    def initialize_losses_history(self, folds_number, epochs_number, test=False):
        if test == True:
            self.losses_history = {'train_loss': np.zeros((epochs_number,folds_number)),
                            'val_loss': np.zeros((epochs_number,folds_number)),
                            'test_loss': np.zeros((epochs_number,folds_number))}
        else:
            self.losses_history = {'train_loss': np.zeros((epochs_number,folds_number)),
                                   'val_loss': np.zeros((epochs_number,folds_number))}
    def get_losses_value(self, train_patterns, validation_patterns, test_patterns=torch.tensor([]), fix_permutation=False):
        if test_patterns.shape[0] > 0:
            if fix_permutation == True:
                dim = int(np.sqrt(train_patterns.shape[1]))
                train_patterns = train_patterns.reshape((train_patterns.shape[0],dim,dim))
                validation_patterns = validation_patterns.reshape((validation_patterns.shape[0],dim,dim))
                test_patterns = test_patterns.reshape((test_patterns.shape[0],dim,dim))

                train_patterns = get_flattened_permuted_matrix_tensor(train_patterns)
                validation_patterns = get_flattened_permuted_matrix_tensor(validation_patterns)
                test_patterns = get_flattened_permuted_matrix_tensor(test_patterns)

            loss_train = self.loss(self.model(train_patterns), train_patterns).item()
            loss_val = self.loss(self.model(validation_patterns), validation_patterns).item()
            loss_test = self.loss(self.model(test_patterns), test_patterns).item()
            return loss_train, loss_val, loss_test
        else:
            if fix_permutation == True:
                dim = int(np.sqrt(train_patterns.shape[1]))
                train_patterns = train_patterns.reshape((train_patterns.shape[0],dim,dim))
                validation_patterns = validation_patterns.reshape((validation_patterns.shape[0],dim,dim))

                train_patterns = get_flattened_permuted_matrix_tensor(train_patterns)
                validation_patterns = get_flattened_permuted_matrix_tensor(validation_patterns)

            loss_train = self.loss(self.model(train_patterns), train_patterns).item()
            loss_val = self.loss(self.model(validation_patterns), validation_patterns).item()
            return loss_train, loss_val
    def update_losses_history(self, fold, epoch, loss_train, loss_val, loss_test=np.nan):
        if np.isnan(loss_test):
            self.losses_history['train_loss'][epoch][fold] = loss_train
            self.losses_history['val_loss'][epoch][fold] = loss_val
        else:
            self.losses_history['train_loss'][epoch][fold] = loss_train
            self.losses_history['val_loss'][epoch][fold] = loss_val
            self.losses_history['test_loss'][epoch][fold] = loss_test
    def update_encoding_history(self, fold, epoch, train_patterns):
        x = train_patterns
        for module_name, module in self.model.named_children():
            y = module(x)
            x = y
            if module_name == 'encode':
                self.encoding_history.append({'fold': fold, 'epoch': epoch, 'h1': y.data[:, 0], 'h2': y.data[:, 0]})
    def initialize_models_weights(self, layer):
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
    def train_with_external_crossvalidation(self, x_train, folds_number, epochs_number, testset=torch.tensor([]),
                                            batch_dim=128, encoding=False, gpu=False, nprint=100, fix_permutation=False):
        if gpu == True:
            if torch.cuda.is_available():
                print("Using GPU to enhance computation...")
                self.device = torch.device('cuda:0')
            else:
                print("WARNING: no GPU detected! Utilizing CPU instead.")
        self.model.to(self.device) # sendig model to cpu or gpu

        if testset.shape[0] > 0:
            test = True
            testset = testset.to(self.device)
        else:
            test = False

        self.initialize_losses_history(folds_number, epochs_number, test=test)
        start_time = time.time()
        for fold in range(folds_number):
            print("\n### Grouping of folds number %d ###"%(fold+1))

            if self.optimizer == 'sgd':
                trainset, validation_patterns = self.get_train_valid_sets(x_train, folds_number, fold)
                trainloader = DataLoader(trainset, batch_size = batch_dimension, shuffle=True)
                train_patterns, validation_patterns = trainset[:][0].to(self.device), validation_patterns.to(self.device)
            else:
                train_patterns, validation_patterns = self.get_train_valid_sets(x_train, folds_number, fold)
                train_patterns, validation_patterns = train_patterns.to(self.device), validation_patterns.to(self.device)

            self.model.apply(self.initialize_models_weights)
            if fix_permutation == True: dim = int(np.sqrt(train_patterns.shape[1]))
            for epoch in range(epochs_number):

                if self.optimizer == 'sgd':
                    for idata, data in enumerate(trainloader):
                        inputs, labels = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.optimizer.zero_grad()
                        y_pred = self.model(inputs)
                        if fix_permutation == True:
                            y_pred = y_pred.reshape((y_pred.shape[0],dim,dim))
                            y_pred = get_flattened_permuted_matrix_tensor(y_pred)
                        loss = self.loss(y_pred, labels)
                        loss.backward()
                        self.optimizer.step()
                else:
                    self.optimizer.zero_grad()
                    predicted_patterns = self.model(train_patterns)
                    if fix_permutation == True:
                         predicted_patterns = predicted_patterns.reshape((predicted_patterns.shape[0],dim,dim))
                         predicted_patterns = get_flattened_permuted_matrix_tensor(predicted_patterns)
                    loss = self.loss(predicted_patterns, train_patterns)
                    loss.backward()
                    self.optimizer.step()

                if test == True:
                    train_loss, val_loss, test_loss = self.get_losses_value(train_patterns, validation_patterns, testset, fix_permutation)
                    self.update_losses_history(fold, epoch, train_loss, val_loss, test_loss)
                    if epoch % nprint == (nprint-1):
                        print('[folds-group: %d, epoch: %d]\t train loss: %.3f\t val loss: %.3f\t test loss: %.3f' %
                              (fold+1, epoch+1, train_loss, val_loss, test_loss))
                else:
                    train_loss, val_loss = self.get_losses_value(train_patterns, validation_patterns, fix_permutation)
                    self.update_losses_history(fold, epoch, train_loss, val_loss)
                    if epoch % nprint == (nprint-1):
                        print('[folds-group: %d, epoch: %d]\t train loss: %.3f\t val loss: %.3f' %
                              (fold+1, epoch+1, train_loss, val_loss))

                if encoding == True:
                    if epoch % nprint == (nprint-1):
                        self.update_encoding_history(fold, epoch, train_patterns)

        passed_time_tot_sec = time.time() - start_time
        passed_time_hour = passed_time_tot_sec/3600
        passed_time_min = (passed_time_hour - int(passed_time_hour))*60
        passed_time_sec = (passed_time_min - int(passed_time_min))*60
        print('\nTraining completed! The traing procedure has taken %dh %dmin %ds'
              % (int(passed_time_hour), int(passed_time_min), int(passed_time_sec)) )

        if encoding == True:
            return self.losses_history, self.encoding_history
        else:
            return self.losses_history
