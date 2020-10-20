import os
import sys
import src.params as params
import torch

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
