import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt

torch.set_num_threads(2)
torch.manual_seed(42) # for determinism

#check if encoding dir exists
if not os.path.exists('encoding'):
    os.makedirs('encoding')

dataset_path="../dataset/"
test_file="test" # Name of the .dat test file in the dataset dir
train_file="train" # Name of the .dat train file in the dataset dir

### MODEL DEFINITION ########################
D_in = 45    # Dimension of the INPUT LAYER: distance matrix of binding sites
D_out = 45   # Dimension of the OUTPUT LAYER
H1 = 50     # Dimension of the fisrt HIDDEN LAYER
H = 90        # Dimension of the central HIDDEN LAYER

LEARNING_RATE = 0.001
MOMENTUM = 0.5
WEIGHT_DECAY = 1E-5

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1, bias=True),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H1, H, bias=True),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H, H1, bias=True),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(H1, D_out, bias=True),
)

loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

def init_weights(m):	# Funzione che inizializza i pesi dei layer nn.Linear() della rete definita con la funzione nn.Sequential()
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

### MAYBE LOAD PREVIOUSLY TRAINED PARAMETERS
"""
#param_file = sys.argv[0][:-3]
#param_file = "params_"+param_file+".pt"
model.load_state_dict(torch.load(param_file))
"""

### LOADING DATA FROM DATABASE #################################################
print("\nLoading data for training and testing ...")
## Load data for the first time from a regular file .dat
train_patterns_list = []
test_patterns_list = []
#read and select desired batch
with open(dataset_path+train_file+'.dat') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=' ')
    for i,row in enumerate(readCSV):
        row=list(map(float,row[:]))
        train_patterns_list.append(row)

with open(dataset_path+test_file+'.dat') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=' ')
    for i,row in enumerate(readCSV):
        row=list(map(float,row[:]))
        test_patterns_list.append(row)

test_patterns_tensor = torch.FloatTensor(test_patterns_list)
testset = TensorDataset(test_patterns_tensor, test_patterns_tensor)

test_patterns = testset[:][0]
test_labels = testset[:][1]

print("Loading of data completed.")

### CREATION AND GROUPING OF THE DIFFERENT FOLDS ###############################
print("\nStart creation, grouping and training of/on the different folds...")
folds_number = 13 # Number of folds for the external cross validation
N_epochs = 10000

tr_sum = [0]*N_epochs
val_sum = [0]*N_epochs
test_sum = [0]*N_epochs
tr_sum2 = [0]*N_epochs
val_sum2 = [0]*N_epochs
test_sum2 = [0]*N_epochs
for fold in range(folds_number):
    print("\n### Grouping of folds number %d ###" % (fold+1))

    fold_train_patterns_list = train_patterns_list.copy()

    start = int( fold*len(train_patterns_list)/folds_number )
    end = int( (fold+1)*len(train_patterns_list)/folds_number )
    print("Validation-fold start at: %d\t and end at: %d" % (start, end))

    val_pattern_list = fold_train_patterns_list[start:end]

    del fold_train_patterns_list[start:end]

    train_patterns_tensor = torch.FloatTensor(fold_train_patterns_list)
    val_pattern_tensor = torch.FloatTensor(val_pattern_list)

    trainset = TensorDataset(train_patterns_tensor, train_patterns_tensor)
    validationset = TensorDataset(val_pattern_tensor, val_pattern_tensor)

    validation_patterns = validationset[:][0]
    validation_labels = validationset[:][1]

    train_patterns = trainset[:][0]
    train_labels = trainset[:][1]

    print("Train size: ", len(trainset))
    print("Validation size: ", len(validationset))
    print("Test size: ", len(testset))

### TRAINING ON THE SINGLE FOLDS-GROUP #########################################
    print("\nTraining on the single grouping of folds...")

    model.apply(init_weights)	# weights initialization
    for epoch in range(N_epochs):  # loop over the dataset multiple times

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        y_pred = model(train_patterns)
        loss = loss_fn(y_pred, train_labels)
        loss.backward()
        optimizer.step()

		# Computing loss
        val_prediction = model(validation_patterns)
        train_prediction = model(train_patterns)
        test_prediction = model(test_patterns)

        loss_train=loss_fn(train_prediction,train_labels)
        loss_val=loss_fn(val_prediction,validation_labels)
        loss_test = loss_fn(test_prediction,test_labels)
        print('[folds-group: %d, epoch: %d]\t train loss: %.4f\t validation loss: %.4f' % (fold+1, epoch + 1, loss_train.item(), loss_val.item()))

        tr_sum[epoch]+=(loss_train.item())
        tr_sum2[epoch]+=(loss_train.item())**2
        val_sum[epoch]+=(loss_val.item())
        val_sum2[epoch]+=(loss_val.item())**2
        test_sum[epoch]+=(loss_train.item())
        test_sum2[epoch]+=(loss_train.item())**2

        # Print encoding plot
        if epoch % 100 == 0:
            patterns = train_patterns.type(torch.FloatTensor)

            linear_layer = model[0](patterns)
            hidden_layer1 = model[1](linear_layer)
            hidden_layer2 = model[2](hidden_layer1)

            plt.scatter(hidden_layer2.data[:, 0], hidden_layer2.data[:, 1], s=0.5)
            plt.title("fold: {:3d}, epoch: {:3d}, loss: {:10.2f}".format(fold, epoch,loss_train.item()))
            #plt.xlim((0,1))
            #plt.ylim((0,1))
            plt.xlabel('H1 value')
            plt.ylabel('H2 value')
            plt.savefig("encoding/encoding_plot_fold{:03d}_ep{:03d}.png".format(fold,epoch))
            plt.clf()  # Clear the figure for the next loop

    print('\nTraining completed.')

### COMPUTING THA ACCURACY OF THE SINGLE FOLD ##################################

outputfile1 = open('epoch_loss.dat', 'w')

for i in range(N_epochs):
	# loss
    tr_mean=tr_sum[i]/folds_number
    if folds_number > 1:
        tr_dev_std=math.sqrt((tr_sum2[i]/folds_number-tr_mean*tr_mean)/(folds_number-1))
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
