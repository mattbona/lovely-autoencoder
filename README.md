# lovely-autoencoder

My best attempt to construct a lovely code that implement *Autoencoders Neural Networks*.

### Installing
If you want to use this code, after downloading this repo, make sure to have all the required libraries, that are:

|Package        |
|---------------|
|matplotlib     |
|numpy          |
|torch          |
|os             |
|sys            |         
|math           |
|csv            |

I suggest you to configure an ad hoc *conda environment*.

### Usage
After loading your datasets in the directory `dataset` and after specifing your training parameters of interest in the file `src/params.py`, you can simply start the training using the command `python autoencoder.py`.
The program will print on a file in the `results` dir the average and the standard deviation of the losses computed over train set, validation set (according to the external cross validation procedure) and test set (if test set is present and you want to print also this observables).
