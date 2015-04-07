from pandas.io.parsers import read_csv #to read in our data to a dataframe
import numpy as np
import cPickle as pickle #cpickle in order to serialize neural networks - save the weights
from sklearn.utils import shuffle #shuffling the data for train, test split
from lasagne import layers #For creating layers of the neural network
from lasagne.updates import nesterov_momentum #For accelerated gradient descent
from lasagne.nonlinearities import sigmoid, rectify, tanh
from nolearn.lasagne import NeuralNet #For creating the neural network
import os.path #For getting the path to files on the computer
import matplotlib.pyplot as plt #For plotting images / graphs
import theano
from sklearn.externals import joblib
import sys
import theano
from sklearn.grid_search import GridSearchCV


#For shared learning/update rates
def float32(k):
    return np.cast['float32'](k)

FTRAIN = '../Data/training.csv'
FTEST = '../Data/test.csv'


def load(test=False, cols=None, drop_missing=True):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns. By default, samples with missing coordinates are dropped.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]
   
    df = df.dropna()  # drop all rows that have missing values in them


    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=65539)  # shuffle train data using seed for consistency
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

# Use early stopping http://danielnouri.org/notes/category/deep-learning/
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

#http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#changing-learning-rate-and-momentum-over-time
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    hidden_nonlinearity=sigmoid,
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values
    # optimization params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    regression=True, 
    max_epochs=1000,
    on_epoch_finished=[
        EarlyStopping(patience=50)
        ],
    verbose=1
    )


#params for cv
param_grid = {
'more_params': [{'hidden_num_units': 100}, {'hidden_num_units':150}, {'hidden_num_units': 200},
{'hidden_num_units': 250}, {'hidden_num_units': 300}]
}
 
#find net with best params , and then refit with all data on best net
gs = GridSearchCV(net1, param_grid, cv=2, refit=True, verbose=4)

#Load data
X,y=load()
#Fit CV model
gs.fit(X,y)
#Get the best estimator

with open('net1_gridsearch.pickle', 'wb') as f:
    # we serialize the best model:
    pickle.dump(gs, f, -1)