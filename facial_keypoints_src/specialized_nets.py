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

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

# Use early stoppin http://danielnouri.org/notes/category/deep-learning/

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

X, y = load()

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


#from lasagne.regularization import 
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
    max_epochs=500,
    on_epoch_finished=[
        EarlyStopping(patience=200)
        ],
    verbose=1
    )

from sklearn.grid_search import GridSearchCV
param_grid = {
'more_params': [{'hidden_num_units': 100}, {'hidden_num_units':150}, {'hidden_num_units': 200}],
'update_momentum': [0.9, 0.98]
}
#gs = GridSearchCV(net1, param_grid, cv=2, refit=False, verbose=4)
#joblib.dump(gs, 'gridsearch_pre.pkl') 

#gs.fit(X, y)
#joblib.dump(gs,'gridsearch.pkl')

#x = gs.get_params()
#print '\nparams:\n'
#print(x)
#joblib.dump(x, 'gridsearch_params.pkl')


SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            )
        ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
            )
        ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            )
        ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            )
        ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            )
        ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            )
        ),
    ]

from collections import OrderedDict
from sklearn.base import clone

def fit_specialists():
    specialists = OrderedDict()

    net = NeuralNet(
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
        output_num_units=30,  # 30 target values for original, but we're changing that  on line 227
        # optimization params
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(0.01)),
        update_momentum=theano.shared(float32(0.9)),
        regression=True, 
        max_epochs=500,
        on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=200)
            ],
        verbose=1
        )

    for setting in SPECIALIST_SETTINGS:
        cols = setting['columns']
        X, y = load(cols=cols)

        model = clone(net)
        model.output_num_units = y.shape[1]
        # set number of epochs relative to number of training examples:
        model.max_epochs = int(1e7 / y.shape[0])
        if 'kwargs' in setting:
            # an option 'kwargs' in the settings list may be used to
            # set any other parameter of the net:
            vars(model).update(setting['kwargs'])

        print("Training model for columns {} for up to {} epochs".format(
            cols, model.max_epochs))
        model.fit(X, y)
        specialists[cols] = model

    with open('net-specialists.pickle', 'wb') as f:
        # we persist a dictionary with all models:
        pickle.dump(specialists, f, -1)

fit_specialists()