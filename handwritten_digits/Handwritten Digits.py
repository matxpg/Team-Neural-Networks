
# coding: utf-8

## Handwritten Digit Classification

# **Outline:**
# 0. Description of the problem
# 1. Exploratory Analysis
# 2. Image Processing Considerations
# 3. Models
#     -KNN
#     -Multilayer perceptron
#     -Convolutional Neural Network
# 4. Actual handwritten digits

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[15]:

images = pd.read_csv('digitTrain.csv')


### Description of the problem

# The full title of my problem is as follows: (From Kaggle):
# 
# ** Classify handwritten digits using the famous MNIST data ** 
# The goal in this competition is to take an image of a handwritten single digit and determine what that digit is.
# 
# The MNIST data is a "classic" data set and has been explored quite a bit. In fact, if there's a paper on an improvement or new ML method, there's a good chance that MINST is used as a metric. 
# 
# The Kaggle data is a small subset of the whole data set which is 14Gb! The best score on the whole data set is 99.78%, using a convolutional neural network, which is pretty impressive.

### Exploratory Analysis

# Since this is a famous data set, as well as being from Kaggle, it's fair to assume that no direct cleaning is needed, unless we would like to clean our data from an image processing perspective. 

# The MINST dataset is a collection of 28x28 pixel greyscale images where the pixel value is an integer between 0 and 255 (This is referred to as 8-bit greyscale). This means we have 784 features. 
# 
# Each image has a label which is between 0 and 9 depending on what the digit is classified as. 

# In[3]:

print len(images)
print images['label'].value_counts(ascending = True)
print images['label'].astype(float).value_counts().mean()


# There are 42000 images in the train set, and there's an average of 4200 in each classification.
# 
# I think that this is likely kaggle selectively choosing the number of images, since the full data set is over 14Gb

# Now, let's look at what some of the images look like.

# In[4]:

labels = images['label']


# In[5]:

pixels = images.drop('label', axis = 1).values
reshaped = [pixels[i].reshape(28,28) for i in range(1,10)]


# In[6]:

import matplotlib.cm as cm
fig, axes= plt.subplots(1,4)
axes[0].imshow(reshaped[1], cmap = cm.gist_yarg);
axes[1].imshow(reshaped[2], cmap = cm.gist_yarg);
axes[2].imshow(reshaped[3], cmap = cm.gist_yarg);
axes[3].imshow(reshaped[8], cmap = cm.gist_yarg);


# We can see that there is a fair bit of variation in the style of writing, as we would expect. 
# 
# To get an idea of what the average digit looks like, let's plot the average digit as above.

# In[7]:

fig, axes= plt.subplots(2,5)
k = 0
averageImage = []
for i in range(2):
    for j in range(5):
        curnum = images[images['label'] == k].drop('label',axis =1)
        x = curnum.mean().values
        averageImage.append(x)
        x = x.reshape(28,28)
        axes[i][j].imshow(x, cmap = cm.gist_yarg);
        k=k+1


# We can see that many of the digits have weaker averages than others. In particular, 4 and 5 don't have very many darker black regions, which means that those digits don't have as consistent averages.

### Image Processing considerations

# Before we move on to fitting some models, I'm going to consider some image processing methods we can use to clean our data.
# 
# **Darker backgrounds**
# 
# In algorithms like KNN, we could suffer from the curse of dimensionality, especially because of the distance that edge pixels have relative to internal pixels. We can "reduce" this by darkening the background to a higher shade of grey. This is a delicate process because we do not want to lose information. 
# 
# If we can find a good way to bring the always white pixels to higher grey values, we will in some sense *compact* the space a little bit, without losing information about the digit. This will help algorithms such as KNN 
# 
# With that in mind, we will try using a values of 25 on pixels that are nearly always 0 (Which can be found from averages above). This will also include values that are less than 1 (Since that an average pixel value less than 1 is insignificant). If this gives us an improvement, I will consider tuning the value further.

# In[8]:

q = np.array(averageImage)
df = pd.DataFrame(q)
fig, axes= plt.subplots(2,5)
k = 0
for i in range(10):
    idx = np.where(df.ix[i] < 1)
    df.ix[i].ix[idx] = 15
for i in range(2):
    for j in range(5):
        curnum = df.ix[k]
        x = curnum.values
        x = x.reshape(28,28)
        axes[i][j].imshow(x, cmap = cm.gist_yarg);
        k=k+1
darkened = images.copy()
darkened = darkened.drop('label', axis = 1)
for k in range(len(darkened)):
    darkened.ix[k].ix[idx] = 15
darkened['label'] = labels


# **Averaging - Dimension Reduction**
# 
# Another possibility is some simple dimension reduction. We can partition the pixels in to 196 2x2 squares and average them together to produce a 14x14 pixel image. A reduction in dimensionality should help KNN and definitely offere benefits to Neural Networks.

# In[91]:

q = 0
newDF = images.loc[0,:'pixel195']
for image in images.values:
    label = image[0]
    image = image[1:].reshape(28,28)
    newImList = np.zeros(784).reshape(28,28)
    for i in range(0,27,2):
        for j in range(0,27,2):
            newPixel = np.mean([image[j][i],image[j+1][i], image[j][i+1], image[j+1][i+1]])
            newImList[j][i] = newPixel
            newImList[j+1][i] = newPixel
            newImList[j][i+1] = newPixel
            newImList[j+1][i+1] = newPixel
    
    newDF.loc[q] =np.append([label],newImList[:195])
    q = q + 1


# In[154]:

fig, axes = plt.subplots(1,2)
axes[0].imshow(images.loc[1].values[1:].reshape(28,28), cmap = cm.gist_yarg)
axes[1].imshow(newDF.loc[1][1:].reshape(28,28), cmap = cm.gist_yarg)


### Models

#### Using straight data, with no image processing:

# **KNN Classification**
# 
# I chose this because I felt that it would be a not bad first attempt at classification. I used 10 Stratified K-Folds, because the data set is large, and I'm not too concerned with computation time.

# In[12]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
for i in xrange(1,14,2):
    model = KNeighborsClassifier(n_neighbors = i)
    skf = StratifiedKFold(images['label'], n_folds=10)
    scores = [model.fit(images.drop('label', axis=1).values[train], labels[train]).score(images.drop('label', axis =1).values[test],
                                            labels[test])for train, test in skf]
    meanscore = np.mean(scores)
    print str(meanscore) + str(model)


# Using the default KNN with n = 1,3,5,7,9,11 and 13, the average 10 fold cross validation scores
# 
# 0.967380872089  n_neighbors=1 
# 
# 0.968261003894  n_neighbors=3
# 
# 0.967332930741  n_neighbors=5
# 
# 0.966428677921  n_neighbors=7
# 
# 0.965118597957  n_neighbors=9
# 
# 0.963475422204  n_neighbors=11
# 
# 0.962142297907  n_neighbors=13
#            
#            

# We can see that using simply the default settings (and a whole night of computation) that KNN does a really good job of classifying this problem. The curse of dimensionality is avoided because of how similar the pixel values are within a particular digit. We can see this when viewing the averages.
# 
# Now, let's choose our best KNN (k=3) and run it using identical settings on the darker background dataset and see if things are improved.

# In[97]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
model = KNeighborsClassifier(n_neighbors = 3)
skf = StratifiedKFold(darkened['label'], n_folds=10)
scores = [model.fit(darkened.drop('label', axis=1).values[train], labels[train]).score(darkened.drop('label', axis =1).values[test],
                                            labels[test])for train, test in skf]
meanscore = np.mean(scores)
print str(meanscore) + str(model)


# We can see that the darker background offered a slight improvement for KNN.

# **Simple Multilayer perceptron**
# 
# The NeuralNet uses a test/train split, but we include a dropout layer to avoid overfitting. As a first simple example, let's train a single hidden layer neural network. We will use a lower number of hidden units, and a lower drop out probability so that we can get a general idea of how well multilayer perceptron performs on our problem. 
#     
# I'm using a library called lasagne, which implements a really easy to use neural network interface, with full control of the parameters. 

# In[10]:

import pandas as pd
from sklearn.cross_validation import train_test_split, KFold 
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers
from lasagne.nonlinearities import softmax
from lasagne.updates import momentum, nesterov_momentum, sgd, rmsprop
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle



train_df = pd.read_csv('digitTrain.csv') 


train_label = train_df.values[:,0]
train_data = train_df.values[:, 1:]
print "train:", train_data.shape, train_label.shape


train_data = train_data.astype(np.float)
train_label = train_label.astype(np.int32)
train_data, train_label = shuffle(train_data, train_label, random_state = 21)


fc_1hidden = NeuralNet(
    layers = [
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('dropout', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape = (None, 784), #28x28
    hidden_num_units = 200, 
    dropout_p = 0.25, # dropout probability
    output_nonlinearity = softmax, 
    output_num_units = 10,  

    update = sgd, #Gradient descent
    update_learning_rate = 0.001,

    eval_size = 0.1,
    max_epochs = 100,
    verbose = 1,
    )

fc_1hidden.fit(train_data, train_label)




# In[11]:

def plot_loss(net):
    """
    Plot the training loss and validation loss versus epoch iterations with respect to 
    a trained neural network.
    """
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    pyplot.plot(train_loss, linewidth = 3, label = "train")
    pyplot.plot(valid_loss, linewidth = 3, label = "valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    #pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()
plot_loss(fc_1hidden)


# As we can see, it's not too difficult to get a good score. Since the hidden layer size is fairly small, the dropout probability is as well. We can see the validation loss decreasing with every iteration. Examining the plot of validation vs training loss demonstrates a learning curve, where we can identify the necessary number of epochs needed to train the network to an acceptable tolerance.
# 
# Now, let's consider a more complicated neural network. The following network is described in a paper about dropout as a metric for the improvement that dropout can provide.

# In[12]:

import pandas as pd
from sklearn.cross_validation import train_test_split, KFold 
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers
from lasagne.nonlinearities import softmax
from lasagne.updates import momentum, nesterov_momentum, sgd, rmsprop
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

train_df = pd.read_csv('digitTrain.csv') 

train_label = train_df.values[:,0]
train_data = train_df.values[:, 1:]
print "train:", train_data.shape, train_label.shape


train_data = train_data.astype(np.float)
train_label = train_label.astype(np.int32)
train_data, train_label = shuffle(train_data, train_label, random_state = 21)


betterNet= NeuralNet(
    layers = [  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
       ('hidden2', layers.DenseLayer),
        ('hidden3', layers.DenseLayer),
        ('dropout', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape = (None, 784),  # 28x28 input pixels per batch
    hidden1_num_units = 1024,  # number of units in hidden layer
    hidden2_num_units = 1024,
    hidden3_num_units = 1024,
    dropout_p = 0.4, # dropout probability
    output_nonlinearity = softmax,  # output layer uses softmax function
    output_num_units = 10,  # 10 labels

    # optimization method:
    #update = nesterov_momentum,
    update = sgd,
    update_learning_rate = 0.001,
    #update_momentum = 0.9,

    eval_size = 0.1,

    # batch_iterator_train = BatchIterator(batch_size = 20),
    # batch_iterator_test = BatchIterator(batch_size = 20),

    max_epochs = 100,  # we want to train this many epochs
    verbose = 1,
    )

betterNet.fit(train_data, train_label)



# In[13]:

plot_loss(betterNet)


# In[37]:

with open('betterNet.pickle', 'wb') as f:
    pickle.dump(betterNet, f, -1)


# We can see that using more hidden layers can produce a better result. 
# Using more units in each hidden layer can also improve our model, but it increases the computation time quite a bit. That being said, there's a high chance that we are overfitting the model here, since the validation accuracy levels out.
# 
# An important consideration when tuning a neural network is the dropout probability. Using more units in each layer increases the need for a higher drop out probability, because we have more models to approximate.
# 
# 

# **Image Processed variants**
# 
# Let's train the same neural networks on the image processed data sets. 
# 
# First the darkened background.

# In[17]:



train_label = darkened['label']
train_data = darkened.drop('label', axis = 1)
print "train:", train_data.shape, train_label.shape


train_data = train_data.astype(np.float)
train_label = train_label.astype(np.int32)
train_data, train_label = shuffle(train_data, train_label, random_state = 21)


img = NeuralNet(
    layers = [
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('dropout', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape = (None, 784), #28x28
    hidden_num_units = 200, 
    dropout_p = 0.25, # dropout probability
    output_nonlinearity = softmax, 
    output_num_units = 10,  

    update = sgd, #Gradient descent
    update_learning_rate = 0.001,

    eval_size = 0.1,
    max_epochs = 100,
    verbose = 1,
    )

img.fit(train_data, train_label)


# In[18]:

plot_loss(img)


# The image processed variant with darker backgrounds does slightly better (Not insigificant in terms of the randomness of drop out probability) and learns at a slightly faster rate. 
# 
# Now the downsampled data set:

# In[124]:

import pandas as pd
from sklearn.cross_validation import train_test_split, KFold 
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers
from lasagne.nonlinearities import softmax
from lasagne.updates import momentum, nesterov_momentum, sgd, rmsprop
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

train_label =np.array([ a[0] for a in newDF.loc[0:].values])
train_data = np.array([ a[1:197] for a in newDF.loc[0:].values])
print "train:", train_data.shape, train_label.shape
train_data = train_data.astype(np.float)
train_label = train_label.astype(np.int32)

train_data, train_label = shuffle(train_data, train_label, random_state = 21)

ds = NeuralNet(
    layers = [
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('dropout', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape = (None, 196), #14x14
    hidden_num_units = 200, 
    dropout_p = 0.25, # dropout probability
    output_nonlinearity = softmax, 
    output_num_units = 10,  

    update = sgd, #Gradient descent
    update_learning_rate = 0.001,

    eval_size = 0.1,
    max_epochs = 100,
    verbose = 1,
    )

ds.fit(train_data, train_label)


# In[126]:

plot_loss(ds)


# We can see that the downsampled data does not work all the well, but converges quite quickly and does cut the cost of an iteration in half. It's not really worth it.

# Let us now move to a convolutional neural network. Let's start with two convolutional layers, and one hidden layer (With dropout of course!)

# In[28]:

import pandas as pd
from sklearn.cross_validation import train_test_split 
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers
from lasagne.nonlinearities import softmax, rectify
from lasagne.updates import momentum, nesterov_momentum, sgd, rmsprop
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def plot_loss(net):
    """
    Plot the training loss and validation loss versus epoch iterations with respect to 
    a trained neural network.
    """
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    pyplot.plot(train_loss, linewidth = 3, label = "train")
    pyplot.plot(valid_loss, linewidth = 3, label = "valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    #pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()


train_df = pd.read_csv('digittrain.csv') 

train_label = train_df.values[:, 0]
train_data = train_df.values[:, 1:]
print "train:", train_data.shape, train_label.shape


train_data = train_data.astype(np.float)
train_label = train_label.astype(np.int32)
train_data, train_label = shuffle(train_data, train_label, random_state = 21)


train_data = train_data.reshape(-1, 1, 28, 28) 
CUDA_CONVNET = False
if CUDA_CONVNET:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer, MaxPool2DCCLayer
    Conv2DLayer = Conv2DCCLayer
    MaxPool2DLayer = MaxPool2DCCLayer
else:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

cnn = NeuralNet(
    layers = [  # three layers: one hidden layer
        ('input', layers.InputLayer),

        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),

        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer), 
        ('dropout2', layers.DropoutLayer),      

        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),

        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape = (None, 1, 28, 28),  # 28x28 input pixels per batch

    conv1_num_filters = 32, conv1_filter_size = (3, 3), pool1_ds = (2, 2), dropout1_p = 0.5,
    conv2_num_filters = 64, conv2_filter_size=(2, 2), pool2_ds=(2, 2), dropout2_p = 0.5,
    conv3_num_filters = 128, conv3_filter_size = (2, 2), pool3_ds = (2, 2), dropout3_p = 0.5, 

    hidden4_num_units = 500, dropout4_p = 0.5,

    output_num_units = 10,  # 10 labels

    conv1_nonlinearity = rectify, conv2_nonlinearity = rectify, conv3_nonlinearity = rectify, 
    hidden4_nonlinearity = rectify, 
    output_nonlinearity = softmax,  # output layer uses softmax function
    
    # optimization method:
    #update = adagrad,
    update = rmsprop,
    update_learning_rate = 0.0001,
    #update_learning_rate = 0.01,
 
    eval_size = 0.1,

    max_epochs = 200,  # we want to train this many epochs
    verbose = 1,
    )

cnn.fit(train_data, train_label)
plot_loss(cnn)


# In[125]:

import cPickle as pickle
with open('knn.pickle', 'wb') as f:
    pickle.dump(model, f, -1)


# In[132]:

import cPickle as pickle
with open('conv.pickle', 'rb') as f:
    cnn = pickle.load(f)


# In[142]:

test_df = pd.read_csv('test.csv')
test_data = test_df.values
test_data = test_data.reshape(-1, 1, 28, 28)
pred = cnn.predict(test_data)

output = pd.DataFrame(data = {"ImageId": range(1, 28001), "Label": pred})
output.to_csv("./fc_2hidden_predict.csv", index = False, quoting = 3)


# With the Convolutional NN we can see that despite its extremely expensive computation, it does an excellent job, even though it's underfitting. (Since Train/Validation loss >1). To fully convince myself of the fit, I made a submission to Kaggle and scored 98.11% placing me 56th on the leaderboard at the time of writing, which is pretty good.

## Classifying REAL Handwritten Digits

# Using my artistically inclined friend with a drawing tablet, I produced a small set of handwritten digits that I then converted to the same format as the MNIST data. It is worth noting that I did not apply the exact transformations of the data (20x20 centered on a 28x28 grid). These are the results.
# 
# 
# The first class is using a soft brush to produce a similar antialiasing effect to the original 

# In[130]:

from PIL import Image


# In[142]:

q = 28000
test_df = pd.read_csv('test.csv')
correct = 0
totalPred = 0
for i in range (0,10): 
    for j in range(1,4):
        im = Image.open(str(i)+'-'+str(j)+'.png', 'r')
        pix_val = list(im.getdata())
        test_df.loc[q] = pix_val
        result = cnn.predict(test_df.loc[q].values.reshape(-1, 1, 28, 28))
        if result[0] == i:
            correct = correct +1
        totalPred = totalPred + 1
print "Prediction accuracy on homemade digits :" + str(100*(float(correct)/float(totalPred)))


# Below is the score on a hard brush, which has no antialiasing

# In[143]:

q = 28000
test_df = pd.read_csv('test.csv')
correct = 0
totalPred = 0
for i in range (0,10): 
    for j in range(1,4):
        im = Image.open(str(i)+'-'+str(j)+'.png', 'r')
        pix_val = list(im.getdata())
        test_df.loc[q] = pix_val
        result = cnn.predict(test_df.loc[q].values.reshape(-1, 1, 28, 28))
        if result[0] == i:
            correct = correct +1
        totalPred = totalPred + 1
print "Prediction accuracy on homemade digits :" + str(100*(float(correct)/float(totalPred)))


# I'm not sure why the cnn performed so poorly. I think the reason could lie in the semi-complicated process that gets applied to the MNIST data. Using a soft brush is not equivalent to antialiasing hard brush images. Also the image's center of mass is calculated and centered on a grid. 
# The hand drawn images are just drawn in Photoshop on a 28x28 grid. There's also a few hard digits, like klaxon 0s and strange looking 1s.

## References

# 
# Useful code snippets: http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
# 
# 1-hidden-layer MLP - http://deeplearning.net/tutorial/mlp.html
# 
# Theano - http://deeplearning.net/software/theano/
# 
# Lasagne - http://lasagne.readthedocs.org/en/latest/
# 
# Nolearn - https://pythonhosted.org/nolearn/
# 

# In[ ]:



