# Team-Neural-Networks

---

Authors: Mitch & Matt

The purpose of this repo is to provide an introduction to the theory of ANNs (Artifical Neural Networks) and demonstrate their applications on data. 

###Tentative presentation plans:

**1. Introduction:**    
  >- What are Artifical Neural Networks?
  >- (A *very brief history* of ANNs ) - optional, but good to know regardless


*The next few topics are taken from Wikipedia, so* **TODO:** *Find good resources!*

**Network function:**  
  >- Outline the typical ANN parameters - interconnection pattern between different layers of neruons, learning process for updating weights of the interconnections, and the activation function that converts a neuron's weighted input to its output activation

**Learning:**  
  >- Cost functions / Choosing a Cost function
  >- Paradigms : In both of our datasets, we will likely rely on supervised learning techniques 
  >- Mean squared error

###FeedForward Neural Network

**Feed Forward:**  
  >- Connections between units do not form a directed cycle. Different from recurrent neural networks (RNNs).


**Single Layer Perceptron:**  
  >- Linear Classifier
  >- Uses Linear predictor function combining set of weights with the feature vector
  >- Online learning - process data in piece by piece fashion (ie in order that input is given, not having entire input avail from start)

**Multilayer Perceptron:**  
  >- Multi-layers of computational units conected in feed-forward manner
  >- Each neuron in a layer has direct connections to neurons in subsequent layer
  >- Sigmoid Function  (Common activation function)
  >- **Back-propagation**
  >- >- Output values compared with correct answers to find value of some error function, and then the error is fed back through the network, and the weights of each connection are adjusted in order to reduce the value of the error function. Cycles of this eventually converge to a state where the error of the calculations is small. 
  >- Weight adjusting - general method for non-linear optimization is Gradient Descent, where the derivative of the error function with respect to the network weights is computed, and weights are changed such that error decreases.
  >- Back-propagation can only be applied on ANNs with differentiable activation functions because of this

**Misc:** 
  >- Additional techniques
  >- Danger of overfitting the training data and not capturing true model of the dat
  >- To avoid overfitting one heuristic called early stopping can be used 
  >- Speed of convergence in back-propagation algorithms
  >- Possibility of ending up in a local minimum of the error function

**Convolutional Neural Network:**  
  >- Feed-forward
  >- Individual neurons tiled in an overlapping manner of regions in visual field
  >- Inspired by biological processes / are variations of multilayer perceptron
  >- During backpropagation momentum and weight decay are introduced, to avoid much oscillation during stochastic gradient descent
  >- **CNN Layers:**  
  >- Convolutional Layer
  >- >- Parameters of each convolutional kernal are trained by backpropagation algorithm
  >- >- many convolution kernels in each layer 
  >- >- each kernel is replicated over entire image with same parameters
  >- >- Function of the convolution operators is: Extract different features of the input
  >- >- First convlution layers will obtain low-level features such as edges, line curves, and the more layers there are, the more higher-level features it will get
  >- ReLU Layer
  >- >- Rectified Linear Units
  >- >- Layer of neurons using non-saturating activation function f(x) = max(0,x) thereby increasing nonlinear properties of the decision function without affecting receptive fields of the convolution layer
  >- >- Other functions are used to increase nonlinearity such as hyperbolic tangent - f(x) - tanh(x), f(x) = |tanh(x)|, and the sigmoid function f(x) = (1+e^(-x))^(-1).
  >- >- Adv. of ReLU is that compared to these functions, neural network trains several times faster
  >- Dropout "layer"
  >- >- Fully connected layer occupies most of the parameters and is prone to overfitting
  >- >- Dropout method introduced to prevent overfitting
  >- >- Also improves speed of training
  >- >- Dropout is performed randomly - in input layer, probability of dropping a neuron is between 0.5, 1 and in hidden layers, a probability of 0.5 is used. Neurons that get dropped do not contribute to the forward pass and back propagation. 
  >- Loss layer
  >- >- Loss functions could be used for different tasks
  >- Softmax loss
  >- >- Predicting single class of K mutually exclusive classes
  >- Sigmoid cross-entropy loss
  >- >- predicting K independent probability values in [0,1]
  >- Euclidean loss
  >- >- Regressing to real-valued labels [-inf,inf]

###Problems in Neural Networks  

**Association Rule Learning**  
  >- Find interesting rules and relations between variables in large databases

**Anomaly Detection**  
  >- Identification of items that do not conform to expected pattern 
  >- E.g. Intrusion detection, Fraud detection

**Grammar Induction**  
  >- Grammatical inference aka syntactic pattern recognition
  >- Learning a formal grammar from a set of observations

**Classification (What we are doing)**  
  >- Handwritten numbers
  >- Facial Keypoints

###Applications

**Data processing**

**EDA**

**How we chose parameters**

**How many learning cycles did it take to reach a low error on the data (graphs)**

**Performance comparison of differnet neural networks we used (if we used more than one)**

**How do we compare against others who did this problem on Kaggle (if we have time to put together proper submission with test data)**


###Fine-tuning  
>- Dealing with small amount of training data
>- ???


###Tools

**Python:**  
  >- Theano  
  >- Numpy  
  >- ?
  
###Remarks

  >- 
  >- In conclusion could briefly mention [Recurrent Neural Networks](http://en.wikipedia.org/wiki/Recurrent_neural_network#Fully_recurrent_network)
  >-This is just a very rough, unfinished outline of what we may choose to include in our Neural Network presentation!
  >- TO BE UPDATED!

