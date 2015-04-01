# Team-Neural-Networks
---

####Authors: Mitch & Matt

#1. Introduction to Artificial Neural Networks (ANNs)

> The question 'What is a neural network?' is ill-posed.  
>                                                             - Pinkus (1999)</quote>
 [[3]](#[3])  

##1.0. Biological Inspiration 

  >- ANNs are inspired from biological neural networks. [[1]](#[1])

In biology, a neuron is a cell that is part of the nervous system. The neuron has three parts: First, a cell body containing it's nucleus and other organelles. Second, dendrites - short branching fibers which **receive** signals. Third, an axon - a long projecting branch which **sends** signals. The axon can branch into many other cells, where the output signal it sends can be used as the input signals for other cells.[[2]](#[2])

##1.1. Different definitions of ANNs...

###DARPA Neural Network Study (1988, AFCEA International Press, p. 60):  [[4]](#[4])  
  >-... a neural network is a system composed of many simple processing elements operating in parallel whose function is determined by network structure, connection strengths, and the processing performed at computing elements or nodes.  

###Haykin (1994), p. 2:  [[5]](#[5])  
  >-A neural network is a massively parallel distributed processor that has a natural propensity for storing experiential knowledge and making it available for use. It resembles the brain in two respects: 
  Knowledge is acquired by the network through a learning process. 
  Interneuron connection strengths known as synaptic weights are used to store the knowledge.

###According to Nigrin (1993), p. 11:  [[6]](#[6])  
  >-A neural network is a circuit composed of a very large number of simple processing elements that are neurally based. Each element operates only on local information. Furthermore each element operates asynchronously; thus there is no overall system clock.

###According to Zurada (1992), p. xv:  [[7]](#[7])  
  >-Artificial neural systems, or neural networks, are physical cellular systems which can acquire, store, and utilize experiential knowledge.

**For this presentation, we are going to consider one class of ANNs - Feed-Forward neural networks.



##1.2. Applications

##1.2.1. Classification

  ANNs can be used for image classification problems, such as recognizing handwritten characters [[8]](#[8]), labeling images into a variety of categories (such as types of animals, objects, etc.)[[9]](#[9]),






#2.Feed-Forward Neural Networks


##2.0. Model of a neuron/Activation functions
##2.0.1 Neurons
We have neural networks, we of coruse need to talk about what a "Neuron" means to us. The Neuron has 3 main parts:
  >- Synapses or "Links", each of which has different weights. If we have an input signal coming in to a neuron, we may wish to weight the input features differrently. 
  >- Adder - Used for summing the input signals which have been weighted. 
  >- Activation function - is used to limit the output of a neuron. Essentially what the activation function does is require that a sufficiently strong output is reached. This is very similar to the neurons in our brains, which fire or activate in response to a change in electrical charge.

##2.0.2 Activation function 
There are a few different choices for the activation function
  >- Threshold function - Is basically 0 or 1 depending on if the input is positive or negative. - Not particularly useful
  >- Piecewise linear - This is simply an approximation of non-linear behavior. Has sharp corners
  >- Sigmoid function - The typical choice for activation. It is s-shaped and strictly increasing.




##2.1. Single-layer

A single layer perceptron is a binary classifier that maps an input to a binary value based on the weight that the features have. A single layer perceptron needs the inputs to be linearly separable, or it will not be able to classify the outputs correctly. Linear separability is more or less the existence of a hyperplane "decision boundary" in which the two classifications of data can be split apart.

It is actually possible to prove that if the training set is linearly separable, then the perceptron will converge.

The single layer perceptron can collapse to both logistic regression or support vector machines with particular choices of the weights.

##2.2. Multi-layer perceptron.
 >- Multi-layers of computational units conected in feed-forward manner
  >- Each neuron in a layer has direct connections to neurons in subsequent layer
  >- We can also use dropout layers to Dropout is a technique that addresses both these issues. It prevents overfitting and provides a way of approximately combining exponentially many different neural network architectures efficiently. The term “dropout” refers to dropping out units (hidden and visible) in a neural network. By dropping a unit out, we mean temporarily removing it from the network, along with all its incoming and outgoing connections [[10]](#[10])
  >- Sigmoid Function  (Common activation function)
  >- **Back-propagation**
  >- >- Output values compared with correct answers to find value of some error function, and then the error is fed back through the network, and the weights of each connection are adjusted in order to reduce the value of the error function. Cycles of this eventually converge to a state where the error of the calculations is small. 
  >- Weight adjusting - general method for non-linear optimization is Gradient Descent, where the derivative of the error function with respect to the network weights is computed, and weights are changed such that error decreases.
  >- Back-propagation can only be applied on ANNs with differentiable activation functions because of this

##2.3. Convolutional
http://media.wiley.com/product_data/excerpt/19/04713491/0471349119.pdf

http://media.wiley.com/product_data/excerpt/19/04713491/0471349119.pdf





The neurons perform basically the following function: all the inputs to the cell, which may vary by the strength of the connection or the frequency of the incoming signal, are summed up. The input sum is processed by a threshold function and produces an output signal. The processing time of about 1ms per cycle and transmission speed of the neurons of about 0.6 to 120 {ms} are comparingly slow to a modern computer [zell94, p24,] , [barr88, p35,].

The brain works in both a parallel and serial way. The parallel and serial nature of the brain is readily apparent from the physical anatomy of the nervous system. That there is serial and parallel processing involved can be easily seen from the time needed to perform tasks. For example a human can recognize the picture of another person in about 100 ms. Given the processing time of 1 ms for an individual neuron this implies that a certain number of neurons, but less than 100, are involved in serial; whereas the complexity of the task is evidence for a parallel processing, because a difficult recognition task can not be performed by such a small number of neurons, example taken from [zell94, p24,]. This phenomenon is known as the 100-step-rule.


Biological neural systems usually have a very high fault tolerance. Experiments with people with brain injuries have shown that damage of neurons up to a certain level does not necessarily influence the performance of the system, though tasks such as writing or speaking may have to be learned again. This can be regarded as re-training the network.

In the following work no particular brain part or function will be modeled. Rather the fundamental brain characteristics of parallelism and fault tolerance will be applied.

  >- Family of statistical learning algorithms inspired by biological neural networks.


  In machine learning and cognitive science, artificial neural networks (ANNs) are a family of statistical learning algorithms inspired by biological neural networks (the central nervous systems of animals, in particular the brain) and are used to estimate or approximate functions that can depend on a large number of inputs and are generally unknown. Artificial neural networks are generally presented as systems of interconnected "neurons" which can compute values from inputs, and are capable of machine learning as well as pattern recognition thanks to their adaptive nature.






*The next few topics are taken from Wikipedia, so* **TODO:** *Find good resources!*

**Network function:**  
  >- Outline the typical ANN parameters - interconnection pattern between different layers of neruons, learning process for updating weights of the interconnections, and the activation function that converts a neuron's weighted input to its output activation
  >- Training Epoch - What is it, what does it mean vs iteration - An epoch is a measure of the number of times all of the training vectors are used once to update the weights.
**Learning:**  
  >- Cost functions / Choosing a Cost function
  >- Paradigms : In both of our datasets, we will likely rely on supervised learning techniques 
  >- Mean squared error

###FeedForward Neural Network

**Misc:** 
  >- Additional techniques
  >- Danger of overfitting the training data and not capturing true model of the data 
  >- To avoid overfitting one heuristic called early stopping can be used 
  >- dropout layer
  >- Speed of convergence in back-propagation algorithms
  >- Possibility of ending up in a local minimum of the error function

**Convolutional Neural Network:**  
  >- Feed-forward
  >- Individual neurons tiled in an overlapping manner of regions in visual field
  >- Inspired by biological processes / are variations of multilayer perceptron
  >- During backpropagation momentum and weight decay are introduced, to avoid much oscillation during stochastic gradient descent
  >- **CNN Layers:**  
  >- Convolutional Layer
  >- >- Parameters of each convolutional kernel are trained by backpropagation algorithm
  >- >- many convolution kernels in each layer 
  >- >- each kernel is replicated over entire image with same parameters
  >- >- Function of the convolution operators is: Extract different features of the input
  >- >- First convolution layers will obtain low-level features such as edges, line curves, and the more layers there are, the more higher-level features it will get
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

###Problems in Neural Networks  - MAKE THESE BRIEF

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
     >- lol.

**How we chose parameters**

**How many learning cycles did it take to reach a low error on the data (graphs)**

**Performance comparison of different neural networks we used (if we used more than one)**

**How do we compare against others who did this problem on Kaggle (if we have time to put together proper submission with test data)**


###Fine-tuning  
>- Dealing with small amount of training data
>- ???


###Tools

**Python:**  
  >- Theano  
  >- Numpy  
  >- Lasagne
  >- Nolearn
  
###Remarks

  >- 
  >- In conclusion could briefly mention [Recurrent Neural Networks](http://en.wikipedia.org/wiki/Recurrent_neural_network#Fully_recurrent_network)


###Sources

##### <a name="[1]"/> [1]: http://www.teco.edu/~albrecht/neuro/html/node7.html
##### <a name="[2]"/> [2]: http://en.wikiversity.org/wiki/Fundamentals_of_Neuroscience/Neural_Cells
##### <a name="[3]"/> [3]: ftp://ftp.sas.com/pub/neural/FAQ.html#A2

##### <a name="[4]"/> [4]:Pinkus, A. (1999), "Approximation theory of the MLP model in neural networks," Acta Numerica, 8, 143-196.

##### <a name="[5]"/> [5]:Haykin, S. (1999), Neural Networks: A Comprehensive Foundation, NY: Macmillan.

##### <a name="[6]"/> [6]:Nigrin, A. (1993), Neural Networks for Pattern Recognition, Cambridge, MA: The MIT Press.

##### <a name="[7]"/> [7]:Zurada, J.M. (1992), Introduction To Artificial Neural Systems, Boston: PWS Publishing Company.

##### <a name="[8]"/> [8]:http://cs.stanford.edu/people/eroberts/courses/soco/projects/2000-01/neural-networks/Applications/character.html

##### <a name="[9]"/> [9]:http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
##### <a name ="[10]"/> [10]: 
http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
