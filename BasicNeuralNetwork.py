import math
import sys
import numpy as np
import matplotlib
import nnfs
import cv2
from nnfs.datasets import spiral_data

nnfs.init()


X, y = spiral_data(100, 3)




# There is a weight for every connection to the neuron just like how each connection to the neuron gives one input
# In simpler terms there is at least one weight value for the size of each vector. So there can be a matrix of weights
    # while still only having a simple vector of inputs

# Shape is a description of the dimenensions of the matrix you have by its length
# A list of vectors is a matrix
# A tensor is a object that *can* be represented as an array

# BIGGER PICTURE:
# Weights are used within a neural network to define how much at a certain point the neural network is impacted (think how steep a slope is)
# Bias is used to control when a neuron will be activated (after what value on a function should you be using a neuron or stop using a neuron)



# Everything in a neural network has a forward pass and then a backward pass, the forward pass runs the network,
    # The backward pass detects how it should change the network on the next run
class Layers:
    def __init__(self, n_inputneurons, n_outputneurons):
        # The .1 normalizes the data to be between -1 and 1 (usually) that way outputs don't compound as they move through the network
        # Also randomizes the weights
        self.weights = .1 * np.random.randn(n_inputneurons, n_outputneurons)
        # .zeros just returns a array filled with 0s, unless your network is dead you don't want to change your biases
        self.biases = np.zeros((1, n_outputneurons))
    def forward(self, inputs):
        #Just doing the dot products discussed previously, do to how we've shaped the neurons in this class,
            # transposition is done inherently
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        #Is the gradient of the parameters for the function
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Is the gradient on the values the function produces
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    #Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    #Backward pass
    def backward(self, dvalues):

        self.dinputs = dvalues.copy()
        #Creates 0s in the inputs where the value is less than 0
        self.dinputs[self.inputs <= 0] = 0


# SOFTMAX ACTIVATION
class Activation:
    def forward(self, inputs):

        self.inputs = inputs

        values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = (values / np.sum(values, axis=1, keepdims=True))
        self.output = probabilities
    def backward(self, dvalues):

        # Creates an array with nothing in it, dvalues is passed in to define the shape of the array
        self.dinputs = np.empty_like(dvalues)
        # This next block of code will be going through each output and value and comparing how the weights and biases affected the output
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

            # Re shapes and flattens out a singular output to fit our next step
            single_output = single_output.reshape(-1, 1)

            # Creates a jacobian Matrix with the given output.
            # You may be asking what a jacobian matrix is, and you can think of it as a matrix (or an array) of values that \
            # are the partial derivatives of a previous matrix's values
            # The reason why the following line of code creates a Jacobian matrix is because of linear algebra that I haven't learned yet ¯\_(ツ)_/¯
            jacobianmatrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # The following line of code calculates an individual sample gradient and adds that to our list of sample gradients
            self.dinputs[index] = np.dot(jacobianmatrix, single_dvalues)
            # Its just doing a dot product of the partial derivatives of the matrix (the single gradient value) and the output value


#Calculates loss
class Loss:
    def calculate(self, output, intendedValues):
        sample_losses = self.forward(output, intendedValues)
        dataloss = np.mean(sample_losses)
        return dataloss

class Loss_Categorical_Cross_Entropy(Loss):
    # Gets the Loss through categorical cross entropy when passed in a prediction and a true value
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # This next part handles if a user passes in one hot values or normal values (user is me)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # 2 means its one hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        # Returns all the likelihoods of loss throughout the matrix
        loss_likelihoods = -np.log(correct_confidences)
        return loss_likelihoods

        # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


def getAccuracy(softMaxOutputs, class_targets):
    predictions = np.argmax(softMaxOutputs, axis=1)
    accuracy = np.mean(predictions == class_targets)
    return accuracy



# The next bit of code is the whole purpose of the project, it is the way the neural network is learning
# To understand what SGD is doing I recommend you watch these videos:
    # https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3
    # https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=4
class SGD_Optimizer():

    # Learning rate is how fast the program is trying to learn
    # From what I can tell:
    # Momentum is how much the weights and biases of certain areas have been changed already (think the direction and speed the gradient is changing already)
    def __init__(self, learning_rate=1, decay=0, momentum=0):
        # A Learning rate of 1 just means the network won't try to change how fast its learning starting out
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        # There is no momentum yet on definition
        self.momentum = momentum
    # To be called once before any further actions with the optimizer is taken
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + (self.decay * self.iterations) ))

    # Updates the parameters, the whole point of the SGD process. We will now adjust the weights and biases
    def update_params(self, layer):
        # Checks if we have momentum
        if self.momentum:

            # Will check if the layer already has momentum arrays, if not then it will create momentum arrays for the layer all filled with 0s
            if not hasattr(layer, 'weights_momentums'):

                #Creates momentum arrays for both weights and biases
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            #Updates the weights and biases based off of momentum, learning rate, and the back propogation values
            weight_updates = self.current_learning_rate * layer.dweights * self.momentum * layer.weight_momentums
            bias_updates = self.current_learning_rate * layer.dbiases * self.momentum * layer.bias_momentums
            # Updates the layers momentum
            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        else:

            # Will perform normal SGD without momentum
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Now updates the layers using whatever method was done for SGD
        layer.weights += weight_updates
        layer.biases +=  bias_updates

    # Call once after every iteration
    def post_update_params(self):
            self.iterations += 1




# Just learned this, output layer should always be softmax but the hidden layer should(?) be reLU



# We will now be actually using the neural network off of all the framework we have built thus far
# ================== Definition ==================
# Layer here has to be 2 as this is with a spiral which means there is only two inputs of values, x and y
# Creates two layers, notice how the output of the hidden layer has to match the input of the output layer
hiddenLayer = Layers(2, 200)
outputLayer = Layers(200,3)
learning_rate = 2
hiddenActivation = Activation_ReLU()
Activation2 = Activation()
loss_function = Loss_Categorical_Cross_Entropy()
optimizer = SGD_Optimizer(learning_rate, decay=(3.951e-4))


# ================== Running the network ==================
for epoch in range(10000):

    # Performs a forward pass of the network, a.k.a runs the neural network
    hiddenLayer.forward(X)
    hiddenActivation.forward(hiddenLayer.output)

    outputLayer.forward(hiddenActivation.output)
    Activation2.forward(outputLayer.output)

    loss = loss_function.calculate(Activation2.output, y)
    accuracy = getAccuracy(Activation2.output, y)

    # checks if this is the last run of the dataset
    if not (epoch % 100):
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    #Runs the backward pass of the neural network, basically running the network back and looking for how each weight and bias is affecting things
    loss_function.backward(Activation2.output, y)
    outputLayer.backward(loss_function.dinputs)
    hiddenActivation.backward(outputLayer.dinputs)
    hiddenLayer.backward(hiddenActivation.dinputs)

    #Updates the weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(hiddenLayer)
    optimizer.update_params(outputLayer)
    optimizer.post_update_params()

# Obtained more accurate information