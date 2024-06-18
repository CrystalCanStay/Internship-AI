import sys
import numpy as np
import matplotlib
import Reader

# There is a weight for every connection to the neuron just like how each connection to the neuron gives one input
# In simpler terms there is at least one weight value for the size of each vector. So there can be a matrix of weights
    # while still only having a simple vector of inputs
weights = [[.2, .8, -.5],
           [1, .8, -.2],
           [2, .8, -.3]]
biases = [3, 4, 5]

# Not needed as you just can use numpy, this really is just doing a dot product
#for i in range(Reader.inputData.__len__()):
#   output += (Reader.inputData[i])*(weights[i])
#output += bias

output = np.dot(Reader.inputBatch, np.array(weights).T) + biases
print(output)

# Shape is a description of how many lists you have in a list and then the length of those lists
# Arrays and lists in python have to be homologous to determine shape, so each dimension has the same size
# A list of vectors is a matrix
# A tensor is a object that *can* be represented as an array
# Once you get a list of lists you have a matrix of vectors