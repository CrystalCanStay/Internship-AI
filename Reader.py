import numpy as np


# This is going to be the list we later fill up by reading from a file
inputBatch = [[]]
inputData = []
# placeholder for doing basic neuron stuff
inputBatch.remove(inputData)
for k in range(3):
    for i in range(1):
        inputData.insert(i, np.random.randn())
    inputBatch.insert(k, inputData)