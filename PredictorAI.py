import sys
import numpy as np
import matplotlib
import Reader

weights = [.2, .8, -.5]
bias = 3
output = float()

for i in range(Reader.inputData.__len__()):
    output += (Reader.inputData[i])*(weights[i])
output += bias
print(Reader.inputData.__len__())
print(output)

#test