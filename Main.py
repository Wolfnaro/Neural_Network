# -*- coding: utf-8 -*-
"""
Created on Sun Dec 01 01:20:08 2019
@author: G-naro

This code was written and adapted in Python from the video tutorial series 
"Chapter 10 of The Nature of Code: Neural Networks" by "The Coding Train".
"""

import NeuralNetworks 
import time

###################### Tranning data for exemple XOR ##########################
inp = [[0,1], [1,0], [0,0], [1,1]]          # input
tar = [[1], [1], [0], [0]]                  # output

##################### Construction of the neural network ######################
n_input = 2
n_hidden = 128
n_output = 1 

nn = NeuralNetworks.NeuralNetworks(n_input, n_hidden, n_output)

################################ Trainning ####################################
start_time = time.time()
for ite in range (1000):
    for i in range(len(inp)):
        nn.train(inp[i], tar[i])

execution_time = time.time() - start_time
print("tiempo de aprendizaje"+str(execution_time))

########################## Using the Neural Network ########################### 
for i in range(len(inp)):
    print(nn.feedforward(inp[i]))

prediction_time = time.time() - start_time - execution_time
print("tiempo de ejecucion"+str(prediction_time))
