# -*- coding: utf-8 -*-
"""
Created on Sun Dec 01 01:00:16 2019
@author: Genaro

This code was written and adapted in Python from the video tutorial series 
"Chapter 10 of The Nature of Code: Neural Networks" by "The Coding Train".


"""
import numpy as np

lr = 0.1 #Learning rate

############################ Mathematical functions ###########################
# Sigmoid Function (Activation Function)
def f(x):
    pr = 1/(1+1/(np.exp(x)))
    return pr 

# Column Vecteur x Vecteur = Square Matriz  
def productColRox(r,c):
    ones = np.zeros((c.size, r.size))+1
    pr = ones*r
    pr = pr.transpose()
    pr = pr*c
    return pr 
    
###############################################################################
class NeuralNetworks:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.weights_ih = 2*np.random.rand(hidden_nodes, input_nodes)-1        #random in [-1, 1]
        self.weights_ho = 2*np.random.rand(output_nodes, hidden_nodes)-1       #random in [-1, 1]

        self.bias_h = np.random.rand(hidden_nodes)
        self.bias_o = np.random.rand(output_nodes)

        self.learning_rate = lr

###############################################################################

    def feedforward(self, input_array):
        #Generating the hidden Outputs
        inputs = np.array(input_array)
        hidden = self.weights_ih.dot(inputs)
        hidden = hidden + self.bias_h
        #Activation function
        hidden = f(hidden)

        #Generating the outputs 
        outputs = np.array
        outputs = self.weights_ho.dot(hidden)
        outputs = outputs + self.bias_o
        #Activation function
        outputs = f(outputs)
        return outputs        

    def train(self, inputs, targets):
        #Generating the hidden Outputs
        inputs = np.array(inputs)
        hidden = self.weights_ih.dot(inputs)
        hidden = hidden + self.bias_h
        #Activation function
        hidden = f(hidden)
        #Generating the outputs 
        outputs = np.array
        outputs = self.weights_ho.dot(hidden)
        
        outputs = outputs + self.bias_o
#        #Activation function
        outputs = f(outputs)

        targets = np.array(targets)
        #Calculate the error
        #Error = Target - outputs
        output_errors = targets - outputs

################################# trainning ###################################

        ######## for the hidden errors
        w_ho_t = self.weights_ho.transpose()
        hidden_errors = w_ho_t.dot(output_errors)
        
        ######## Calculate the gradients        
        gradients = outputs*(1-outputs)
        gradients = gradients*output_errors
        gradients = gradients*self.learning_rate 

        ######## Calculate deltas
        weight_ho_deltas = productColRox(gradients,hidden)
        #Adjust the weights by deltas
        self.weights_ho = self.weights_ho + weight_ho_deltas
        #Adjust the bias by its deltas
        self.bias_o = self.bias_o + gradients        

        ######## for the hidden gradients        
        hidden_gradients = hidden*(1-hidden)
        hidden_gradients = hidden_gradients*hidden_errors
        hidden_gradients = hidden_gradients*self.learning_rate 

        ######## Calculate input->hidden deltas")        
        weight_ih_deltas = productColRox(hidden_gradients,inputs)
        #Adjust the weights by deltas
        self.weights_ih = self.weights_ih + weight_ih_deltas
        #Adjust the bias by its deltas
        self.bias_h = self.bias_h + hidden_gradients     