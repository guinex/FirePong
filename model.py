# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:26:58 2019

@author: Guinex
"""
import numpy as np
from random import *
class model:
    def __init__(self, weights):
        self.input_size = 4
        self.W1, self.W2, self.W3 = weights
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        ans = 1/(1 + np.exp(-z))
        return ans
    
    def forward_propagate(self, X):
        Z1 = np.dot(self.W1, X.T) 
        A1 = self.relu(Z1) # relu for hidden
        A1 = np.concatenate((np.ones((1,A1.shape[1])), A1))
        Z2 = np.dot(self.W2, A1)
        A2 = self.relu(Z2) # relu for another hidden
        A2 = np.concatenate((np.ones((1,A2.shape[1])), A2))
        Z3 = np.dot(self.W3, A2)
        A3 = self.sigmoid(Z3) # sigmoid for output
        return A3

    def predict(self, X):
        result = self.forward_propagate(X)
        print(result)
        operators = ['up', 'down']
        if result == float('nan'):
            return choice(operators)
        if result >0.45:
            return "up"
        elif result < 0.55:
            return "down"
