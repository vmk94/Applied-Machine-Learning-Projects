# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:02:07 2020

@author: mk
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10**6)

class Logistic_Regression:
    def __init__ (self,lrate=None,threshold=None):
        self.__lrng_rate = lrate
        self.__threshold =  threshold
        self.__final_omega = np.array([])
        
    """Compute normalized value for data"""
    def normalize (self, data):
        data_min = np.min(data, axis = 0)
        data_max = np.max(data, axis = 0)
        data_range = data_max - data_min
        norm_data = np.where(data_range == 0, 1, (1 - (data_max - data)/data_range))
        #norm_data = 1 - (data_max - data)/data_range
        return norm_data
    
    """Sigmoid function"""
    def sigmoid_func(self, z): 
        return 1.0/(1.0 + np.exp(-z))

    def cost_func(self, data, y_actual, omega):
        z = np.dot(data,omega.T)
        val1 = np.multiply (y_actual, np.log1p(np.exp(-z)))
        val2 = np.multiply((1-y_actual), np.log1p(np.exp(z)))
        cost = val1 + val2
        return np.mean(cost)
    
    def grad_desc(self,train_data, y_train_actual, val_data, y_val_actual, omega):
        if (val_data != 0 and y_val_actual != 0):
            validate_data = val_data
            y_validate_actual = y_val_actual
        else:
            validate_data = train_data
            y_validate_actual = y_train_actual

        cost = self.cost_func(validate_data, y_validate_actual, omega)
        diff_cost = 1
        num_iter = 1
        cost_history = []
        cost_history.append(cost)

        while (num_iter < self.__threshold):
        #while (num_iter < self.__threshold and diff_cost > 0):
            old_cost = cost
            z = np.dot(train_data,omega.T)
            predict_diff = self.sigmoid_func(z)  - (y_train_actual)
            omega = omega - (self.__lrng_rate*(np.dot(predict_diff.T,train_data))/len(train_data[0]))
            cost =  self.cost_func(validate_data, y_validate_actual, omega)
            diff_cost = old_cost - cost
            num_iter+=1
            cost_history.append(cost)

        return omega, num_iter,cost_history


    def predicted_values(self, data, omega): 
        z = np.dot(data,omega.T)
        pred_prob = self.sigmoid_func(z)
        pred_value = np.where(pred_prob >= .5, 1, 0) 
        return np.squeeze(pred_value)
    
    """ Public Methods"""
    def fit(self,train_data, y_train_actual, val_data=0, y_val_actual=0):
        """ Normalize input"""
        train_data = self.normalize(train_data)
        """ Stack with ones. Make first column ones"""
        train_data = np.hstack((np.matrix(np.ones(train_data.shape[0])).T,train_data))
        """If validate is available"""
        if (val_data != 0 and y_val_actual != 0):
            val_data = self.normalize(val_data)
            val_data = np.hstack((np.matrix(np.ones(val_data.shape[0])).T,val_data))
            
        """Initial omega Values"""
        omega = np.matrix(np.zeros(train_data.shape[1]))
        """Gradient descend"""
        self.final_omega, num_iter, cost_history = self.grad_desc(train_data, y_train_actual, val_data, y_val_actual, omega)
        """ Return omega and number of iterations"""
        return self.final_omega, num_iter, cost_history
    
    def predict(self,data):
        """ Normalize input"""
        data = self.normalize(data)
        """ Stack with ones. Make first column ones"""
        data = np.hstack((np.matrix(np.ones(data.shape[0])).T,data))
        """Predicted values"""
        y_predicted = self.predicted_values(data,self.final_omega)
        
        return y_predicted
