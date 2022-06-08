# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 01:08:33 2020

@author: mk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from logistic_regression import Logistic_Regression as logr

def evaluate_acc(actual, predicted):
    actual = np.squeeze(actual)
    correctly_predicted = np.sum(actual== predicted)
    return (correctly_predicted/len(actual))

def cross_validation(k_val, data, label, alpha=None, iters=None, stop_criteria=None):
    
    total_samples = len(data[:])
    k_size = np.round(total_samples / k_val, 0)
    k_split = np.round(k_size*2/3, 0) 

    k_fold_data = np.array_split(data,k_val,axis=0)
    k_fold_label = np.array_split(label,k_val,axis=0)
    
    acc_score = []

    for i_cnt in range(k_val):
        """Test data and label"""
        k_test_data = k_fold_data[i_cnt]
        k_test_label = k_fold_label[i_cnt]
        """Train data and label"""
        temp_data = np.delete(k_fold_data,i_cnt,axis=0)
        temp_label = np.delete(k_fold_label,i_cnt,axis=0)
        """concatenate the array"""
        k_train_data = temp_data[0]
        k_train_label = temp_label[0]
        for i_cnt2 in range(k_val-2):
            k_train_data = np.concatenate((k_train_data,temp_data[i_cnt2+1]))
            k_train_label = np.concatenate((k_train_label,temp_label[i_cnt2+1]))
             
        k_validate_data = k_test_data[:int(k_split)]
        k_validate_label = k_test_label[:int(k_split)]
        
        k_train_label = k_train_label.reshape(k_train_data.shape[0], -1)
        k_validate_label = k_validate_label.reshape(k_validate_data.shape[0], -1)
        k_test_label = k_test_label.reshape(k_test_data.shape[0], -1)
        
        lg = logr(alpha,iters)
        omega,num_iter,cost = lg.fit(k_train_data,k_train_label)
        y_pred = lg.predict(k_test_data[int(k_split):])
        acc = evaluate_acc(k_test_label[int(k_split):],y_pred)
        acc_score.append(acc)
    return acc_score

#Import Cleaned data
adult_test_df = pd.read_csv('adult_test_cleaned.csv', header = None)
adult_train_df = pd.read_csv('adult_cleaned.csv', header = None)

adult_test_df  = adult_test_df .sample(frac = 1)
adult_train_df = adult_train_df.sample(frac = 1)

test_rows, test_columns = adult_test_df.shape
train_rows, train_columns = adult_train_df.shape

train_data = []
test_data = []
train_label = []
test_label = []


train_label = adult_train_df[train_columns-1].to_numpy (dtype = 'int')
test_label = adult_test_df[test_columns-1].to_numpy (dtype = 'int')

del adult_train_df[train_columns-1]
del adult_test_df[test_columns-1]

"""Create a list of input data"""
for i in range (train_rows):
    train_data.append(adult_train_df.iloc[i].to_numpy())
    
for i in range (test_rows):
    test_data.append(adult_test_df.iloc[i].to_numpy())
    
""" Convert list to array"""
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)

test_label = np.asarray(test_label)
train_label = np.asarray(train_label)

train_label = train_label.reshape(train_data.shape[0], -1)
test_label = test_label.reshape(test_data.shape[0], -1)

print ("Start:\n\n")

w, h = 3, 10
acc_interm = [[0 for x in range(w)] for x in range(h)] 
iter_history = [[0 for x in range(w)] for y in range(h)] 

alpha = [0.00001, 0.00015, 0.0005]

for x in range (h):
    iterations = 1000 + 1000*x
    for y in range(w):
        acc_all = cross_validation(5,train_data,train_label,alpha[y],iterations)
        acc = sum(acc_all)
        acc/=5
        acc_interm[x][y] = acc
        iter_history[x][y] = iterations

acc_interm = np.asarray(acc_interm)
iter_history = np.asarray(iter_history)

for z in range (w):
    plt.figure(figsize=(8,8))
    plt.plot(iter_history[:,z],acc_interm[:,z])
    plt.savefig('Accuracy_vs_l_rate {:03d}.png'.format(z))