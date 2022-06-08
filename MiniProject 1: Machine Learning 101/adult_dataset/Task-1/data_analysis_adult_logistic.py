# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 01:54:03 2020

@author: mk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns; sns.set()

#Import adult.data file to string array---------------------------------------------

Data2_test = np.array(np.genfromtxt('adult_test.csv',delimiter=',', dtype="|U50"))
Data2 = np.array(np.genfromtxt('adult.csv',delimiter=',', dtype="|U50"))

#Reading Dataset number of rows and columns-----------------------------------------
rows, columns = Data2.shape
R = np.arange(rows)
C = np.arange(columns)

rows_test, columns_test = Data2_test.shape
R_test = np.arange(rows_test)
C_test = np.arange(columns_test)

#Target class binary encoding-------------------------------------------
Class_Binary = np.zeros((rows,1))
Target = np.unique(Data2[:,-1])
Class_Binary[np.where(Data2[:,-1] == Target[0])] = 1

Class_Binary_test = np.zeros((rows_test,1))
Target_test = np.unique(Data2_test[:,-1])
Class_Binary_test[np.where(Data2_test[:,-1] == Target_test[0])] = 1

#Dataset split from target---------------------------------------------------------------
Data2_test = Data2_test[:,:-1]
Data2 = Data2[:,:-1]

rows, columns = Data2.shape
R = np.arange(rows)
C = np.arange(columns)

rows_test, columns_test = Data2_test.shape
R_test = np.arange(rows_test)
C_test = np.arange(columns_test)

#Identification of numerical indices and categorical indices------------------------------
Categorical_id = np.zeros((1,1))
Index_numerical = np.array([])
Index_categorical = np.array([])
for x in C:    
    try:
        Categorical_id = Data2[0,x].astype('float')
        Index_numerical = np.concatenate((Index_numerical, x),axis= None).astype('int')
    except ValueError:
        Index_categorical = np.concatenate((Index_categorical, x),axis= None).astype('int')



#Filling missing values---------------------------------------------------
Index_malformed = np.array([0])
for x in Index_categorical:
    Subcategory = np.unique(Data2[:,x])
    Frequency = np.zeros((Subcategory.size))
    for y in np.arange(Subcategory.size):
        Frequency[y] = np.array(np.where(Data2[:,x] == Subcategory[y])).size
    Mode = Subcategory[(np.where(Frequency == np.amax(Frequency)))]
    Data2[np.where(Data2[:,x]== " ?"),x] = Mode

#Eliminating malformed features-------------------------------------------
    if (np.array(np.where(Data2[:,x]== Mode)).size*100/rows) > 80:
        Index_malformed = np.array([np.concatenate((Index_malformed, np.where(Index_categorical == x)), axis = None)]).astype('int')
Index_malformed = np.sort(np.concatenate((Index_malformed, 1),axis= None))
Index_categorical_cleaned = np.delete(Index_categorical, Index_malformed[1:])

Data2 = np.delete(Data2, Index_malformed,1)
Data2_test = np.delete(Data2_test, Index_malformed,1)
Data2 = Data2.tolist()
Data2 = np.concatenate(Data2,Data2_test, axis=None)
R,C = Data2.shape