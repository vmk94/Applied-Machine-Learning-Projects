#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Algorithm - functions definition

# # Dataset 1

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10**6)
from statistics import mode as modevalue

class Cross_Validation(object):
    def k_fold(self,DataSet, Y, c):
        K = 5
        DataSet = np.hstack((DataSet, Y))
        split = np.ceil(np.shape(DataSet)[0]/K).astype('int')
        split_array = np.array([split, split*2, split*3, split*4])
        DS1, DS2, DS3, DS4, DS5 = np.vsplit(DataSet, split_array)
        
        if c == 1:
            Train_Data = np.vstack((DS4, DS3, DS2, DS1))
            Y_train = Train_Data[:,-1].astype('float')
            Y_train.shape = (Train_Data.shape[0],1)
            Train_Data = Train_Data[:,:-1]

            Test_Data = np.array(DS5)
            Y_test = Test_Data[:,-1].astype('float')
            Y_test.shape = (Test_Data.shape[0],1)
            Test_Data = Test_Data[:,:-1]
            return Train_Data, Test_Data, Y_train, Y_test

        
        elif c == 2:
            Train_Data = np.vstack((DS3, DS2, DS1, DS5))
            Y_train = Train_Data[:,-1].astype('float')
            Y_train.shape = (Train_Data.shape[0],1)
            Train_Data = Train_Data[:,:-1]

            Test_Data = np.array(DS4)
            Y_test = Test_Data[:,-1].astype('float')
            Y_test.shape = (Test_Data.shape[0],1)
            Test_Data = Test_Data[:,:-1]
            return Train_Data, Test_Data, Y_train, Y_test

        
        elif c == 3:
            Train_Data = np.vstack((DS2, DS1, DS5, DS4))
            Y_train = Train_Data[:,-1].astype('float')
            Y_train.shape = (Train_Data.shape[0],1)
            Train_Data = Train_Data[:,:-1]

            Test_Data = np.array(DS3)
            Y_test = Test_Data[:,-1].astype('float')
            Y_test.shape = (Test_Data.shape[0],1)
            Test_Data = Test_Data[:,:-1]
            return Train_Data, Test_Data, Y_train, Y_test

        
        elif c == 4:
            Train_Data = np.vstack((DS1, DS5, DS4, DS3))
            Y_train = Train_Data[:,-1].astype('float')
            Y_train.shape = (Train_Data.shape[0],1)
            Train_Data = Train_Data[:,:-1]

            Test_Data = np.array(DS2)
            Y_test = Test_Data[:,-1].astype('float')
            Y_test.shape = (Test_Data.shape[0],1)
            Test_Data = Test_Data[:,:-1]
            return Train_Data, Test_Data, Y_train, Y_test

        
        elif c == 5:
            Train_Data = np.vstack((DS5, DS4, DS3, DS2))
            Y_train = Train_Data[:,-1].astype('float')
            Y_train.shape = (Train_Data.shape[0],1)
            Train_Data = Train_Data[:,:-1]

            Test_Data = np.array(DS1)
            Y_test = Test_Data[:,-1].astype('float')
            Y_test.shape = (Test_Data.shape[0],1)
            Test_Data = Test_Data[:,:-1]
            return Train_Data, Test_Data, Y_train, Y_test

class Naive_Bayes(object):
    def __init__(self):
        self.__mean = np.array([])
        self.___std = np.array([])
        self.__Cond_matrix = np.array([])
        self.__py_1 = np.array([])
        self.__py_0 = np.array([])
        
    def Num_Cat(self,DataSet):
        Categorical_id = np.zeros((1,1))
        Index_numerical = np.array([])
        Index_categorical = np.array([])
        for x in np.arange(np.shape(DataSet)[1]):    
            try:
                Categorical_id = DataSet[0,x].astype('float')
                Index_numerical = np.concatenate((Index_numerical, x),axis= None).astype('int')
            except ValueError:
                Index_categorical = np.concatenate((Index_categorical, x),axis= None).astype('int')
        return Index_numerical, Index_categorical
    
    def Prior_Prob(self,Class_Binary):
        rows = np.shape(Class_Binary)[0]
        Index_1 = np.array(np.where(Class_Binary==1))[0,:]
        Npy_1 = Index_1.shape[0]
        py_1 = Npy_1/rows
        
        Index_0 = np.array(np.where(Class_Binary==0))[0,:]
        Npy_0 = Index_0.shape[0]
        py_0 = Npy_0/rows
        
        self.py_1 = py_1
        self.py_0 = py_0
        return Npy_1, Npy_0, self.py_1, self.py_0, Index_1, Index_0
    
    def Cond_Cat(self, DataSet_categorical, Npy_1, Npy_0, Index_1, Index_0):
        Cond_matrix = np.array(np.zeros((1,3)))
        for x in np.arange(np.shape(DataSet_categorical)[1]):
            Subcategory = np.unique(DataSet_categorical[:,x])
            dummy_matrix = np.zeros((Subcategory.size, 3)).astype('unicode')
            dummy_matrix[:,0] = Subcategory
            for y in np.arange(Subcategory.size):
                Nxd = np.array(np.where(DataSet_categorical[:,x] == Subcategory[y]))
                dummy_matrix[y,1] = (np.intersect1d(Nxd,Index_1).size+1)/(Npy_1+2)
                dummy_matrix[y,2] = (np.intersect1d(Nxd,Index_0).size+1)/(Npy_0+2)    
            Cond_matrix = np.vstack((Cond_matrix,dummy_matrix))
        Cond_matrix = Cond_matrix[1:,:]
        self.Cond_matrix = Cond_matrix
        return self.Cond_matrix
    
    def mean_std(self, DataSet_numerical, Index_1, Index_0):
        mean = np.zeros((2,np.shape(DataSet_numerical)[1]))
        std = np.array(mean)
        mean[0,:] = np.mean(DataSet_numerical[Index_1,:], axis = 0)
        std[0,:] = np.std(DataSet_numerical[Index_1,:], axis = 0)
        mean[1,:] = np.mean(DataSet_numerical[Index_0,:], axis = 0)
        std[1,:] = np.std(DataSet_numerical[Index_0,:], axis = 0)
        self.mean = mean
        self.std = std
        return self.mean, self.std
    
    
    def Gauss_NB(self, Test_Data_num, mean, std, y):
        #Generate conditional probability matrix for numerical features(row 1 = yes, row 2 = no)
        Pcond_num = np.zeros((2,np.shape(Test_Data_num)[1]))
        #If std. deviation = 0, we define that the conditional probability is equal to 1 for both classes
        for x in np.arange(np.shape(Test_Data_num)[1]):
            if std[0,x] == 0 or std[1,x] ==0:
                Pcond_num[:,x] = 1
            else:   
                Pcond_num[0,x] =(np.exp((-0.5*(Test_Data_num[y,x]-mean[0,x])**2)/(std[0,x]**2)))/(np.sqrt(2*np.pi*(std[0,x]**2)))
                Pcond_num[1,x] =(np.exp((-0.5*(Test_Data_num[y,x]-mean[1,x])**2)/(std[1,x]**2)))/(np.sqrt(2*np.pi*(std[1,x]**2)))
        return Pcond_num
    
    
    def Bernoulli_NB(self, Cond_matrix, Test_Data_cat, y):
        #Generate conditional probability matrix for categorical features(row 1 = yes, row 2 = no)
        Pcond_cat = np.zeros((2,np.shape(Test_Data_cat)[1]))
        for x in np.arange(np.shape(Test_Data_cat)[1]):
            Index = np.array(np.where(Cond_matrix[:,0] == Test_Data_cat[y,x]))[0]
            Index_shape = np.array(Index.shape[0]).astype('float')
            #If subcategory is non existent in training dataset, then conditional probability for both classes becomes 1.
            if Index_shape < 1:
                Pcond_cat[0,x] = 1
                Pcond_cat[1,x] = 1
            else:
                Pcond_cat[0,x] = np.array(Cond_matrix[Index, 1])
                Pcond_cat[1,x] = np.array(Cond_matrix[Index, 2])
        return Pcond_cat
    
    def fit(self, DataSet, Y):
        #Identification of numerical and categorical indices
        Index_numerical, Index_categorical = self.Num_Cat(DataSet)
        #Calculate prior class probability
        Npy_1, Npy_0, self.py_1, self.py_0, Index_1, Index_0 = self.Prior_Prob(Y)
        if (Index_numerical.shape[0]) > 0:
            DataSet_numerical = DataSet[:,Index_numerical].astype('float')
            #Calculate mean and std. dev. matrices for each class   
            self.mean, self.std = self.mean_std(DataSet_numerical, Index_1, Index_0)
        if (Index_categorical.shape[0]) > 0:
            DataSet_categorical = DataSet[:,Index_categorical]
            #Calculate conditional probabilities for categorical data
            self.Cond_matrix = self.Cond_Cat(DataSet_categorical, Npy_1, Npy_0, Index_1, Index_0)
            return self.mean, self.std, self.Cond_matrix, self.py_1, self.py_0
        else:
            return self.mean, self.std, self.py_1, self.py_0
        
    def predict(self,Test_Data):
        row = np.shape(Test_Data)[0]
        Prediction = np.zeros((row,1))
        Index_numerical, Index_categorical = self.Num_Cat(Test_Data)
        if (Index_categorical.shape[0]) > 0:
            Test_Data_cat = Test_Data[:,Index_categorical]
        if (Index_numerical.shape[0]) > 0:
            Test_Data_num = Test_Data[:,Index_numerical].astype('float')
        for y in np.arange(row):
            #Conditional probability for numerical features (Gaussian Naive Bayes)
            if (Index_numerical.shape[0]) > 0:
                Pcond_num = self.Gauss_NB(Test_Data_num, self.mean, self.std, y)
            else: 
                Pcond_num = np.array([[1 , 1], [1, 1]])
            #Conditional probability for categorical features (Bernoulli/Multinomial Naive Bayes)
            if (Index_categorical.shape[0]) > 0:
                Pcond_cat = self.Bernoulli_NB(self.Cond_matrix, Test_Data_cat, y)
            else: 
                Pcond_cat= np.array([[1 , 1], [1, 1]])
            #Joint probability calculation -----------------------------------------
            Pcond_Total = (np.prod(Pcond_cat, axis = 1))*(np.prod(Pcond_num, axis = 1))
            Pjoint_y1 = Pcond_Total[0]*self.py_1
            Pjoint_y0 = Pcond_Total[1]*self.py_0
            Evidence  = Pjoint_y1 + Pjoint_y0
            
            #Posterior conditional probability---------------------------------------
            Py1_x = Pjoint_y1/Evidence
            Py0_x = Pjoint_y0/Evidence
            
            #Prediction------------------------------------------------------------
            if Py1_x > 0.5:
                Prediction[y,0]= 1
            else:
                Prediction[y,0]= 0  
        return Prediction
class Performance(object):
    def accuracy(self, Prediction, Y_test):
        Accuracy = (np.sum(Prediction == Y_test) / np.size(Y_test)) * 100
        return Accuracy


# # Naive Bayes algorithm - code excecution

# In[ ]:


#Fetch Data from source
Data1 = np.array(np.genfromtxt('ionosphere.csv',delimiter=',', dtype="|U50"))

#Malformed index from distribution
Malformed = [1]

#Read dataset number of rows and columns
R,C = Data1.shape

#Target Binary Encoding
Target_Binary = np.zeros((R))
Target, counts = np.unique(Data1[:,-1], return_counts =True)
Target_Binary [np.where(Data1[:,-1] == Target[0])] = 1

Target_Binary = Target_Binary.reshape(Data1.shape[0],-1)

#Delete Target
Data1 = Data1[:,:-1]

R,C = Data1.shape

#Eliminating malformed features
Data1 = np.asarray(Data1).astype('float')
DataSet_unclean = np.array(Data1)
Data1 = np.delete(Data1,Malformed,1)

DataSet = Data1
Y = Target_Binary



Accuracy_vector_NB = np.array([])
for c in np.arange(1,6):
    CV = Cross_Validation()
    nb = Naive_Bayes()
    perf = Performance()
    
    Train_Data_NB, Test_Data_NB, Y_train_NB, Y_test_NB = CV.k_fold(DataSet, Y, c)
    nb.fit(Train_Data_NB, Y_train_NB)
    Prediction_NB = nb.predict(Test_Data_NB)
    Accuracy_NB = perf.accuracy(Prediction_NB, Y_test_NB)
    Accuracy_vector_NB = np.concatenate((Accuracy_vector_NB, Accuracy_NB), axis = None)
    
Accuracy_mean_NB_DS1 = np.mean(Accuracy_vector_NB)

print("Average Accuracy: ")
print(Accuracy_mean_NB_DS1)

