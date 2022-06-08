#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Algorithm - functions definition

# # Dataset4

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10**6)
from statistics import mode as modevalue

class DataSet2_Cleanup(object):
    def Class_Encoding(self,Class):
            Class_Vector = np.zeros((Class.shape[0],1))
            Target = np.unique(Class)
            Class_Vector[np.where(Class == Target[0])] = 1
            return Class_Vector

    def Num_Cat(self,DataSet):
            Categorical_id = np.zeros((1,1))
            Index_numerical = np.array([])
            Index_categorical = np.array([])
            C = np.arange(np.shape(DataSet)[1])
            for x in C:    
                try:
                    Categorical_id = DataSet[0,x].astype('float')
                    Index_numerical = np.concatenate((Index_numerical, x),axis= None).astype('int')
                except ValueError:
                    Index_categorical = np.concatenate((Index_categorical, x),axis= None).astype('int')
            return Index_numerical, Index_categorical

    def Fill_Numerical(self,DataSet, Index_numerical):
        Mean_Vector = np.mean(DataSet[:,Index_numerical].astype('float'), axis = 0)
        for x in Index_numerical:
            DataSet[np.where(DataSet[:,x]== " ?"),x] = Mean_Vector[np.where(Index_numerical == x)]
        return Mean_Vector, DataSet
    
    def Fill_Categorical(self,DataSet, Index_categorical):
        Mode_Vector = np.array([])
        for x in Index_categorical:
            Subcategory = np.unique(DataSet[:,x])
            Frequency = np.zeros((Subcategory.size))
            for y in np.arange(Subcategory.size):
                Frequency[y] = np.array(np.where(DataSet[:,x] == Subcategory[y])).size
            Mode = Subcategory[(np.where(Frequency == np.amax(Frequency)))]
            DataSet[np.where(DataSet[:,x]== " ?"),x] = Mode
            Mode_Vector = np.concatenate((Mode_Vector, Mode), axis = None)
        return Mode_Vector, DataSet
    
    def Clean_Categorical(self,DataSet, Index_categorical, Mode_Vector):
        Percentage_Categorical = np.sum(DataSet[:,Index_categorical] == Mode_Vector, axis = 0)*100/DataSet.shape[0]
        Index_malformed = np.array(np.where(Percentage_Categorical > 80)[0])        
        Index_categorical_cleaned = np.delete(Index_categorical,Index_malformed)
        #Eliminate manually Study-feature since its been already numerically labeled
        Index_categorical_cleaned = np.delete(Index_categorical_cleaned,[1])
        return Index_categorical_cleaned

    def One_Hot_Encoding(self,DataSet, Index_categorical_cleaned):
        rows = DataSet.shape[0]
        OH_Encoding = np.zeros((rows,1))
        for x in Index_categorical_cleaned:
            Subcategory = np.unique(DataSet[:,x])
            Encoding_dummy = np.zeros((rows, Subcategory.size))
            for y in np.arange(Subcategory.size):
                Encoding_dummy[np.where(DataSet[:,x] == Subcategory[y]),y] = 1 
            OH_Encoding = np.hstack((OH_Encoding,Encoding_dummy[:,:-1]))
        OH_Encoding = np.array(OH_Encoding[:,1:])        
        return OH_Encoding

    def Normalize(self, DataSet, Index_numerical):
        Range = np.zeros((3,Index_numerical.size))
        Range[0,:] = np.amax(DataSet[:,Index_numerical].astype('float'), axis = 0)
        Range[1,:] = np.amin(DataSet[:,Index_numerical].astype('float'), axis = 0)
        Range[2,:] = Range[0,:]-Range[1,:]
        DataSet[:,Index_numerical] = (np.subtract(DataSet[:,Index_numerical].astype('float'),Range[1,:])/Range[2,:]).astype('unicode')
        return DataSet
        
    def Data_Clean(self,Train_Data, Test_Data):
            #Binary encoding for target class
            Class_Binary = self.Class_Encoding(Train_Data[:,-1])
            Test_Class_Binary = self.Class_Encoding(Test_Data[:,-1])

            #Join Train_Data and Test_Data for future K-Folding
            DataSet = np.vstack((Train_Data, Test_Data))
            Y = np.vstack((Class_Binary, Test_Class_Binary))

            #Split Dataset from Target
            DataSet = np.array(DataSet[:,:-1])
            
            #Obtain numerical and categorical data indices
            Index_numerical, Index_categorical = self.Num_Cat(DataSet)

            #Fill Categorical Data missing values
            Mode_Vector, DataSet = self.Fill_Categorical(DataSet, Index_categorical)
            
            #Fill Numerical Data missing values
            Mean_Vector, DataSet = self.Fill_Numerical(DataSet, Index_numerical)

            #Identify malformed categorical features
            Index_categorical_cleaned = self.Clean_Categorical(DataSet, Index_categorical, Mode_Vector)

            #One Hot Encoding for categorical data
            OH_Encoding = self.One_Hot_Encoding(DataSet, Index_categorical_cleaned)
            OH_Encoding_unclean = self.One_Hot_Encoding(DataSet, Index_categorical)
            #Normalization of numerical features
            DataSet = self.Normalize(DataSet, Index_numerical)
            
            #Data Concatenation
            DataSet_1 = np.delete(DataSet, Index_categorical, axis = 1)
            
            DataSet_NB = np.hstack((DataSet_1, DataSet[:,Index_categorical_cleaned]))
            DataSet_NB_unclean = np.hstack((DataSet_1, DataSet[:,Index_categorical]))
            DataSet_LR = np.hstack((DataSet_1,OH_Encoding)).astype('float')
            DataSet_LR_unclean = np.hstack((DataSet_1,OH_Encoding_unclean)).astype('float')
            
            return DataSet_LR, DataSet_NB, DataSet_LR_unclean, DataSet_NB_unclean, Y 

class DataSet4_Cleanup(object):
    def Class_Encoding(self,Class):
        Class_Vector = np.zeros((Class.shape[0],1))
        Target = np.unique(Class)
        Class_Vector[np.where(Class == Target[0])] = 1
        return Class_Vector 
    
    def reduce(self, DataSet): 
        #Separate header
        Header = DataSet[0,:]
        DataSet = DataSet[1:,:]
        #Separate Train Data from Class
        Y = np.array(DataSet[:,-1])
        DataSet = DataSet[:,:-1].astype('float')
        Index = np.array([0,4,8,12])
        DataSet = np.delete(DataSet, Index, axis = 1)
        return DataSet, Y
    
    def corr(self, DataSet, Y):
        Corr_Data = np.hstack((DataSet,Y))
        Correlation = np.corrcoef(Corr_Data)[-1,:]
        return Correlation

    def Normalize(self, DataSet):
        Range = np.zeros((3,DataSet.shape[1]))
        Range[0,:] = np.amax(DataSet.astype('float'), axis = 0)
        Range[1,:] = np.amin(DataSet.astype('float'), axis = 0)
        Range[2,:] = Range[0,:]-Range[1,:]
        DataSet = (np.subtract(DataSet.astype('float'),Range[1,:])/Range[2,:])
        return DataSet
    
    def clean(self, DataSet):
        DataSet, Y = self.reduce(DataSet)
        Y = self.Class_Encoding(Y)
        DataSet = self.Normalize(DataSet)
        Correlation = self.corr(DataSet,Y)
        return DataSet, Y, Correlation
        
        
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


# # Naive Bayes Algorithm - Code Execution

# In[ ]:


#DATASET 4 ACCURACY TEST

DataSet = np.array(np.genfromtxt('Electric_Grid_Stability.csv',delimiter=',', dtype="|U50"))    

cln4 = DataSet4_Cleanup()
DataSet, Y, Correlation = cln4.clean(DataSet)

Accuracy_vector_NB = np.array([])

for c in np.arange(1,6):
    CV = Cross_Validation()
    nb = Naive_Bayes()
    perf = Performance()
    lr = Logistic_Regression()
    
    Train_Data_NB, Test_Data_NB, Y_train_NB, Y_test_NB = CV.k_fold(DataSet, Y, c)
    nb.fit(Train_Data_NB, Y_train_NB)
    Prediction_NB = nb.predict(Test_Data_NB)
    Accuracy_NB = perf.accuracy(Prediction_NB, Y_test_NB)
    Accuracy_vector_NB = np.concatenate((Accuracy_vector_NB, Accuracy_NB), axis = None)
     
Accuracy_mean_NB_DS4 = np.mean(Accuracy_vector_NB)

print("Average Accuracy: ")
print(Accuracy_vector_NB)

