#!/usr/bin/env python
# coding: utf-8

# # Dataset 2 Cleanup code - function definition

# In[ ]:


import numpy as np

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


# # Dataset 2 Cleanup code - execution

# In[ ]:


Train_Data = np.array(np.genfromtxt('adult.csv',delimiter=',', dtype="|U50"))
Test_Data = np.array(np.genfromtxt('adult_test.csv',delimiter=',', dtype="|U50"))

cln = DataSet2_Cleanup()

#Two CLEAN datasets are given: 1 for Naive Bayes, 1 for Linear Regression
#Two CLEAN datasets are given: 1 for Naive Bayes, 1 for Linear Regression
#A single target is given

DataSet_LR, DataSet_NB, DataSet_LR_unclean, DataSet_NB_unclean, Y = cln.Data_Clean(Train_Data, Test_Data)

