#!/usr/bin/env python
# coding: utf-8

# # Dataset 4 Cleanup code - function definition

# In[ ]:


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


# # Dataset  4 Cleanup code - execution

# In[ ]:


DataSet = np.array(np.genfromtxt('Electric_Grid_Stability.csv',delimiter=',', dtype="|U50"))    
#A single Dataset is given
#A single target is given
cln4 = DataSet4_Cleanup()
DataSet, Y, Correlation = cln4.clean(DataSet)

