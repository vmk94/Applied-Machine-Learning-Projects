
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
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
        return DataSet, Y

DataSet = np.array(np.genfromtxt('Electric_Grid_Stability.csv',delimiter=',', dtype="|U50"))    

cln4 = DataSet4_Cleanup()
DataSet, Y = cln4.clean(DataSet)


R,C = DataSet.shape
#Split data
train_range = (R*0.85)
train_range = int(train_range)

train_data = DataSet[:train_range,:]
test_data = DataSet[train_range:,:]

train_label = Y[:train_range]             
test_label = Y[train_range:]

#Logistic Regression
alpha = [0.00001, 0.00005, 0.0001]
learning_rate = 1000
max_acc = 0
opt_aplha_itr = 0

for k in range(len(alpha)):
    accuracy = cross_validation(5,train_data,train_label,alpha[k],learning_rate)
    print ("Acc:", accuracy,"\n")
    acc = sum(accuracy)
    acc/=5
    print ("KFOLD_AV:",acc, "\n\n")
    
    if (acc > max_acc):
        max_acc = acc
        opt_aplha_itr = k


lg = logr(alpha[opt_aplha_itr],learning_rate)
omega,num_iter,cost = lg.fit(train_data,train_label)
y_pred = lg.predict(test_data)
accuracy = evaluate_acc(test_label,y_pred)

print ("\nTEST ACCURACY:", accuracy, "\nOptimal Alpha:",alpha[opt_aplha_itr])