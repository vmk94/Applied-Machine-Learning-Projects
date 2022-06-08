
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from logistic_regression import Logistic_Regression as logr
#from sklearn.model_selection import train_test_split;

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
    
#Fetch Data from source
Data1 = np.array(np.genfromtxt('ionosphere.csv',delimiter=',', dtype="|U50"))

np.random.seed(10)
np.random.shuffle(Data1)
#Malformed index from distribution
Malformed = [0,1]

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

#Delete Malformed array
Data1 = np.delete(Data1,Malformed,1)

#Split data
train_range = (R*0.85)
train_range = int(train_range)

train_data = Data1[:train_range,:]
test_data = Data1[train_range:,:]

train_label = Target_Binary[:train_range]             
test_label = Target_Binary[train_range:]

#Logistic Regression
alpha = [0.0001, 0.0005, 0.00005]
learning_rate = 15000
max_acc = 0
opt_aplha_itr = 0

for k in range(len(alpha)):
    accuracy = cross_validation(5,train_data,train_label,alpha[k],15000)
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
