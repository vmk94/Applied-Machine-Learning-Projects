
#Include Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mode as d_mode
import seaborn as sns; sns.set()

#Fetch Data from source
Data1 = np.array(np.genfromtxt('breast-cancer-wisconsin.csv',delimiter=',', dtype="|U50"))

#Read dataset number of rows and columns
R,C = Data1.shape

#Target Binary Encoding
Target_Binary = np.zeros((R))
Target, counts = np.unique(Data1[:,-1], return_counts =True)
Target_Binary [np.where(Data1[:,-1] == Target[0])] = 1

#Replace missing values
for x in range(C):
    Mode = d_mode(Data1[:,x])
    Data1[np.where(Data1[:,x]== "?"),x] = Mode.astype('int')
    
#Delete Target
Data1 = Data1[:,:-1]
R,C = Data1.shape

#Distribution
Instances = R
Benign = (counts[0]/Instances)*100
Malignant = (counts[1]/Instances)*100

#Cross-correlation
Data1 = Data1.astype('float')
DF = pd.DataFrame(Data1)
DF.columns=['Id number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses']
DF = DF.drop(['Id number'], axis=1)
statistics = DF.describe()
corr = DF.corr().round(2)

Statistical_Value = ['count', 'mean', 'std', 'min','25%', '50%','75%', 'max']
statistics['stat'] = Statistical_Value
statistics.insert(0,None,Statistical_Value)

#Save data in a file
np.savetxt('statistics.csv',statistics, delimiter=',', fmt = '%s')
np.savetxt('correlation.csv',corr, delimiter=',')

#Plot Target distribution
col = 'g','c'
Plot_Lables = 'Benign','Malignant'
Plot_Sizes = [Benign, Malignant]

plt.pie(Plot_Sizes,labels=Plot_Lables,radius=1,colors=col, autopct='%1.2f%%', startangle=90)
#plt.show()
plt.savefig('Target_Distribution.png')

#plot correlation matrix
plt.figure(figsize=(10, 10))
ax = sns.heatmap(corr, annot=True, linewidths=1, linecolor="white", cmap="YlGnBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 2, top - 2)
plt.savefig('Correlation.png')

#plot distribution
for i in range(C):
    plt.figure(figsize=(8, 8))
    sns.distplot(Data1[:,i])
    plt.savefig('Feature_Distribution {:03d}.png'.format(i))
    plt.show()