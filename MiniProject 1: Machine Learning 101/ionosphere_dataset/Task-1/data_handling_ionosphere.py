
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
#Fetch Data from source
Data1 = np.array(np.genfromtxt('ionosphere.csv',delimiter=',', dtype="|U50"))

#Read dataset number of rows and columns
R,C = Data1.shape

#Target Binary Encoding
Target_Binary = np.zeros((R))
Target, counts = np.unique(Data1[:,-1], return_counts =True)
Target_Binary [np.where(Data1[:,-1] == Target[0])] = 1

#Delete Target
Data1 = Data1[:,:-1]
R,C = Data1.shape

#Distribution
Instances = R
Bad_Percent = (counts[0]/Instances)*100
Good_Percent = (counts[1]/Instances)*100

#Statistics
Data1 = Data1.astype('float')
DF = pd.DataFrame(Data1)

statistics = DF.describe()
#Cross-correlation
corr = DF.corr().round(2)

Statistical_Value = ['count', 'mean', 'std', 'min','25%', '50%','75%', 'max']
statistics['stat'] = Statistical_Value
statistics.insert(0,None,Statistical_Value)

np.savetxt('statistics.csv',statistics, delimiter=',', fmt = '%s')
np.savetxt('correlation.csv',corr, delimiter=',')

#Plot target distribution
col = 'g','c'
Plot_Lables = 'Bad','Good'
Plot_Sizes = [Bad_Percent, Good_Percent]

plt.figure(figsize=(20, 20))
plt.pie(Plot_Sizes,labels=Plot_Lables,radius=0.5,colors=col, autopct='%1.2f%%', startangle=90)
plt.savefig('Target_Distribution.png')
plt.show()

#plot correlation matrix
plt.figure(figsize=(20, 20))
ax = sns.heatmap(corr, annot=True, linewidths=1, linecolor="white", cmap="YlGnBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 2, top - 2)
plt.savefig('Correlation.png')

#plot distribution
for i in range(C):
    plt.figure(figsize=(8,8))
    sns.distplot(Data1[:,i])
    plt.savefig('Feature_Distribution {:03d}.png'.format(i))
    plt.show()