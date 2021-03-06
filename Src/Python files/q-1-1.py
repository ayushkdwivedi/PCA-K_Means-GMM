#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[14]:


def loadData(data):
    df = pd.read_csv(data)

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    return x,y


# In[15]:


def PCA(x):
    
    print('PCA Started !')
    print('')
    
    mu = np.mean(x, axis=0)
    cov = ( ((x - mu).T).dot(x - mu) ) / (x.shape[0]-1)
#     print('Covariance matrix \n%s' %cov)

    eigenVal, eigenVec = np.linalg.eig(cov)
#     print('Eigenvectors \n%s' %eigenVec)
#     print('\nEigenvalues \n%s' %eigenVal)
    
    eList = []
    for i in range(len(eigenVal)):
        eList.append((np.abs(eigenVal[i]), eigenVec[:,i]))
#     print(eList)

    eList.sort(key=lambda x:x[0])
    eList.reverse()

#     print('Eigenvalues in descending order:')
#     for i in eList:
#         print(i[0])
    
    eSum = sum(eigenVal)
    eVar = []
    for i in sorted(eigenVal, reverse=True):
        eVar.append((i / eSum)*100)
    
    eVar = np.abs(np.cumsum(eVar))
#     print(eVar)

    # Calculating the index of first eigen value, upto which error is <10%
    index = next(x[0] for x in enumerate(eVar) if x[1] > 90)
    print('Number of eigen values selected to maintain threshold at 10% is:',index+1)
    print('')
    
    w = eList[0][1].reshape(len(eigenVec),1)
    for i in range(1,index+1):
        w = np.hstack((w, eList[i][1].reshape(len(eigenVec),1))) #Concatinating Eigen Vectors column wise to form W matrix
#     print('Matrix W:\n', w)
#     print(w.shape)

    x_reduced = x.dot(w)
    print('PCA Reduced Data')
    print('')
    print(x_reduced)
    print('')
    print('PCA Completed !')
    
    return x_reduced
    


# In[16]:


if __name__ == '__main__':
    
    data = '../Dataset/intrusion_detection/data.csv'
    x,y = loadData(data)
#     x = x.iloc[:10,:]
    

    x = StandardScaler().fit_transform(x)
    x_reduced = PCA(x)
#     print(x_reduced)
#     x_reduced = pd.DataFrame(x_reduced)


# In[ ]:




