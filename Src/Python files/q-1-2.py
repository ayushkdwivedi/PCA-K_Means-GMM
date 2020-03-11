#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")


# In[4]:


def loadData(data):
    df = pd.read_csv(data)

    # print(df.shape[1])

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    return x,y


# In[5]:


def PCA(x,D):
    
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
    if D == 1:
        index = next(x[0] for x in enumerate(eVar) if x[1] > 90)
    else:
        index = 1
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
    
    return x_reduced
    


# In[6]:


def dist(a, b):
    return np.linalg.norm(a - b, axis=1)


# In[43]:


def kmeans(x_reduced,y,k,D):
    
    centroid= np.random.randint(0, np.max(x_reduced), size=k)
    centroid = centroid.reshape(k,1)
    for i in range (1,x_reduced.shape[1]):    
        centroid_i = np.random.randint(0, np.max(x_reduced), size=k)  #Generate random values between 0 and max(x)-20 of k dimention
        centroid_i = centroid_i.reshape(k,1)
        centroid = np.hstack((centroid,centroid_i))
        
    print("Clustering Started !!!")
    print('')
    print("Initial Centroids Set to:")
    print(centroid)
    print('')

    centroid_old = np.zeros(centroid.shape)
    clusters = np.zeros(len(x_reduced))
    error_dist = dist(centroid, centroid_old)    

    while np.count_nonzero(error_dist) != 0:  #Iterate until there is no change in centroid position
        
        for i in range(len(x_reduced)):  # Labelling each point w.r.t its nearest cluster
            distances = dist(x_reduced[i], centroid)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        
        centroid_old = deepcopy(centroid)
        for i in range(k): #New centroid will be the average of the distances
            points = [x_reduced[j] for j in range(len(x_reduced)) if clusters[j] == i]  #All the points which are under ith cluster as per current clustering
            centroid[i] = np.mean(points, axis=0)
        error_dist = dist(centroid, centroid_old)
        print('Distance moved by Centroids in next interation')
        print(error_dist)
        print('')
    print('Clustering Completed !!!')
    print('')

    if D == 0: 
        color=['cyan','magenta','red','blue','green']
        labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
        for i in range(k):
            points = np.array([x_reduced[j] for j in range(len(x_reduced)) if clusters[j] == i])
    #         print('Points in Cluster: ',i)
    #         print(points)
            if points.size != 0:
                plt.scatter(points[:,0],points[:,1],c=color[i],label=labels[i])
    #     plt.scatter(centroid[:,0],centroid[:,1],s=300,c='y',label='Centroids')
        plt.title('Clusters of Network Attack after reducing data to 2-D using PCA')
        plt.xlabel('Attribte 1')
        plt.ylabel('Attribute 2')
        plt.legend()
        plt.show()


# In[44]:


if __name__ == '__main__':
    
    data = '../Dataset/intrusion_detection/data.csv'
    x,y = loadData(data) 
    x = StandardScaler().fit_transform(x)
    
    print('Doing Clustering by selecting reduced number of dimentions in PCA as per threshold of 10%')
    print('')
    x_reduced = PCA(x,1)    
    kmeans(x_reduced,y,5,1)

    print('')
    print('')
    
    print('Doing Clustering by selecting reduced number of dimentions in PCA as 2 for getting plots')
    print('')
    x_reduced = PCA(x,0)    
    kmeans(x_reduced,y,5,0)


# ![index.png](attachment:index.png)

# In[ ]:




