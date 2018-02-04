import numpy as np
import pandas as pd

data1=pd.read_csv('D:/pythontest/digit_recognizer/test.csv')
#data=np.loadtxt('D:/pythontest/digit_recognizer/test.csv',dtype=np.str,delimiter=',')
#data=data[1:,:]     #type: np.ndarray
data=data1.values


def distance(val1,val2):
    return np.linalg.norm(val1-val2)

#随机求取中心点
def center(X,k):
    n=X.shape[1]
    centroids=np.empty((k,n))
    for j in range(n):
        minj=min(X[:,j])
        maxj=max(X[:,j])
        rangej=maxj-minj
        centroids[:,j]=(minj+rangej*np.random.rand(k,1)).flatten()
    return centroids

def fit(X,centroids):
    if not isinstance(X,np.ndarray):
        try:
            X = np.asarray(X)
        except:
            raise TypeError("numpy.ndarray required for X")

    m=X.shape[0]
    k=centroids.shape[0]
    cluster1=np.empty((m,2))
    clusterchanged=True
    for i in range(m):
        mindist=np.inf
        minindex=-1
        for j in range(k):
            distj=distance(centroids[j,:],X[i,:])
            if distj<mindist:
                mindist=distj
                minindex=j
                if cluster1[i,0]!=minindex:
                    clusterchanged=True
                    cluster1[i,:]=minindex,mindist**2
    return cluster1

haha=center(data,9)


