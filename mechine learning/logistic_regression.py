import numpy as np
import math

#逻辑斯蒂回归算法，用到梯度上升的思想

class logistic_regression(object):
    def __init__(self,hi=None,alpha=1.0,weigh=None):
        self.hi=hi
        self.alpha=alpha
        self.weigh=weigh

    def _sigmoid(self,index):
        res=1.0/(1+math.exp(-index*self.hi))
        return res

    def _gradascent(self,data,label,maxcircle):
        datam=np.mat(data)
        labelm=np.mat(label).transpose()
        (m,n)=datam.shape

        self.alpha=0.1
        self.weigh=np.ones((n,1))
        for i in range(maxcircle):
            res=self._sigmoid(datam*self.weigh)
            error=labelm-res
            self.weigh=self.weigh+self.alpha*datam.transpose()*error
        return  self

    def classify(self,X):
        num=X.shape[0]
        label=[]
        for i in range(num):
            res=self._sigmoid(X[i]*self.weigh)
            if res>0.5:
                label.append('1')
            else:
                label.append('0')
        return label











