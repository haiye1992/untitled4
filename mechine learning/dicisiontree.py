import numpy as np
from math import log

class decisiontree:
   def __init__(self):
        self._tree=None


   def calentropy(self,label):
      size=label.shape[0]
      count={}
      for i in label:
        count[i]=count.get(i,0)+1
      infoentropy=0
      for key in count:
        pxi=float(count[key])/size
        infoentropy-=pxi*log(pxi,2)
      return infoentropy

   def splitdata(self,X,y,index,value):
       ret=[]
       feature=X[:,index]
       X=X[:,[i for i in range(X.shape[1]) if i!=index]]
       for i in range(len(feature)):
           if feature[i]==value:
               ret.append(i)
       return X[ret,:], y[ret]

   def choosebesttree(self,X,y):    #用信息熵最佳增益的方法来选择最佳分类方法
       features=X.shape[1]
       oldinfo=self.calentropy(y)
       bestinfo=0
       bestindex=-1
       for i in features:
           featurelist=X[:,i]
           uniquefeature=set(featurelist)
           newinfo=0.0
           for j in uniquefeature:
               sub_x,sub_y=self.splitdata(X,y,i,j)
               prob=len(sub_y)/float(len(y))
               newinfo+=prob*self.calentropy(sub_y)
           infogain=oldinfo-newinfo
           if infogain>bestinfo:
              bestinfo=infogain
              bestindex=i
       return bestindex

   def majority(self,labellist):
       labelcount={}
       for i in labellist:
           labelcount[i]=labelcount.get(i,0)+1
       sortedindex=sorted(labelcount.items(),key=lambda x:x[1],reverse=True)
       return sortedindex[0][0]

   def createtree(self,X,y,featureindex):
       listlabel=list(y)
       if listlabel.count(listlabel[0])==len(listlabel):
           return listlabel[0]
       if len(featureindex)==0:
           return  self.majority()
       bestindex=self.choosebesttree(X,y)

       bestindexstr=featureindex[bestindex]
       featureindex=list(featureindex)    #list可以删除其中的元素
       featureindex.remove(bestindexstr)
       featureindex=tuple(featureindex)
       myTree={bestindexstr:{}}
       featurevalue=X[:,bestindex]
       unique=set(featurevalue)
       for value in unique:          #递归调用
           sub_x,sub_y=self.splitdata(X,y,bestindex,value)
           myTree[bestindexstr][value]=self.choosebesttree(sub_x,sub_y)
       return myTree

   def fit(self, X, y):

       if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
           pass
       else:
           try:
               X = np.array(X)
               y = np.array(y)
           except:
               raise TypeError("numpy.ndarray required for X,y")

       featureindex = tuple(['x' + str(i) for i in range(X.shape[1])])
       self._tree = self.createtree(X, y, featureindex)
       return self




