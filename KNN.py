import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

#KNN加PCA处理mnist手写数字

#定义分类函数
def classify(X,dataset,labels,k):
    size=dataset.shape[0]
    diffmat=np.tile(X,(size,1))       #复制矩阵操作
    diffmat=diffmat-dataset
    sqdiff=diffmat**2
    sqDistance=sqdiff.sum(axis=1)   #axis=1代表求行的值的和
    distance=sqDistance**0.5
    sortdisindice=distance.argsort()
    classcount={}
    for i in range(k):
        votelabel=labels[sortdisindice[i]]
        classcount[votelabel]=classcount.get(votelabel,0)+1
        sortedClassCount=sorted(classcount.items(),key=lambda d: d[1],reverse=True)
    return sortedClassCount[0][0]

group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
label = ['A', 'A', 'B', 'B']
res=classify([0,0],group,label,3)
print(res)
train1 = pd.read_csv('D:/pythontest/digit_recognizer/train.csv')
#test=np.loadtxt('D:/pythontest/digit_recognizer/test.csv',dtype=np.str,delimiter=',')
train=train1.values
train=train[1:,:]
trainpca=train[:,1:]/255
pca=PCA(n_components=20)
newx=pca.fit_transform(trainpca)
#train.astype(np.float)
trainword=newx[1:2000,:]
labeltrain=train[1:2000,0]
testword=newx[2000:4000,:]
labeltest=train[2000:4000,0]
sizetest=testword.shape[0]
error=0
for j in range(sizetest):
    imagenum=testword[j]
    wordlabel=classify(imagenum,trainword,labeltrain,5)
    if wordlabel!=labeltest[j]:
        error=error+1
accuarcy=1-(error/sizetest)
print('The accuarcy is:{}'.format(accuarcy))

