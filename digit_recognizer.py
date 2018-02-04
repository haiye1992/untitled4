import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

np.random.seed(2)

#preprocess
train_data=pd.read_csv('D:/pythontest/digit_recognizer/train.csv')
test_data=pd.read_csv('D:/pythontest/digit_recognizer/test.csv')
trainlabel=train_data['label']
trainpix=train_data.drop(['label'],axis=1)       #type: pd.DataFrame
trainpix=preprocessing.minmax_scale(trainpix) #type: pd.DataFrame
pca=PCA(n_components=20)
trainpix=trainpix.reshape(-1,28,28,1)
trainpix=pca.fit_transform(trainpix)
X_train,X_test,Y_train,Y_test=train_test_split(trainpix,trainlabel,test_size=0.2,random_state=0)

clf=GradientBoostingClassifier(n_estimators=30).fit(X_train,Y_train)
xixi=clf.score(X_test,Y_test)