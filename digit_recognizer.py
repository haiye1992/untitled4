import numpy as np
import pandas as pd
from sklearn import preprocessing
import cv2
from sklearn.model_selection import train_test_split

np.random.seed(2)

#preprocess
train_data=pd.read_csv('D:/pythontest/digit_recognizer/train.csv')
test_data=pd.read_csv('D:/pythontest/digit_recognizer/test.csv')
trainlabel=train_data['label']
trainpix=train_data.drop(['label'],axis=1)       #type: pd.DataFrame
trainpix=preprocessing.minmax_scale(trainpix)    #type: pd.DataFrame
