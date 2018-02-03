import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


haha=pd.Series([3,5,6,88,3,54])
print(haha)
a=haha.values                 #type: np.ndarray
index=haha.index
gender=pd.read_csv('D:/pythontest/gender_submission.csv')
train_data=pd.read_csv('D:/pythontest/train.csv')
test_data=pd.read_csv('D:/pythontest/test.csv')
continuous_column_list = ['Age', 'SibSp', 'Fare', 'Parch']
decrete_column_list = ['Sex', 'Pclass', 'Embarked']
cont_train_data=train_data.filter(continuous_column_list)
det_train_data=train_data.filter(decrete_column_list)
cont_train_data.describe()
print ("Parch:", chi2(train_data.filter(["Parch"]), train_data['Survived']))
print ("SibSp:", chi2(train_data.filter(["SibSp"]), train_data['Survived']))


feature = 'Parch'
feature_data = train_data.filter([feature, 'Survived'])
survived_data = feature_data[feature_data.Survived == 1][feature].value_counts()
unsurvived_data = feature_data[feature_data.Survived == 0][feature].value_counts()
df = pd.DataFrame({'Survived': survived_data, 'UnSurvivied': unsurvived_data})
df.plot(kind='bar', stacked=True)
plt.title('Survived_' + feature)
plt.xlabel(feature)
plt.ylabel(u'Number of people')
plt.show()
feature_data.groupby(feature).hist()


for colnum in det_train_data:
       print(det_train_data[colnum].value_counts())

def print_value(features):
    features_data=train_data.filter([features,'Survived'])
    survived=features_data[features_data.Survived==1][features].value_counts()
    unsurvived=features_data[features_data.Survived==0][features].value_counts()
    data= pd.DataFrame({'Survived': survived, 'UnSurvivied': unsurvived})#创建一个df,花括号内为一个字典数据
    data.plot(kind='bar',stacked=True)
    plt.title('Survived_' + features)
    plt.xlabel(features)
    plt.ylabel('number of people')
    plt.show()
print_value('Pclass')

pre_data=train_data.copy()  #type: pd.DataFrame
mean=pre_data.mean(axis=0,skipna=True)
age=mean['Age']
pre_data.loc[np.isnan(train_data['Age']),'Age']= age
pre_data['Age'].fillna(pre_data['Age'].median(),inplace=True)
#pre_data.fillna({'Age':0},inplace=True)
#pre_data.loc[pre_data['Cabin'].notnull, 'Cabin']= 1
#pre_data.loc[, 'Cabin']= 0
#填充数据中的nan值和非nan值
pre_data.loc[pd.notnull(pre_data['Cabin']),'Cabin']= 1
pre_data.fillna({'Cabin':0},inplace=True)
#most_common=pre_data['Embarked'].value_counts()
haha=pre_data['Embarked'].value_counts().index[0]
pre_data.loc[pd.isnull(pre_data['Embarked']),'Embarked']=haha

#将cabin等数据进行虚拟编码，忽略他们之间的距离。
Cabin_dummy=pd.get_dummies(pre_data['Cabin'],prefix='Cabin')
Sex_dummy=pd.get_dummies(pre_data['Sex'],prefix='Sex')
Embarked_dummy=pd.get_dummies(pre_data['Embarked'],prefix='Embarked')

#对数据进行归一化处理，并去除无用的数据
dummy_data=pd.concat([pre_data,Cabin_dummy,Sex_dummy,Embarked_dummy],axis=1)
dummy_data.drop(['Cabin','Sex','Embarked'],axis=1,inplace=True)
#dummy_data['Fare']=StandardScaler.fit_transform(dummy_data.filter(['Fare']))
dummy_data['Fare']=preprocessing.minmax_scale(dummy_data['Fare'])
dummy_data.drop(['PassengerId','Name','Ticket'],axis=1, inplace=True)
#hehe=predict_data.values
xixi=dummy_data.mode()
traindata_afterpre=dummy_data.drop(['Survived'],axis=1)
haha=traindata_afterpre.values    #svm需要使用ndarrary数据，需要转换
predict_data=dummy_data.values
predict_data=predict_data[:,0]
clf=svm.SVC()
clf.fit(haha,predict_data)

#对test数据进行预处理

test_age=test_data['Age'].mean(skipna=True)
test_data['Age'].fillna(test_data['Age'].median(),inplace=True)
test_data.loc[pd.notnull(test_data['Cabin']),'Cabin']=1
test_data.loc[pd.isnull(test_data['Cabin']),'Cabin']=0
most_E=test_data['Embarked'].value_counts().index[0]
test_data.fillna({'Embarked': most_E},inplace=True)
Cabin_dummy_test=pd.get_dummies(test_data['Cabin'],prefix='Cabin')
Sex_dummy_test=pd.get_dummies(test_data['Sex'],prefix='Sex')
Embarked_dummy_test=pd.get_dummies(test_data['Embarked'],prefix='Embarked')
dummmy_test=pd.concat([test_data,Cabin_dummy_test,Sex_dummy_test,Embarked_dummy_test],axis=1)
dummmy_test.drop(['Cabin','Sex','Embarked'],axis=1,inplace=True)
Fare=dummmy_test['Fare'].mean(skipna=True)
dummmy_test.loc[pd.isnull(dummmy_test['Fare']),'Fare']= Fare
dummmy_test['Fare']=preprocessing.minmax_scale(dummmy_test['Fare'])
dummmy_test.drop(['PassengerId','Name','Ticket'],axis=1, inplace=True)

testdata_after=dummmy_test.values
res=clf.predict(testdata_after)   #type: np.ndarray
result=pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':res.astype(np.int32)})
result.to_csv('D:/pythontest/result.csv',index=False)
comp=gender.values
comp=comp[:,1]
enen=abs(comp-res)
accuracy=1-sum(abs(comp-res))/res.size

xtrain,xtest,ytrain,ytest=train_test_split(haha,predict_data,test_size=0.4,random_state=0)
scores=cross_val_score(clf,haha,predict_data,cv=5)


