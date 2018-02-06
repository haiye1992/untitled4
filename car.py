import pandas as pd
from sklearn import svm
import xgboost
from sklearn.ensemble import GradientBoostingClassifier

traindata=pd.read_csv('D:/pythontest/car_predict/[new] yancheng_train_20171226.csv')
testdata=pd.read_csv('D:/pythontest/car_predict/yancheng_testA_20171225.csv')
traindata.info()
date1=traindata.filter(['sale_date'])
date=date1.values
class_id=traindata.filter(['class_id'])
classid=class_id.values
hehehe=traindata['sale_quantity'].groupby([date,classid])
hehehe1=hehehe.size
pre=traindata.filter(['class_id','sale_quantity'])
data2=traindata[['department_id','sale_quantity']]
data2mean=data2.groupby(['department_id']).agg('sum').reset_index()
data3=traindata[['sale_quantity','class_id']]
data3mean=data3.groupby(['class_id']).agg('sum').reset_index()
data3mean['predict_quantity']=data3mean.sale_quantity.apply(lambda s: s/70)
data4=traindata[['class_id','sale_quantity','sale_date']]
data4mounthmean=data4.groupby(['class_id','sale_date']).agg('sum').reset_index()
#data5=traindata[['sale_date','sale_']]
data4train=data4mounthmean[((data4mounthmean['sale_date']>=201601)&(data4mounthmean['sale_date']<201612))] #type: pd.DataFrame
data4test=data4mounthmean[((data4mounthmean['sale_date']>201612)&(data4mounthmean['sale_date']<=201710))]  #type: pd.DataFrame
trainlabel=data4train[(data4train['sale_date']==201611)][['class_id','sale_quantity']]
data4train=data4train[(data4train['sale_date']<201611)]

list1=data4train[(data4train['sale_date']==201601)][['class_id','sale_quantity']]
list1.reset_index(inplace=True)
list1=list1.filter(['class_id','sale_quantity'])
list1.columns=['class_id','201601']

list2=data4train[(data4train['sale_date']==201602)][['class_id','sale_quantity']]
list2.reset_index(inplace=True)
list2=list2.filter(['class_id','sale_quantity'])
list2.columns=['class_id','201602']

list3=data4train[(data4train['sale_date']==201603)][['class_id','sale_quantity']]
list3.reset_index(inplace=True)
list3=list3.filter(['class_id','sale_quantity'])
list3.columns=['class_id','201603']

list4=data4train[(data4train['sale_date']==201604)][['class_id','sale_quantity']]
list4.reset_index(inplace=True)
list4=list4.filter(['class_id','sale_quantity'])
list4.columns=['class_id','201604']

list5=data4train[(data4train['sale_date']==201605)][['class_id','sale_quantity']]
list5.reset_index(inplace=True)
list5=list5.filter(['class_id','sale_quantity'])
list5.columns=['class_id','201605']

list6=data4train[(data4train['sale_date']==201606)][['class_id','sale_quantity']]
list6.reset_index(inplace=True)
list6=list6.filter(['class_id','sale_quantity'])
list6.columns=['class_id','201606']

list7=data4train[(data4train['sale_date']==201607)][['class_id','sale_quantity']]
list7.reset_index(inplace=True)
list7=list7.filter(['class_id','sale_quantity'])
list7.columns=['class_id','201607']

list8=data4train[(data4train['sale_date']==201608)][['class_id','sale_quantity']]
list8.reset_index(inplace=True)
list8=list8.filter(['class_id','sale_quantity'])
list8.columns=['class_id','201608']

list9=data4train[(data4train['sale_date']==201609)][['class_id','sale_quantity']]
list9.reset_index(inplace=True)
list9=list9.filter(['class_id','sale_quantity'])
list9.columns=['class_id','201609']

list10=data4train[(data4train['sale_date']==201610)][['class_id','sale_quantity']]
list10.reset_index(inplace=True)
list10=list10.filter(['class_id','sale_quantity'])
list10.columns=['class_id','201610']



list11=data4test[(data4test['sale_date']==201701)][['class_id','sale_quantity']]
list11.reset_index(inplace=True)
list11=list11.filter(['class_id','sale_quantity'])
list11.columns=['class_id','201601']

list12=data4test[(data4test['sale_date']==201702)][['class_id','sale_quantity']]
list12.reset_index(inplace=True)
list12=list12.filter(['class_id','sale_quantity'])
list12.columns=['class_id','201602']

list13=data4test[(data4test['sale_date']==201703)][['class_id','sale_quantity']]
list13.reset_index(inplace=True)
list13=list13.filter(['class_id','sale_quantity'])
list13.columns=['class_id','201603']

list14=data4test[(data4test['sale_date']==201704)][['class_id','sale_quantity']]
list14.reset_index(inplace=True)
list14=list14.filter(['class_id','sale_quantity'])
list14.columns=['class_id','201604']

list15=data4test[(data4test['sale_date']==201705)][['class_id','sale_quantity']]
list15.reset_index(inplace=True)
list15=list15.filter(['class_id','sale_quantity'])
list15.columns=['class_id','201605']

list16=data4test[(data4test['sale_date']==201706)][['class_id','sale_quantity']]
list16.reset_index(inplace=True)
list16=list16.filter(['class_id','sale_quantity'])
list16.columns=['class_id','201606']

list17=data4test[(data4test['sale_date']==201707)][['class_id','sale_quantity']]
list17.reset_index(inplace=True)
list17=list17.filter(['class_id','sale_quantity'])
list17.columns=['class_id','201607']

list18=data4test[(data4test['sale_date']==201708)][['class_id','sale_quantity']]
list18.reset_index(inplace=True)
list18=list18.filter(['class_id','sale_quantity'])
list18.columns=['class_id','201608']

list19=data4test[(data4test['sale_date']==201709)][['class_id','sale_quantity']]
list19.reset_index(inplace=True)
list19=list19.filter(['class_id','sale_quantity'])
list19.columns=['class_id','201609']

list20=data4test[(data4test['sale_date']==201710)][['class_id','sale_quantity']]
list20.reset_index(inplace=True)
list20=list20.filter(['class_id','sale_quantity'])
list20.columns=['class_id','201610']

classs=list6['class_id']
classs=classs.unique()
hahaha=pd.Series(classs)
trainclass=hahaha.to_frame()  #type: pd.DataFrame
trainclass.columns=['class_id']
t=pd.merge(trainclass,list1,how='left')
t=pd.merge(t,list2,how='left')
t=pd.merge(t,list3,how='left')
t=pd.merge(t,list4,how='left')
t=pd.merge(t,list5,how='left')
t=pd.merge(t,list6,how='left')
t=pd.merge(t,list7,how='left')
t=pd.merge(t,list8,how='left')
t=pd.merge(t,list9,how='left')
t=pd.merge(t,list10,how='left')


classs1=testdata['class_id']
classs1=classs1.unique()
hahaha1=pd.Series(classs1)
testclass=hahaha1.to_frame()  #type: pd.DataFrame
testclass.columns=['class_id']
t1=pd.merge(testclass,list11,how='left')
t1=pd.merge(t1,list12,how='outer')
t1=pd.merge(t1,list13,how='outer')
t1=pd.merge(t1,list14,how='outer')
t1=pd.merge(t1,list15,how='outer')
t1=pd.merge(t1,list16,how='outer')
t1=pd.merge(t1,list17,how='outer')
t1=pd.merge(t1,list18,how='outer')
t1=pd.merge(t1,list19,how='outer')
t1=pd.merge(t1,list20,how='outer')
t1=pd.merge(testclass,t1,how='left')




filterlabel=t['class_id']
filterlabel=filterlabel.to_frame()
trainlabel.reset_index(inplace=True)
trainlabel=trainlabel.filter(['class_id','sale_quantity'])

trainlabel=pd.merge(trainlabel,filterlabel,how='right')
trainlabel1=trainlabel.filter(['sale_quantity'])
dummy_class=pd.get_dummies(t['class_id'])
t.fillna(0,inplace=True)
t1.fillna(0,inplace=True)
del t['class_id']
t=t/200
res=pd.concat([t,dummy_class],axis=1)   #type: pd.DataFrame
#res=res.drop('class_id',axis=1)

t2=pd.get_dummies(t1['class_id'])
del t1['class_id']
t1=t1/200
res1=pd.concat([t1,t2],axis=1)       #type: pd.DataFrame


clf=svm.SVC()
clf.fit(t,trainlabel1)
result=clf.predict(t1)

gdbt=GradientBoostingClassifier(n_estimators=100,max_depth=10)
gdbt.fit(t,trainlabel1)
result1=gdbt.predict(t1)

res2=pd.DataFrame(result1)  #type: pd.DataFrame
res2.columns=['predict_quantity']
testdata1=testdata.drop(['predict_quantity'],axis=1)
final=pd.concat([testdata1,res2],axis=1)     #type: pd.DataFrame
final.to_csv('D:/pythontest/car_predict/result2.csv',index=True)




