import pandas as pd
from sklearn import svm
import xgboost

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
#class_id=traindata.filter(['class_id'])
pre1=traindata.filter(['class_id','sale_quantity'])
pre2=pre1.groupby(["class_id"]).sum()
classsum=pre.groupby(pre['class_id']).sum()       #type: pd.DataFrame
monthsum=classsum/70            #type: pd.DataFrame
monthsum=monthsum.astype(int)
#monthsum=monthsum.reindex(range(140),method='ffill')
mounth=monthsum.values
mounthsum=pd.DataFrame(mounth,columns=['quantity'])
hehe=classsum.index
listhehe=[]
for i in hehe:
    listhehe.append(i)
dfhehe=pd.DataFrame(listhehe,columns=['class_id'])
trainhehe=pd.concat([dfhehe,mounthsum],axis=1,join_axes=[dfhehe.index])
clf=svm.SVC(monthsum,dfhehe)
