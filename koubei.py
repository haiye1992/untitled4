import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import jieba

data=pd.read_csv('D:/pythontest/jingqukoubei/train_first.csv')
data.replace(',',' ',inplace=True)
score=data.filter(['Score'])
score_1=score[score.Score==1].count()
score_2=score[score.Score==2].count()
score_3=score[score.Score==3].count()
score_4=score[score.Score==4].count()
score_5=score[score.Score==5].count()
#listscore=pd.Series([score_1,score_2,score_3,score_4,score_5])
hehe=jieba.cut('我最喜欢的球星是麦迪')
split=jieba.cut(data['Discuss'],cut_all=False)
splitword=[]
for i in data['Discuss']:
   split=jieba.cut(i,cut_all=False)
   split=" ".join(split)
   splitword.append(split)

discuss_split=pd.Series(splitword)
discuss_split.replace(',',' ',inplace=True)





