import numpy as np
import naive_bayes

X1=np.array([
        [1,2,3,4,5,3,2,3,4,3,4,3,5,6],
        [2,3,5,5,6,7,7,8,9,5,3,3,4,7]
])
X1=X1.T
y1=np.array([1,1,1,1,1,0,0,0,0,1,0,1,0,0])

clf=naive_bayes.naive_bayes.fit(naive_bayes.naive_bayes,X1,y1)
print(clf.predict(np.arange(5,7)))