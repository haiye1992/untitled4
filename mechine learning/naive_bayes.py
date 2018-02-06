import numpy as np

#朴素的贝叶斯法的学习和分类/多项式模型
#需要用到平滑参数来处理原数据中不存在的数据值
#其实就是极大似然的理论
#还可以用高斯模型来做，把每个符合某一标签的某一个特征看成高斯分布

class naive_bayes(object):
    def __init__(self,alpha=1.0,label_prior=None):
        self.alpha=alpha          #平滑参数
        self.uniquelabel=None
        self.label_prior=label_prior
        self.condition_prob=None



    def _calculate_feature_prob(self,label):
        unique=np.unique(label)
        total=float(len(label))
        label_prob={}
        for i in unique:
            label_prob[i]=(np.sum(np.equal(label,i)+self.alpha)/(total+len(unique)*self.alpha))
        return label_prob

    def fit(self,X,y):
        self.uniquelabel=np.unique(y)
        #if self.label_prior is None:
        label_sum=len(self.uniquelabel)
        self.label_prior=[]
        sample_num=float(len(y))
        for v in self.uniquelabel:
            self.label_prior.append(np.sum(np.equal(y,v)+self.alpha)/
                                        (sample_num+label_sum*self.alpha))

        self.condition_prob={}
        for c in self.uniquelabel:
            self.condition_prob[c]={}
            for i in range(len(X[0])):
                value=X[np.equal(y,c)][:,i]
                self.condition_prob[c][i]=self._calculate_feature_prob(value)
        return self

    def _predict_single(self,X):
        res=-1
        maxprob=0
        for c_index in self.uniquelabel:
            class_prior=self.label_prior[c_index]
            condition_prior=1.0
            feature_prior=self.condition_prob[self.uniquelabel[c_index]]
            j=0
            for feature in feature_prior.keys():      #根据字典的Key遍历
                i=feature_prior[feature]      #type: dict
                s=X[j]
                if s in i:
                    condition_prior*=i[X[j]]
                else:
                    condition_prior*=(1/len(i))
                j+=1
            if condition_prior*class_prior>maxprob:
                maxprob=condition_prior
                res=self.uniquelabel[c_index]
        return res

    def predict(self,X):
        if X.ndim==1:
            return self._predict_single(X)
        else:
            labels=[]
            for i in X.shape[0]:
              label=self._predict_single(X[i])
              labels.append(label)
        return labels





