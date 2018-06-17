# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 09:09:55 2018

@author: ZYH
"""
"""
朴素贝叶斯分类器，简单快速，适用于维度非常高的数据。
因为速度快，适合为分类问题提供粗糙的基本方案
"""
"""
贝叶斯公式，需要知道先验概率，即每个标签情况下P(特征|标签i).
这称为生成模型，因为它可以训练出可以生成数据的随机过程（概率分布）
朴素贝叶斯，做简单假设：分布情况（正态等常见分布），或使用近似解。
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() 
#高斯朴素贝叶斯 :每个标签的数据服从简单的高斯分布
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100,n_features=2,centers=2,random_state=2,
                  cluster_std=1.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='RdBu'); #散点图c-color
#假设数据服从高斯分布，且变量无协方差，即不相关，这样可以计算后验概率
from sklearn.naive_bayes import GaussianNB
model = GaussianNB() #基本没有超参数
model.fit(X,y)
#生成一些新的数据
rng = np.random.RandomState(0)
Xnew = rng.rand(2000,2)*[14,18]+[-6,-14]
ynew = model.predict(Xnew)

plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:,0],Xnew[:,1],c=ynew,s=20,cmap='RdBu',alpha=0.1)
plt.axis(lim);
#显示后验概率分布情况
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)

#多项式朴素贝叶斯：假设特征是由一个简单多项式分布生成
#多项式分布可以描述各种类型样本出现次数的概率，因此多项式朴素贝叶斯非常
#适用于描述出现次数或者出现比例的特征。常用于文本分类，其特征指待分类文本
#的单词出现次数或者频次
##案例使用20个网络新闻组语料库（2000篇新闻）的单词出现次数作为特征
##加载数据
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names
##选择四类新闻来演示
categories = ['talk.religion.misc','soc.religion.christian',
              'sci.space','comp.graphics']
train = fetch_20newsgroups(subset='train',categories=categories)
test = fetch_20newsgroups(subset='test',categories=categories)
print(train.data[5],train.target[5])
#文本特征tf-IDF ,形成管道
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(),MultinomialNB()) #朴素贝叶斯，参数很少
model.fit(train.data,train.target)
labels = model.predict(test.data)
#评估性能，混淆矩阵
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,
            xticklabels=train.target_names,yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predict label')
#利用训练好的模型做一个可以对任意字符串进行分类的工具
def predict_category(s,train=train,model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]
predict_category('sending a payload to the ISS')

"""
朴素贝叶斯分类器的应用场景
优点：
训练和预测的速度非常快
直接用概率预测
容易解释
可调参数少
适用场景：
假设的分布函数与数据匹配
各种类型的区分度比较高
非常高维度的数据
"""
