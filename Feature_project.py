# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 15:30:59 2018

@author: ZYH
"""
"""
特征矩阵[n_samples,n_features]是建立模型之前的必要条件
所以找到问题相关的任何信息，并转换为特征矩阵的数值很重要
本节内容--特征工程：离散数据，分类数据，文本特征，图像特征
，语音特征等等
"""
#1.分类特征
data = [
        {'price':32323,'rooms':4,'neighborhood':'Queen Anne'},
        {'price':47623,'rooms':5,'neighborhood':'Fremout'},
        {'price':39783,'rooms':2,'neighborhood':'Qesadf'},
        {'price':84323,'rooms':6,'neighborhood':'Fremout'}
        ]
#neighborhood特征为分类特征，如果编码为整数1、2、3在这里
#并不是好办法，因为sklearn基本假设，这些数值可以反映数量，
#可以比较大小，那么需要用到独热编码，即用0，1代表有无
#字典列表，DictVectorizer类可以处理
#选择类
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(dtype=int,sparse=False) #实例化
vec.fit_transform(data) #转化
vec.get_feature_names()
'''
Out: 
['neighborhood=Fremout',
 'neighborhood=Qesadf',
 'neighborhood=Queen Anne',
 'price',
 'rooms']
'''
#但是如果有很多分类变量，维度急剧增加，这时采用稀疏矩阵更好
vec = DictVectorizer(dtype=int,sparse=True) #实例化一个稀疏方法
vec.fit_transform(data) #转化
vec.get_feature_names()
#from sklearn.feature_extraction import FeatureHasher
#from sklearn.preprocessing import OneHotEncoder
#以上两个是另外两个为分类特征编码的工具

#2.文本特征
#文本最基本就是单词统计，即词出现的次数
sample = ['problem of evil',
          'evil queen',
          'horizon problem']
#sklearn的CountVectorizer来处理
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer() #实例化
X = vec.fit_transform(sample)
X #稀疏矩阵
'''
Out: 
<3x5 sparse matrix of type '<class 'numpy.int64'>'
	with 7 stored elements in Compressed Sparse Row format>
'''
#带列标签的DataFrame表示
import pandas as pd
pd.DataFrame(X.toarray(),columns=vec.get_feature_names())

#单词统计，即词出现的次数会是一些常见词聚集太高的权重
#在分类时这样不科学，解决这个问题使用TF-IDF(term frequency-inverse document frequency)
#词频逆文档频率
'''
词频(TF)是一词语出现的次数除以该文件的总词语数。
假如一篇文件的总词语数是100个，而词语“母牛”出现了3次，
那么“母牛”一词在该文件中的词频就是3/100=0.03。
一个计算文件频率(IDF)的方法是文件集里包含的文件总数除以
测定有多少份文件出现过“母牛”一词。
所以，如果“母牛”一词在1,000份文件出现过，
而文件总数是10,000,000份的话，
其逆向文件频率就是 lg(10,000,000 / 1,000)=4。
最后的TF-IDF的分数为0.03 * 4=0.12。
'''
#Tf
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer() #实例化，参数很多
X = vec.fit_transform(sample)
X
X.toarray() #数组
pd.DataFrame(X.toarray(),columns = vec.get_feature_names())

#3.图像特征
#最常用使用像素作为图像的编码方式，这里不再详细介绍，以后涉及在讨论

#4.衍生特征：输入特征经过数学变换衍生出来的新特征
#例如多项式线性回归，不是通过改变模型，而是改变输入数据
#称为基函数回归
#示例
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y);
#一元
from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit);
#多项式回归，先构造特征X2 在进行拟合
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2) #衍生特征
#不改变模型，只改变输入，基函数回归，是核方法的驱动力之一
model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit);

#5缺失值填充
#填充缺失值方法很多，简单的使用列均值，中位数，众数在sklearn中
#应用Imputer类可处理
#生成缺失值数据
from numpy import nan
X = np.array([[ nan, 0,   3  ],
              [ 3,   7,   9  ],
              [ 3,   5,   2  ],
              [ 4,   nan, 6  ],
              [ 8,   8,   1  ]])
y = np.array([14, 16, -1,  8, -5])
#选用类，实例化，参数均值
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean')
X2 = imp.fit_transform(X)
X2
#进行模型拟合应用
model = LinearRegression().fit(X2, y)
model.predict(X2)

#6.特征管道 一系列操作按照一定流程坐下来，可以构造管道
#make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#实例化管道，同实例化模型一样，就是同时实例化多个模型类
#处理缺失值，特征衍生，线性拟合
model = make_pipeline(Imputer(strategy="mean"),
                      PolynomialFeatures(degree=2),
                      LinearRegression())
#数据，特征和目标
print(X) #含有缺失值
print(y)
#模型拟合应用
model.fit(X,y)
print(y)
print(model.predict(X))
