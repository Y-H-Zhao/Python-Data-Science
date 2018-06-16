# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:16:05 2018

@author: ZYH
"""
import os
#获取工作目录
os.getcwd()
#设置工作目录
os.chdir('D:/pythonshell/PythonDataScienceHandbook/notebooks')
"""
机器学习包Scikit-Learn，干净，统一，管道命令式的API
支持Numpy数组，Pandas数据框，Scipy稀疏矩阵
"""
#鸢尾花数据
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
type(iris) #Out: pandas.core.frame.DataFrame
#数据表，每行代表一个样本，列为属性,特征
#可视化
sns.set()
sns.pairplot(iris,hue='species',size=1.5) #自带的可视化方法
#一般在使用Scikit-Learn之前，将数据中特征变量和目标变量分别提取出来
#分为特征矩阵和目标数组
X_iris = iris.drop('species',axis=1) #去掉目标列得到特征矩阵
X_iris.shape #它的一个属性，不用加() ，方法需要交()
y_iris = iris['species'] #字典索引
y_iris.shape
y_iris = iris.loc[:,'species'] #显式索引
y_iris.shape
#Scikit-Learn评估器API
#使用步骤
'''
1.通过从Scikit-Learn中导入适当的评估器类，选择模型类
2.合适数值对模型实例化，配置模型超参数
3.整理数据，获取二维特征矩阵[n_sampLes,n_feratures]和目标数组[n_sampLes,?]
4.调用模型实例的fit()方法进行拟合
5.应用模型：predict() transfrom()
'''
#实例1
##简单线性回归
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.RandomState(42) #实例化一个随机器，42为种子
x = 10*rng.rand(50)
y = 2*x-1+rng.randn(50)
plt.scatter(x,y);
##1.通过从Scikit-Learn中导入适当的评估器类，选择模型类
from sklearn.linear_model import LinearRegression
##2.模型实例化,配置模型超参数
###fit_intercept=True 拟合截距项
model = LinearRegression(fit_intercept=True) 
model
'''
out:LinearRegression(copy_X=True,fit_intercept=True,
                 n_jobs=1, normalize=False)
'''
##3.整理数据，获取特征矩阵和目标数组
##特征矩阵[n_sampLes,n_feratures]二维特征矩阵
X = x[:,np.newaxis]
X.shape
type(X) #numpy.ndarray
##4.调用模型实例的fit()方法进行拟合
model.fit(X,y) #运算结果保存在模型属性中，fit()的结果属性均带_
model.coef_ #斜率
model.intercept_ #截距
##5.应用模型：predict() transfrom()
xfit = np.linspace(-1,11,num=50) #默认num=50
Xfit = xfit[:,np.newaxis]
Xfit.shape
yfit = model.predict(Xfit)
##可视化结果
plt.scatter(x,y) #散点图
plt.plot(xfit,yfit) #一起运行就在一张图上
#实例2
##鸢尾花分类
##1.通过从Scikit-Learn中导入适当的评估器类，选择模型类
from sklearn.naive_bayes import GaussianNB #高斯朴素贝叶斯
##2.模型实例化,配置模型超参数
model = GaussianNB()
##3.整理数据，获取特征矩阵和目标数组
##鸢尾花数据
import seaborn as sns
iris = sns.load_dataset('iris')
##可视化
sns.set()
sns.pairplot(iris,hue='species',size=1.5) #自带的可视化方法
##分为特征矩阵和目标数组
X_iris = iris.drop('species',axis=1) #去掉目标列得到特征矩阵
y_iris = iris['species'] #字典索引
##交叉验证中的划分测试集和训练集的类，比手动划分更高效
from sklearn.cross_validation import train_test_split 
Xtrain,Xtest,ytrain,ytest = train_test_split(X_iris,y_iris,
                                             random_state=1)
##4.调用模型实例的fit()方法进行拟合
model.fit(Xtrain,ytrain)
##5.应用模型：predict() transfrom()
ypredict = model.predict(Xtest)
##有accuracy_score工具来验证准确率
from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypredict)
'''
均遵循选模型类，实例化，处理数据格式，拟合，验证这个顺序
'''
#实例3
##鸢尾花降维
from sklearn.decomposition import PCA
model = PCA(n_components=2) #降至2维
import seaborn as sns
iris = sns.load_dataset('iris')
##特征矩阵
X_iris = iris.drop('species',axis=1) #去掉目标列得到特征矩阵
model.fit(X_iris)
'''
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
'''
##应用模型：transfrom()
X_2D = model.transform(X_iris)
##可视化结果
iris['PCA1'] = X_2D[:,0]
iris['PCA1'] = X_2D[:,1]

sns.lmplot('PCA1','PCA2',hue="species",data=iris,fit_reg=False)
#实例4
##鸢尾花聚类
from sklearn.mixture import GMM
model = GMM(n_components=3,
            covariance_type='full')
model.fit(X_iris)
'''
GMM(covariance_type='full', init_params='wmc', min_covar=0.001,
  n_components=3, n_init=1, n_iter=100, params='wmc', random_state=None,
  tol=0.001, verbose=0)
'''
##应用模型：predict()
y_gmm = model.predict(X_iris) #确定标签
##验证
iris['cluster'] = y_gmm
sns.lmplot('PCA1','PCA2',hue="species",data=iris,col='cluster',fit_reg=False)
#应用：手写数字探索
##获取数据
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape #Out: (1797, 8, 8) 1797样本，8*8
##可视化前100
import matplotlib.pyplot as plt
##生成10*10子图网格
fig,axes = plt.subplots(10,10,figsize=(8,8),
                       subplot_kw={'xticks':[],'yticks':[]},
                       gridspec_kw=dict(hspace=0.1,wspace=0.1))
for i,ax in enumerate(axes.flat):
    ax.imshow(digits.images[i],cmap='binary',interpolation='nearest')
    ax.text(0.05,0.05,str(digits.target[i]),
            transform=ax.transAxes,color='green')
X = digits.data
X.shape #Out: (1797, 64) 特征矩阵[n_samples,s_features]
y = digits.target
y.shape #Out: (1797,) 目标数组

##降维 64个特征太多了
##应用流行学习算法中的lsomap降维
from sklearn.manifold import Isomap #选模型类
iso = Isomap(n_components=2) #实例化，设置超参
iso.fit(X) #拟合，数据已经整理好
data_projected = iso.transform(digits.data) #转化
data_projected.shape

##可视化
plt.scatter(data_projected[:,0],data_projected[:,1],c=digits.target,
            edgecolors='none',alpha=0.5)
plt.colorbar(label='digit label',ticks=range(10))
plt.clim(-0.5,9.5)

##分类
from sklearn.naive_bayes import GaussianNB #选模型类
model = GaussianNB() #实例化
##处理数据 划分数据集
from sklearn.cross_validation import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(digits.data,
                                             digits.target,random_state=0)
model.fit(Xtrain,ytrain) #拟合
y_predict = model.predict(Xtest) #应用测试集预测
##准确率
from sklearn.metrics import accuracy_score
accuracy_score(ytest,y_predict)
##制作混淆矩阵
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest,y_predict)
##可视化混淆矩阵，使用Seaborn画出来
import seaborn as sns
sns.heatmap(mat,square=True,annot=True,cbar=False)
plt.xlabel('predicted Value')
plt.ylabel('true Value')
##生成10*10子图网格 小标签
fig,axes = plt.subplots(10,10,figsize=(8,8),
                       subplot_kw={'xticks':[],'yticks':[]},
                       gridspec_kw=dict(hspace=0.1,wspace=0.1))
test_images = Xtest.reshape(-1,8,8)
for i,ax in enumerate(axes.flat):
    ax.imshow(test_images[i],cmap='binary',interpolation='nearest')
    ax.text(0.05,0.05,str(y_predict[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == y_predict[i]) else 'red')
