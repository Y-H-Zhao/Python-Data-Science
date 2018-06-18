# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 22:11:55 2018

@author: ZYH
"""
"""
无参数算法随机森林：集成方法,通过集成多个简单的评估器形成累积效应
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set() #绘图风格

#随机森林是建立在决策树上的集成学习器，首先介绍决策树
from sklearn.datasets import make_blobs
#centers 确定有几个标签，几类
X, y = make_blobs(n_samples=300, n_features=2, centers=4,
                  random_state=0,cluster_std=1.0)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='rainbow') #cmap:颜色库
#决策树评估器
from sklearn.tree import DecisionTreeClassifier 
tree = DecisionTreeClassifier().fit(X,y)

#写一个辅助函数，对分类器结果进行可视化 （Copy）
def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
#可视化
visualize_classifier(DecisionTreeClassifier(),X,y);
#可以发现，决策树黄色和蓝色之间有一个狭长的紫色区域，这表现为决策树的过拟合。
#决策树很容易陷的很深。解决这个问题，通过集成方法：袋装算法
#不断抽取子集进行拟合，通过求均值获得更好的结果。随机决策树的集成算法即为随机森林
#袋装方法实现
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier()
#n_estimators评估器数量 max_samples=0.8 每个评估器80%样本
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8,
                        random_state=1)
visualize_classifier(bag,X,y) #可视化结果，内置拟合过程
#随机决策树集成算法的优化：随机森林
from sklearn.ensemble import RandomForestClassifier #随机森林分类
model = RandomForestClassifier(n_estimators=100,random_state=0) 
model.fit(X,y)
visualize_classifier(model,X,y)

#随机森林回归
##生成由快慢震荡组合生成
rng = np.random.RandomState(42)
x = 10*rng.rand(200)
def model(x,sigma=0.3):
    fast_oscillation = np.sin(5*x)
    slow_oscillation = np.sin(0.5*x)
    noise = sigma * rng.rand(len(x))
    
    return slow_oscillation + fast_oscillation + noise

y = model(x)
plt.errorbar(x,y,0.3,fmt='o');
from sklearn.ensemble import RandomForestRegressor #随机森林回归树
forest = RandomForestRegressor(200)
forest.fit(x[:,None],y)

xfit = np.linspace(0,10,1000)
yfit = forest.predict(xfit[:,np.newaxis])
ytrue = model(xfit,sigma=0)

plt.errorbar(x,y,0.3,fmt='o',alpha=0.5)
plt.plot(xfit,yfit,'-r') #锯齿型
plt.plot(xfit,ytrue,'-k',alpha=0.5) #平滑曲线
#无参数的随机森林模型非常适合处理多周期数据

#案例，随机森林识别手写数字
from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()

#设置图形对象
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
#绘制图形
for i in range(64):
    ax = fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
    #加图注
    ax.text(0,7,str(digits.target[i]))
    
#随机森林分类
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data,digits.target,
                                                random_state=0)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(Xtrain,ytrain)
y_pred = model.predict(Xtest)
#分类结果报告
from sklearn import metrics
print(metrics.classification_report(y_pred,ytest))
#混淆矩阵
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest,y_pred)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False)
plt.xlabel('true value')
plt.ylabel('predict value') #效果很好
'''
原理简单，训练预测都快，多棵树可并行
无参数很灵活，其他模型欠拟合可以有较好表现（重抽样的特点）
不易解释
'''
