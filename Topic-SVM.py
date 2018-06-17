# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 16:41:45 2018

@author: ZYH
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns;sns.set()

"""
区别于生成模型，不再为每类数据建模，而是用一条分割线或
者流行体将各种类型分离开
"""
#首先来看一组简易分类数据
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50,n_features=2,centers=2,
                  random_state=0,cluster_std=0.6)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn');
#不只有一条直线可以将它们分离 ，但是新数据点'x'分类却不一样
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5);
#支持向量机，边界最大化
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5);
#可以得到这三条线的边界，中间那条0.55最大，阴影区域面积也最大
#支持向量机。选择边界最大的那条线为模型最优解。支持向量机本质是一个
#边界最大化评估器
#1.拟合支持向量机：使用线性核函数，参数C设置为很大的值
from sklearn.svm import SVC #Support Vector classifier
model = SVC(kernel='linear',C=1E10)
model.fit(X,y)

#创建一个辅助函数画出SVM的决策边界
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],marker="o",
                   s=300,linewidths=1,facecolors='None',edgecolors='green');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plot_svc_decision_function(model);    
#边界上的关键的点称为支持向量,因为只要其他点不越边界，在哪里有多少都无所谓
#对远离边界的点不敏感是支持向量机的优势
model.support_vectors_ #结果属性带_

#2.拟合支持向量机：使用非线性核函数:径向基函数
#首先制作一些非线性可分数据
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(n_samples=100,factor=.1,noise=.1,random_state=1)
clf = SVC(kernel='linear').fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plot_svc_decision_function(clf,plot_support=False)
##线性分不清
##考虑基函数方法，将数据投影到高维空间，从而使线性分割器有效
##简单有效方法，计算一个以数据圆圈为中心的径向基函数
r = np.exp(-(X**2).sum(1))
#将这一个维度数据加进去，绘制一下三维图，可以发现数据可以使用一个
#平面分隔开
'''
这个步骤中，选择径向基函数是一个关键，如果选择的不好，就不能划分得很干净
但是选择径向基函数比较困难，模型可自动指出
一种策略是计算基函数在数据集上每个点的变化结果，让从所有结果中筛选出最优解。这种
基函数的变换方式称为核变换，是基于每对数据点之间的相似度计算。
问题是N个数据点投到N维空间，维度灾难。由于核函数技巧，不需要这样，使SVM强大起来。
'''
clf = SVC(kernel='rbf',C=1E6) #(kernel='rbf'径向基函数 核技巧内置
clf.fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plot_svc_decision_function(clf,plot_support=True)
clf.support_vectors_

#3.拟合支持向量机：SVM优化--软化边界（数据不干净，数据有重叠）
X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=1.2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
'''
解决这个问题，SVM实现一些修正因子来软化边界，它允许一些点在边界内部，边界线的硬度
由C来控制，C很大，边界很硬，数据不能进入边界，反之。
'''
X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=0.8)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none');
    axi.set_title('C = {0:.1f}'.format(C), size=14)
#C值根据实际情况，这是超参数的设置，由交叉验证等方法搞定。
##案例人脸识别
##下载数据
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
##可视化数据
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])
##选择方法类，维度太高PCA降维，然后SVM分类，应用管道
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline
##先实例化，在搞在一起，或者直接带着超参数放进去
pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)
##处理特征矩阵，划分数据
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state=42)
##网格搜索选择最优超参数svc__C svc__gamma：径向基函数核的大小
from sklearn.grid_search import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain,ytrain) #拟合网格内超参数 不拟合是没有用的
##打印最优超参数，最优参数落在网格中间，如果落在网格边界，则需要扩大范围
print(grid.best_params_)
##实例化最有超参数模型，拟合
model = grid.best_estimator_
yfit = model.predict(Xtest)
##可视化结果
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);
##分类效果报告
from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces.target_names))
##绘制混淆矩阵
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
##当然在实际应用中难以获取切割整齐的人脸图片
##需要使用更复杂的算法找到人脸，然后提取图片中与像素无关的人脸特征
##其实就是特征工程的范围，可采用openCV配合其他手段

#总结
'''
优点：
依赖支持向量较少，消耗内存少
一旦完成模型，预测很快
只受边界数据的影响，对高维数据学习效果非常好
与核函数配合具有通用性
缺点：
大样本学习的计算成本非常高，注意是大样本，不是维度大
依赖C的选择，样本大，选择计算量更大
预测结果不能用直接用概率解释，通过SVC的probability可设置，但是计算量很大
'''
#综上：计算资源足以支撑训练和交叉验证，可以获得不错结果
