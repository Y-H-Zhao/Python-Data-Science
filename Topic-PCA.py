# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:42:47 2018

@author: ZYH
"""
"""
主成分PCA，快速的线性无监督降维方法
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

#1.PCA的常规处理
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2,2),rng.randn(2,200)).T
plt.scatter(X[:,0],X[:,1])
plt.axis('equal');

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

pca.components_ #成分
pca.explained_variance_ #可解释方差
pca.explained_variance_ratio_ #累积解释比例
#可视化结果
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal'); 
#箭头长度代表可解释方差大小
#降维转化
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)
#降维结果与原结果对比
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');

#1.PCA的用于可视化
#手写体
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape
#降2维处理
pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)
#可视化
plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap='rainbow')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();

#选择主成分的数量--类似碎石图
pca = PCA().fit(digits.data) #不设置主成分数量
plt.plot(np.cumsum(pca.explained_variance_ratio_)) #累积主成分解释比例
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
#前10累积贡献率
np.cumsum(pca.explained_variance_ratio_)[:10]
##数据维度很大，样本数量很多，可以使用以下PCA
##import sklearn.decomposition import RandomizedPCA
##比标准PCA更快，适用于高维数据，其他用法与PCA一致

'''
降维建模、高维数据可视化、噪音过滤、高维数据特征选择
可解释性强，应用广泛
线性变化，只能降维处理。如果需要非线性变化，或者升维处理，可以考虑流行学习
和深度学习无监督学习：自编码器和受限玻尔兹曼机
'''
