# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:14:57 2018

@author: ZYH
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:,0],X[:,1],s=50);
#K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#可视化结果
plt.scatter(X[:,0],X[:,1],c=y_kmeans,s=50,cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c='black',s=200,alpha=0.5)
#实现kmeans算法
##距离确定标签
from sklearn.metrics import pairwise_distances_argmin
def find_clusters(X, n_clusters, rseed=2):
    # 1. 随机选择中心
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    #开始迭代
    while True:
        # 2a. 根据最近距离确定标签
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. 求均值确定新的簇中心
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. 迭代停止条件
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels
#应用一下
centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis');
'''
kmeans方法简单好用，但是有几点需要注意何改进
1.不保证全局最优，需要多次使用不同的初始值
2.簇的数量必须提前定好，应用轮廓分析，类内类间距离等方法
3.只能确定线性聚类边界(kmeans基本假设，与其他簇的点相比，数据点更接近自己的簇)
但是簇中心点呈现非线性的复杂形状时，算法失效，必要时应用核kmeans
4.每次迭代，计算所有点，数据量增加，算法速度慢，将每次迭代所有点
的条件放宽，每一步仅用子集来更新，即为批处理的思想，可应用
sklearn.cluster.MiniBatchKMeans来实现
'''
#核kmeans演示
##复杂分类应用基本kmeans难以解决
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200,noise=.05,random_state=0)
from sklearn.cluster import KMeans
labels = KMeans(n_clusters=2,random_state=0).fit(X).predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis');
centers = KMeans(n_clusters=2,random_state=0).fit(X).cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c='black',s=200,alpha=0.5)
##应用核方法，将数据投到高维，使用最近邻图来计算数据的高维表示
##然后用kmeans方法来分配标签
from sklearn.cluster import SpectralClustering
#affinity映射高维方法，assign_labels分配标签方法
model = SpectralClustering(n_clusters=2,affinity='nearest_neighbors',
                           assign_labels='kmeans')
model.fit(X)
labels = model.fit_predict(X) #注意这里和其他模型的区别，没有单独的predict方法
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis');

#案例分析：手写体数字聚类
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10,random_state=0)
cluster = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape #Out: (10, 64)
#显示这十个类中心
fig, ax = plt.subplots(2,5,figsize=(8,3))
centers = kmeans.cluster_centers_.reshape(10,8,8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[],yticks=[])
    axi.imshow(center, interpolation = 'nearest', cmap=plt.cm.binary)
#将学习到的簇标签核真实的相匹配一下
from scipy.stats import mode
labels = np.zeros_like(cluster)
for i in range(10):
    mask = (cluster==i)
    labels[mask] = mode(digits.target[mask])[0]
#可以进行准确性的求解
from sklearn.metrics import accuracy_score
accuracy_score(digits.target,labels) #Out: 0.7935
#混淆矩阵
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');

#更进一步，我们使用t-分布邻域嵌入算法在kmeans前对数据进行预处理
#t-SNE是一个非线性嵌入算法，特别擅长保留簇中的数据点
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,init='pca',random_state=0)
digits_proj = tsne.fit_transform(digits.data) #投影数据

#计算类
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

#排列标签
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters==i)
    labels[mask] = mode(digits.target[mask])[0]
#准确度
accuracy_score(digits.target,labels) #Out: 0.93712

#案例分析：kmeans用于色彩压缩（MiniBatchKMeans）
#pillow图像程序包
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
ax = plt.axes(xticks=[],yticks=[])
ax.imshow(china);

china.shape #(height,width,RGB) #Out: (427, 640, 3)
#其中元素数值0~255的整数表示红黄蓝的信息
#缩放数据至0~1
data = china/255.0
#变形[n_sample,n_features]
data = data.reshape(427*640,3)
data.shape
#可视化前10000个像素的子集
#定义一个可视化子集颜色的函数
def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    
    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);
plot_pixels(data, title='Input color space: 16 million possible colors')

#1600万可能颜色（255*255*255=16581375）缩减到16种
#数据巨大，应用minibatchkmeans
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)

new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
#可视化颜色
plot_pixels(data,colors = new_colors, title='Input color space: 16 colors')

#用计算机结果对原始图像重新上色，每个像素指定为最近的簇中心点
#新的颜色空间（427*640） ，而不是（427*640，3）
china_recolored = new_colors.reshape(china.shape)
china.shape
new_colors.shape

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16);

#效果类似
