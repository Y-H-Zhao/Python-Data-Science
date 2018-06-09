# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:59:50 2018

@author: ZYH
"""
import os
#获取工作目录
os.getcwd()
#设置工作目录
os.chdir('D:/pythonshell/PythonDataScienceHandbook/notebooks')

import numpy as np
print(np.__version__)

#创建数组
##全为0
np.zeros(10,dtype = int)
##全为1
np.ones((3,5),dtype = float)
##全为指定数字
np.full((3,5),3.14)
##list--array
np.array([1,2,3])
##范围内规律数组
np.arange(0,20,2)
##范围内规律数组
np.linspace(1,2,4)
##随机数--均匀分布
np.random.random((3,3))
##随机数--正态分布
np.random.normal(0,1,(3,3))
##随机数--随机范围内整数
np.random.randint(0,10,(3,3))
##单位阵
np.eye(3)


np.random.seed(0) #随机种子
x1 = np.random.randint(10,size=10) #一维
x2 = np.random.randint(10,size=(3,4))
x3 = np.random.randint(10,size=(3,4,5))

#数组属性
##数组维度
print("x3 ndim:",x3.ndim)
##shape
print("x3 shape:",x3.shape)
##size:数组总大小
print("x3 size:",x3.size)
##数据类型
print("x3 dtype:",x3.dtype)
##每个元素字节大小itemsize 总字节
print("itensize:",x3.itemsize,"bytes")
print("nbytes:",x3.nbytes,"bytes")

##数组索引
x1
x1[0]
x1[-1]
x2
x2[0,0]
x2[2,-1]
x1[0] = 8.9
x1[0] #float 赋值给 int 自动截断

##数组切片 x[start:stop:step]
###一维
x1
x1[:5]
x1[5:]
x1[4:7]
x1[::2] #从0开始 每隔一个元素取
x1[2::2] #下标2开始 每隔一个元素取
####step为负数时，start 和 stop 默认被交换
x1[::-1] #step为-1时 其他参数不设定 逆序
x1[5::-1]  #从下标5开始，逆序取到下标为0
###多维子数组
x2[:2,:3]
x2[:3,::2] #三行每隔一列
x2[::-1,::-1] #同时被逆序

###同样效果
print(x2[0,:])
print(x2[0])
###数组切片返回是数组数据的视图，而不是一个副本，那么修改切片数据
###原数据也会随之更改。这意味着处理比较大的数据集时，可以通过处理
###片段来更新原数据集，若想创建数组的副本，那么切片后加上.copy
x2_sub_copy = x2[:2,:2].copy()

##数组变型 reshape
grid = np.arange(1,10).reshape((3,3))
print(grid)
x = np.array([1,2,3])
x[:,np.newaxis] #通过np.newaxis列向量
x.reshape((1,3)) #同样效果

##数组拼接和分裂
x = np.array([1,2,3])
y = np.array([3,2,1])
z = np.array([99,99,99])
print(np.concatenate([x,y,z])) #注意拼接的数组需要使用[]括起来
###二维数组
grid = np.arange(1,10).reshape((3,3))
np.concatenate([grid,grid],axis=0) #默认行合并 垂直方向
np.concatenate([grid,grid],axis=1) #设置列合并 水平方向
###使用np.vstack()垂直栈实现行合并
np.vstack([grid,grid])
###使用np.hstack()水平栈实现列合并
np.hstack([grid,grid])
###np.dstack()沿着第三个维度进行拼接
###np.split() np.vsplit() np.hsplit() 实现分裂
xyz = np.concatenate([x,y,z])
_x, _y, _z = np.split(xyz,[3,6]) #[3,6]分裂点 注意两个分裂点会产生3个片段
print(_x, _y, _z)
grid = np.arange(16).reshape((4,4))
upper,lower = np.vsplit(grid,[2]) #垂直方向划分
left,right = np.hsplit(grid,[2]) #垂直方向划分

#数组计算
##+：np.add -:np.subtract -:np.negative(负数) *:np.multiply
##/:np.divide //:np.floor_divide(取整) **：np.power %:np.mod
## np.abs = np.absolute = abs 
##np.pi np.sin np.cos np.tan
x = np.array([1,2,3])
print("e^x",np.exp(x))
print("2^x",np.exp2(x))
print("3^x",np.power(3,x))

print("log(x)",np.log(x))
print("log2(x)",np.log2(x))
print("log10(x)",np.log10(x))
##复杂函数
from scipy import special
print("gamma(x)",special.gamma(x))
print("ln|gamma(x)|",special.gammaln(x))
print("beta(x,2)",special.beta(x,2))
##所有通用函数可以通过out参数来指定输出
x = np.arange(5)
y = np.empty(5)
np.multiply(x,10,out = y)
print(y)

y = np.empty(10)
np.multiply(x,10,out = y[::2]) #将输出结果存在指定位置
print(y)
##聚合函数reduce 对给定元素重复进行操作，直到剩下一个
x = np.arange(10)
np.add.reduce(x)
np.multiply.reduce(x)
##累积函数accumulate 对给定元素重复进行操作，直到剩下一个 记录每次结果
np.add.accumulate(x)
np.multiply.accumulate(x)
##外积
np.multiply.outer(x,x)

#聚合：最小值最大值和其他值
x = np.random.random(100)
np.sum(x)
x.sum()
np.min(x)
x.min()

x = np.random.random((3,4))
x.sum()
x.sum(axis=0) #对每一列求和
x.sum(axis=1) #对每一行求和
##np.prod:求积 np.mean np.std np.var np.argmin:最小值索引 
##np.argmax：最大值索引 np.median np,percentile:分位点

#广播 向量化操作广播功能：用于不同大小数组的二进制通用函数的一组规则
#不同维度数据做运算的扩展规则，实际应用中体会吧。

#比较，掩码和布尔逻辑 ：基于某些准则来抽取，修改，计数或者其他操作
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set() #设置绘图风格

##利用pd读取数据
rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values
inches = rainfall/254 #单位换算
type(rainfall) #pd读取的values方法返回np.ndarray
inches.shape #2014年1月1日至2014年12月31日的降水量

plt.hist(inches,40); #40设置直方图宽度，“；”号只显示图像，不显示数组数据
##不直观，想知道更多信息，多少天大于0.5英寸等，使用通用函数（>）（<）
##或者掩码
x = np.array([1,2,3,4,5])
x < 3  #Out: array([ True,  True, False, False, False])
x <= 3
x != 3
x == 3
###逐个比较
(x*2) == (x**2) #Out: array([False,  True, False, False, False])
###二维数据
x = np.random.randint(10,size=(3,4))
np.count_nonzero(x<6) #统计小于6的个数
np.sum(x<6) #统计小于6的个数,True记为1
np.sum(x<6,axis=1) #每行小于6
###快速检验任意或者所有是否为True
np.any(x>8)
np.all(x>8)
np.any(x>8,axis=1) #每行
###加入布尔运算符 & |(^) ~ 
np.sum((inches > 0.5) & (inches < 1))
##使用布尔数组作为掩码
x < 3  #Out: array([ True,  True, False, False, False])
##看到x<3返回一个布尔数组，以此作为掩码，索引数据
x[x < 3]
rainy = (inches>0) #为下雨天创建一个掩码
summer = (np.arange(365)-172<90) & (np.arange(365)-172>0) #夏天掩码
print("Rainy day:",inches[rainy])
print("Summer rain:",inches[summer])
#and/or 和&/|

#花哨索引
'''
import random
random.seed(42)
'''
rand = np.random.RandomState(42) #设定种子的另一种方法
x = rand.randint(100,size = 10)
print(x)
##多索引
ind = [3,7,4]
x[ind]
##索引数组
ind = np.array([[3,7],
               [4,6]])
x[ind]  #广播扩展 ，返回同索引数组维度一致的结果
##多维度索引 行列
x = np.arange(12).reshape((3,4))
row_ind = np.array([0,1])
col_ind = np.array([1,3])
x[row_ind,col_ind] #Out: array([1, 7])
x[row_ind[:,np.newaxis],col_ind] #列向量和行向量返回多维结果
'''
Out: 
array([[1, 3],
       [5, 7]])
'''
##注意改变索引结果的值，直接改变原数组的值
##选择随机点的示例 -- 划分数据集
mean = [0,0]
cov = [[1,2],
       [2,5]]
rand = np.random.RandomState(42) 
x = rand.multivariate_normal(mean,cov,100)

import matplotlib.pyplot as plt
import seaborn;seaborn.set() #设置seaborn绘图风格 Seaborn视为matplotlib的补充

plt.scatter(x[:,0],x[:,1]); #';'不显示多余信息
indices = np.random.choice(x.shape[0], 20, replace = False)#随机选择20行

selection = x[indices]
##将选中的点用大圆圈标注出来
plt.scatter(x[:,0],x[:,1],alpha = 0.3)
plt.scatter(selection[:,0],selection[:,1],
            facecolor = 'none',edgecolor = 'b',s=200);
##以后两个语句一起运行可显示在一个图中

#数组的排序
##选择排序
def selection_sort(x):
    for i in range(len(x)):
        swap = i+np.argmin(x[i:])
        (x[i],x[swap]) = (x[swap],x[i])
    return x
x = np.random.randint(10,size=6)    
selection_sort(x)
##numpy中的快速排序 np.sort np.argsort
x = np.random.randint(10,size=6) 
##不修改原始数据的基础上返回一个排序好的数组，使用np.sort
np.sort(x)
print(x)
##修改原始数据返回一个排序好的数组，使用数组的sort方法
x.sort()
print(x)
##返回一个排序好的数组的索引值，使用np.argsort方法
x = np.random.randint(10,size=6) 
ind = np.argsort(x)
x[ind]
##沿行(axis=1)或者列(axis=0)排序
##分割，在左边N个位置返回前N小的值 np.partition np.argpartition(索引值)
x = np.array([7,2,3,1,6,5,4])
np.partition(x,3) #返回数组的前三个值是原数组最小的三个值（任意排序）
ind_3 = np.argpartition(x,3) #返回数组的前三个值是原数组最小的三个值索引值
x[ind_3]
##按行（axis=1） 和按列（axis=0）也是可以的
##示例：k个最近邻
rand = np.random.RandomState(42) 
x = rand.rand(10,2)
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.scatter(x[:,0],x[:,1],s=100);
##计算两两点之间的平方距离
dist_sq = np.sum((x[:,np.newaxis,:]-x[np.newaxis,:,:])**2,axis=-1)
##排序按行
nearest = np.argsort(dist_sq,axis=1)
print(nearest)
##最近邻
k = 2
nearest_partition = np.argpartition(dist_sq,k+1,axis=1)
#将临近节点可视化，将每个点与最近的两个点连接
plt.scatter(x[:,0],x[:,1],s=100)
k = 2
for i in range(x.shape[0]):
    for j in nearest_partition[i,:k+1]:
        #画一条x[i]到x[j]的线
        #使用zip方法
        plt.plot(*zip(x[j],x[i]),color = 'black')
