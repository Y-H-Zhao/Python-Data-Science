# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 09:39:31 2018

@author: ZYH
"""
"""
选择模型类，实例化模型配置超参数，对于模型的结果至关重要
所以需要模型验证(留出集 holdout set)(交叉验证 cross-validation)
"""
#鸢尾花
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
X.shape
y = iris.target
y.shape

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
#不进行模型验证
model.fit(X,y)
y_predict = model.predict(X)
from sklearn.metrics import accuracy_score
accuracy_score(y,y_predict) #Out[63]: 1.0 自己测试自己，肯定是1呀

#留出集 留出一部分 train_test_split
from sklearn.cross_validation import train_test_split
Xtrain, Xvalidation, ytrain, yvalidation = train_test_split(X,y,
                                                            random_state=0,
                                                            train_size=0.5)
model.fit(Xtrain,ytrain)
#验证模型
y_predict = model.predict(Xvalidation)
from sklearn.metrics import accuracy_score
accuracy_score(yvalidation,y_predict) #Out: 0.9067

#交叉验证
import numpy as np
##模型
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
##数据
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
##交叉验证
from sklearn.cross_validation import cross_val_score
score = cross_val_score(model,X,y,cv=5)
np.mean(score) #0.96

#LOO(leave onr out 只留一个训练的交叉验证)
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LeaveOneOut
score = cross_val_score(model,X,y,cv=LeaveOneOut(len(X)))
score.mean()
"""
验证后如何选择最佳模型，效果不好又应该如何改进呢？
用更复杂/更灵活的模型
用更简单/更确定的模型
采集更多数据
对样本添加更多特征
"""
#最有模型时‘偏差’和‘方差’的平衡点
#欠拟合 高偏差 灵活性不足 验证集与训练集表现类似
#过拟合 高方差 灵活性过剩 验证集远远不如训练集表现
#验证曲线：模型负责度为横轴，模型得分为纵轴，绘制在不同数据集的得分
#Scikit-Learn验证曲线 以线性回归做一个例子
from sklearn.preprocessing import PolynomialFeatures #多项式
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline #应用管道

def PolynomialRegression(degree=2,**kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))

##制作一些数据
import numpy as np
def make_data(N, err=1.0, rseed=1):
    #随机抽样数据
    rng = np.random.RandomState(rseed)
    X = rng.rand(N,1)**2
    y = 10-1./(X.ravel()+0.1)
    if err>0:
        y += err*rng.randn(N) #误差
    return X,y
X,y = make_data(40)
##不同多项式，拟合效果
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

X_test = np.linspace(-0.1,1.1,500)[:,None]
plt.scatter(X.ravel(),y,color='black')
axis = plt.axis()
for degree in [1,3,5]:
    y_test = PolynomialRegression(degree).fit(X,y).predict(X_test)
    plt.plot(X_test.ravel(),y_test,label="degree={0}".format(degree))
plt.xlim(-0.1,1.0)
plt.ylim(-2,12)
plt.legend(loc='best') #显示图例
##但是哪种更优呢？sklearn的validation-curve函数
##提供模型，数据，参数名称，验证范围
from sklearn.learning_curve import validation_curve
degree = np.arange(0,21)
##参数名称有严格要求polynomialfeatures__degree表示polynomialfeatures的degree参数
##否则或报错，是两个下划线__
#validation_curve:模型()+数据+参数名称+训练范围+cv
train_score, val_score = validation_curve(PolynomialRegression(),X,y,
                                          'polynomialfeatures__degree',degree,cv=7)
import matplotlib.pyplot as plt
plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score');

"""
学习曲线：反映训练集规模的训练得分/验证集得分曲线
通过以上的验证方法，我们可以得到模型复杂度的选择方案
接下来我们可以根据学习曲线来选择训练数据集的规模
"""
#首先重复上个例子，但是扩大数据量为200
X,y = make_data(200)
from sklearn.learning_curve import validation_curve
degree = np.arange(0,21)
##参数名称有严格要求polynomialfeatures__degree表示polynomialfeatures的degree参数
##否则或报错，是两个下划线__
#validation_curve:模型()+数据+参数名称+训练范围+cv
train_score2, val_score2 = validation_curve(PolynomialRegression(),X,y,
                                          'polynomialfeatures__degree',degree,cv=7)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.plot(degree, np.median(train_score2, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score2, 1), color='red', label='validation score')
plt.plot(degree, np.median(train_score, 1), alpha=0.3,linestyle='dashed',color='blue')
plt.plot(degree, np.median(val_score, 1), alpha=0.3,linestyle='dashed', color='red')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score');
##分析，实线大数据更加支持复杂的模型，学习曲线的特征主要有以下几点
'''
1.特定复杂度的模型对于较小数据集容易过拟合，此时，训练得分高，验证得分低
2.特定复杂度的模型对于较大数据集容易欠拟合
3.模型的验证集得分不会高于训练集
'''
##学习曲线一旦收敛，在不改变模型的情况下，增加更多的数据集是没有用处的
#Scikit-Learn学习曲线 给出2次和9次多项式的学习曲线
from sklearn.learning_curve import learning_curve

fig, ax =plt.subplots(1,2,figsize=(16,6))
fig.subplots_adjust(left=0.0625,right=0.95,wspace=0.1)

for i ,degree in enumerate([2,9]):
    #train_sizes测试使用数据集占全集的比例
    #learning_curve:模型(参数)+数据+cv+训练范围
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree),
                                         X, y, cv=7,
                                         train_sizes=np.linspace(0.3, 1, 25))
    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1],
                 color='gray', linestyle='dashed')

    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree = {0}'.format(degree), size=14)
    ax[i].legend(loc='best')
##二次多项式全集的0.15即可收敛，更多数据浪费，更复杂9次多项式0.5以后收敛
'''
如何应用验证曲线和学习曲线呢？
sklearn 中grid_search提供了一个自动化工具GridSearchCV
下面给出一个示例
'''
from sklearn.grid_search import GridSearchCV
#构造超参数搜索空间，本例涉及三个参数
#polynomialfeatures中的degree多项式次数
#linearregression中fit_intercept是否拟合截距
#linearregression的normalize是否标准化
param_grid = {'polynomialfeatures__degree':np.arange(21),
              'linearregression__fit_intercept':[True,False],
              'linearregression__normalize':[True,False]}
#搜索模型，模型函数别忘了() 模型+参数空间+cv 
grid = GridSearchCV(PolynomialRegression(),param_grid,cv=7)
#拟合搜索模型，结果返回最有超参数
grid.fit(X,y)
#最优超参数
grid.best_params_ #结果属性带_
#应用最优超参数拟合数据
model = grid.best_estimator_
plt.scatter(X.ravel(),y)
lim = plt.axis()
y_test = model.fit(X,y).predict(X_test)
plt.plot(X_test.ravel(),y_test,hold=True);
plt.axis(lim)
##GridSearchCV网格搜索系统许多参数选项，具体问题具体设置，包括自定义的得分函数
