# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 11:01:33 2018

@author: ZYH
"""
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np

#简单线性回归
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y);

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit);
print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)

rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.])

model.fit(X, y)
print(model.intercept_)
print(model.coef_)

#基函数回归：将输入变量做变换，将变量之间的线性回归模型转换为非线性回归
##1.多项式基函数：PolynomialFeatures转化器
from sklearn.preprocessing import PolynomialFeatures
x = np.array([2,3,4])
poly = PolynomialFeatures(degree=3,include_bias=False)
poly.fit_transform(x[:,np.newaxis]) #x[:,np.newaxis]转化为二维

#管道多项式回归
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
model = make_pipeline(PolynomialFeatures(degree=7),
                      LinearRegression(fit_intercept=True))

rng = np.random.RandomState(1)
X = rng.rand(50)*10
y = np.sin(X)+0.1*rng.randn(50)

model.fit(X[:,np.newaxis],y)
y_predict = model.predict(xfit[:,np.newaxis])

plt.scatter(X,y)
plt.plot(xfit,y_predict)

##2.高斯基函数：python没有内置高斯基函数，不过我们可以仿照多项式
##基函数的构造方式自行构造，当然也可以构造其他形式的基函数。
from sklearn.base import BaseEstimator, TransformerMixin
class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)
    
gauss_model = make_pipeline(GaussianFeatures(20),
                            LinearRegression())
gauss_model.fit(X[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.scatter(X, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10);

#正则化：引入基函数，有时引起过拟合，对较大的模型参数进行惩罚
#抑制模型的剧烈波动，解决过拟合问题。
##首先来观察如果选择太多的高斯基函数
model = make_pipeline(GaussianFeatures(30),
                      LinearRegression())
#数据
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
#拟合
model.fit(x[:, np.newaxis], y)
plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis]))
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5);
#看看过拟合原因
def basis_plot(model, title=None):
    fig, ax = plt.subplots(2, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
    
    if title:
        ax[0].set_title(title)

    ax[1].plot(model.steps[0][1].centers_,
               model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location',
              ylabel='coefficient',
              xlim=(0, 10))
    
model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)
##当相邻高斯过近，参数相互抵消，参数近似为0
##1.岭回归，L2范数正则化，吉洪诺夫正则化：对模型参数的平方和进行惩罚
##这些带惩罚项的模型内置在sklearn的Ridge评估器中
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30),Ridge(alpha=0.1))
basis_plot(model,title = 'Ridge Regression') #好很多
model.fit(x[:, np.newaxis], y)
plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis]))
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5);
##2.Lasso回归，L1范数正则化：对模型参数的绝对值和进行惩罚
##这些带惩罚项的模型内置在sklearn的Lasso评估器中
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
model = make_pipeline(GaussianFeatures(30),
                      Lasso(alpha=0.001))
basis_plot(model,title='Lasso Regression')
model.fit(x[:, np.newaxis], y)
plt.scatter(x, y)
plt.plot(xfit, model.predict(xfit[:, np.newaxis]))
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5);
