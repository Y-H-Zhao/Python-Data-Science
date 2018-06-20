# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 18:33:11 2018
@author: ZYH
"""
import os
#获取工作目录
os.getcwd()
#设置工作目录
os.chdir('D:/pythonshell/PythonDataScienceHandbook/notebooks')

import pandas as pd
pd.__version__
import numpy as np
import matplotlib.pyplot as plt

#Pandas三个基本数据结构：Series,DataFrame 和 Index

'''
Series:带索引数据构成的一维数组
'''
##创建Series对象 pd.Series(data,index)
###data为list或者数组
data = pd.Series([0.25,0.5,0.75,1.0])
data #返回索引和值
data.values #返回值构成的数组
data.index
###data为字典 index默认为字典键
data = pd.Series({2:'a',3:'b',4:'c'})
data[2]
##索引
data[1]
data[1:3]
##Series是通用的一维数组，数组索引通过隐式定义的整数索引获取数值
##Series对象用一种显式定义的索引，不再局限与整数，可以是任何有意义的索引
##Series是特殊的字典
data = pd.Series([0.25,0.5,0.75,1.0],
                 index=['a','b','c','d'])
data['a']
data = pd.Series([0.25,0.5,0.75,1.0],
                 index=[2,5,3,7])
data[5]
data[[2,5]] #index为list参数
data[[2,5]].values

'''
Pandas提供了一种高效的DataFrame数据结构
DataFrame本质上是一种带行标签和列标签，
支持相同类型数据和缺失值的多维数组
相对于Series,DataFrame可视为通用的二维数组，也可看作特殊的字典
'''
##创建DataFrame对象
###通过Series对象创建
population_dict = {'california':345342,
                   'Teaxs':34221,
                   'New York':12344,
                   'Florida':67523,
                   'Illinios':2134}
population = pd.Series(population_dict)
pd.DataFrame(population,columns=['population'])
###通过Series对象字典创建
area_dict = {'california':3342,
                   'Teaxs':341,
                   'New York':1344,
                   'Florida':673,
                   'Illinios':24}
area = pd.Series(area_dict)
pd.DataFrame({'population':population,
              'area':area}) #列合并
###通过字典的列表创建，字典键位列名，对于列表中不存在的键以缺失值代替
###字典的列表[{},{},...]
pd.DataFrame([{'a':1,'b':3},{'b':1,'c':78}])
'''
Out: 
     a  b     c
0  1.0  3   NaN
1  NaN  1  78.0
'''
###通过Numpy数组创建，可指定行列索引，若不指定，默认整数
DF = pd.DataFrame(np.random.rand(3,2),
             index=['a','b','c'],
             columns=['fo','bar'])
DF
DF['fo']
'''
Index对象可以看作一个不可变数组或者有序集合，其他不多介绍
'''

#数据的取值和选择
##Series对象的选择办法
##Series看作字典
data = pd.Series([0.25,0.5,0.75,1.0],
                 index=['a','b','c','d'])
data['b']
data[['a','b']]
data[['a','b']].values #返回数组
##检查键值对
'a' in data #Out: True
data.keys()
data.items()
list(data.items())
data['e'] = 3 #添加
data
##Series看作一维数组，可以使用索引，掩码，切片等
data['a':'c'] #显式索引切片操作 包括'c'
data[1:3] #隐式索引切片操作 不包括'3'
data[(data>0.3) & (data<2.2)]
##Series 使用索引器loc,iloc,ix
###loc表示索引取值为显式
data.loc['a']
data.loc['a':'c']
'''
Out: 
a    0.25
b    0.50
c    0.75
dtype: float64
'''
###iloc表示索引取值为隐式，左闭右开
data.iloc[1]
data.iloc[1:3]
'''
Out: 
b    0.50
c    0.75
dtype: float64
'''
###ix主要用于DataFrame 索引显式优于隐式，使用loc使代码可读性更高
population_dict = {'california':345342,
                   'Teaxs':34221,
                   'New York':12344,
                   'Florida':67523,
                   'Illinios':2134}
population = pd.Series(population_dict)
area_dict = {'california':3342,
                   'Teaxs':341,
                   'New York':1344,
                   'Florida':673,
                   'Illinios':24}
area = pd.Series(area_dict)

data = pd.DataFrame({'population':population,
              'area':area}) #列合并
##DataFrame看作字典
data['area']
data.area #同样结果
data.area.values #返回数组了
data['area'] is data.area #Out: True
##应该尽量避免使用data.area方式赋值
##增加一列
data['density'] = data['population']/data['area']
data
##DataFrame看作二维数组
data.values #values属性可以用来查看数组数据
data.T #转置
data.values[0]
##DataFrame 使用索引器
data.loc[:'New York',:'population'] #显式
data.iloc[:3,:2] #隐式
###自由自在的ix ，但是代码可读性降低
data.ix[:3,:'population']
data.ix[:'New York',:2] 
###花哨索引
data.loc[data.density>10,['population','area']] #这有点像SQL
data.loc[data.density<10] 
data.loc[:,['population','area']]
data.iloc[data.density>10,[0,2]] #报错iloc不可以使用data.density>10
data.ix[data.density>10,[0,2]] #可以的

#Pandas数值运算方法
##Pandas继承Numpy的计算功能，也实现了一些高级技巧
##对于一元运算：这些通用函数将在输出结果中保留索引和列标签
##对于二元运算Pandas自定对齐索引计算
##因为Pandas是建立在Numpy的基础上，所以Numpy的通用函数
##同样适用于Series和DataFrame数据
import pandas as pd
import numpy as np

rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0,10,4))
ser
df = pd.DataFrame(rng.randint(0,10,(3,4)),
                  columns=['A','B','C','D'])
df
np.exp(ser) #返回依然是Pandas对象
np.sin(df*np.pi/4)
##索引对齐
area = pd.Series({'Alaska':1234,'Texas':784545,
                  'California':3454},name='area')
population = pd.Series({'California':12334,'Texas':12455,
                  'New York':34554},name='population')

population/area #所引对齐，没有的为NaN 代表缺失值
'''
Out[3]: 
Alaska             NaN
California    3.570932
New York           NaN
Texas         0.015875
dtype: float64
'''
population.index | area.index #获取索引
A = pd.Series([2,4,6],index=[0,1,2])
B = pd.Series([1,3,5],index=[1,2,3])

A + B
'''
Out: 
0    NaN
1    5.0
2    9.0
3    NaN
dtype: float64
'''
##如果想用特定的数字代替NaN ，使用方法代替运算符
A.add(B,fill_value=0)
'''
Out: 
0    2.0
1    5.0
2    9.0
3    5.0
dtype: float64
'''
rng = np.random.RandomState(42)
A = pd.DataFrame(rng.randint(0,20,(2,2)),
                  columns=list("ab"))
B = pd.DataFrame(rng.randint(0,10,(3,3)),
                  columns=list("bac"))
A
B
A + B
##用A均值代替缺失值
fill = A.stack().mean() #A.stack()压缩成一维
A.add(B,fill_value=fill)
'''
+ add()
- sub() subtract()
* mul() multiply()
/ div() dicide() truediv()
// floordiv()
% mod()
** pow()
'''
#python中通用函数基本均可以加参数axis来控制行列，默认0为按列计算

#处理缺失值
##处理缺失值的方法:1.通过一个覆盖全局的掩码表示缺失值
##2.用一个标签值表示缺失值
##Python Pandas选择使用标签 浮点型：NaN，或者None对象
###None：Python对象的缺失值，不能用于Numpy和Pandas数组中，可应用于
###'Object'数组对象（由python对象组成的数组）
vals1 = np.array([1,None,3,4])
vals1 #array([1, None, 3, 4], dtype=object) 注意到dtype=object
vals1.sum() #会报错 TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'

###NaN：通用的缺失值表示
vals1 = np.array([1,np.nan,3,4])
vals1.dtype
type(vals1) #比较输出的不同
vals1.min() #Out: nan
vals1.max() #Out: nan
vals1.sum() #Out: nan
np.nansum(vals1) #忽略缺失值
np.nanmin(vals1) #忽略缺失值
np.nanmax(vals1) #忽略缺失值
##函数使用，有时可以直接类.方法，有时可以方法() 以后建议使用方法()
###Pandas字符串类型数据通常用object类型存储 pandas将NaN和None看成可以等价交换的缺失值
##发现缺失值：isnull()，notnull()
data = pd.Series([2.1,np.nan,'hello',None])
data.isnull()
pd.isnull(data) #同样结果
'''
Out[36]: 
0    False
1     True
2    False
3     True
dtype: bool
'''
data[data.notnull()] #掩码作为索引
##剔除缺失值：dropna()
data.dropna() #方法必须加()
###对于DataFrame对象，有一些其他的参数可以设置
df = pd.DataFrame([[1,np.nan,2],
                   [2,3,5],
                   [np.nan,4,6]])
###默认情况下，dropna()删除含有缺失值的整行
df.dropna()   
###设置axis参数值为1，删除含有缺失值的整列
df.dropna(axis=1)
###只删除全部是缺失值的行或者列 how = 'all' 默认how = 'any'
df.dropna(how='all')
###通过thresh设置非缺失值的最小数目，超过这个数目就保留，否则删除整行或者列
df.dropna(thresh=3)
df.dropna(thresh=3,axis=1)
##填充缺失值：fillna()
data = pd.Series([1,np.nan,2,None,3],index=list('abcde'))
data
###用特定数字填充，例如0来填充
data.fillna(0) #注意，这并不改变data。
###用缺失值前面有效数字填充forward-fill
data.fillna(method='ffill')
###用缺失值后面有效数字填充back-fill
data.fillna(method='bfill')
###DataFrame与Series一样，不过需要设置axis 1代表列与之前不一样

#层级索引：三维或者更高维，通过层级索引，配合不同等级，一级一级索引
#转化为类似一维和二维
##多级索引Series，Pandas的MultiIndex类提供丰富操作方法
##这里不再详细演示，需要用到再参考科学手册，另外pd.Panel为三维
##pd.Panel4D为四维数据，索引印染loc(显式),iloc,ix

#合并数据集Concat和Append
##定义一个创建DataFrame的函数，为了方便
def make_df(cols,ind):
    """一个简单的DataFrame"""
    data = {c:[str(c)+str(i) for i in ind] for c in cols}
    return pd.DataFrame(data,ind)

make_df('ABC',range(3))
'''
Out: 
    A   B   C
0  A0  B0  C0
1  A1  B1  C1
2  A2  B2  C2
'''
##Numpy数组的合并，np.concatenate()
x = [1,2,3]
y = [4,5,6]
z = [7,8,9]
np.concatenate([x,y,z]) #python参数对list列表数据很友好 #列合并
##Out: array([1, 2, 3, 4, 5, 6, 7, 8, 9])
np.concatenate([x,y,z],axis=1) 
##AxisError: axis 1 is out of bounds for array of dimension 1
##一位数组不可以设置axis=1
x = np.array([1,2,3]).reshape((1,3))
y = np.array([4,5,6]).reshape((1,3))
z = np.array([7,8,9]).reshape((1,3))
np.concatenate([x,y,z],axis=0) #行合并
##二维
x = [[1,2],[3,4]]
np.concatenate([x,x],axis=1) #列合并
np.concatenate([x,x],axis=0) #行合并
##通过pd.concat实现简易合并
#pd.concat?
'''
Signature: pd.concat(objs, axis=0, join='outer',
                     join_axes=None, ignore_index=False,
                     keys=None, levels=None, names=None,
                     verify_integrity=False, copy=True)
'''
ser1 = pd.Series(['A','B','C'],index=[1,2,3])
ser2 = pd.Series(['D','E','F'],index=[4,5,6])
pd.concat([ser1,ser2])
pd.concat([ser1,ser2],axis=1) #列合并，根据索引相同合并唯一行，没有用NaN
##行合并，索引保留机制（可能导致索引重复保留）
x = make_df('AB',[0,1])
y = make_df('AB',[2,3])
y.index = x.index #索引一致
pd.concat([x,y])
'''
Out: 
    A   B
0  A0  B0
1  A1  B1
0  A2  B2
1  A3  B3
'''
##捕捉错误，verify_integrity=True,索引重复报错
try:
    pd.concat([x,y],verify_integrity=True)
except ValueError as e:
    print('ValueError:',e)
##忽略索引ignore_index=True，索引将重新更新
pd.concat([x,y],ignore_index=True)
##增加多级索引 keys=[]
pd.concat([x,y],keys=['x','y']) 
##行合并，列名相同合并，不同时，没有的值为NaN
##列合并时，行索引相同合并 默认join='outer'并集合并
df1 = make_df('ABC',[1,2])
df2 = make_df('BCD',[3,4])
df1
df2
pd.concat([df1,df2])
pd.concat([df1,df2],axis=1)
##设置join='inner',只保留相同的
pd.concat([df1,df2],join='inner')
pd.concat([df1,df2],axis=1,join='inner')
'''
无相同行索引
Out: 
Empty DataFrame
Columns: [A, B, C, B, C, D]
Index: []
'''
##设置join_axes=[],只保留指定的行或者列
pd.concat([df1,df2],join_axes=[df1.columns])
pd.concat([df1,df2],axis=1,join_axes=[df1.index])
'''
Out: 
    A   B   C    B    C    D
1  A1  B1  C1  NaN  NaN  NaN
2  A2  B2  C2  NaN  NaN  NaN
'''
##当然pd对象也支持append函数，但是需要注意，这些函数不改变原数据
##需要用新变量来保存，这有时很低效，那么我们将继续学习更好的办法

#合并数据集：合并与连接
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##Pandas的基本特性之一就是高性能的内存数据连接核合并方式
##主要是pd.merge函数 基于关系代数
#1.pd.merge() 一对一连接
df1 = pd.DataFrame({'employee':['Bob','Jake','Lisa','Sue'],
                    'group':['Accounting','Engineering','Engineering','HR']})
df2 = pd.DataFrame({'employee':['Bob','Jake','Lisa','Sue'],
                    'hire_data':['2004','2008','2012','2014']})
print(df1);print(df2)
##合并
df3 = pd.merge(df1,df2) #默认how='inner',按某一共同的键内连接
##自动更新行索引
print(df3)

##2.pd.merge() 多对一连接 有一列值有重复，会保存重复值
df4 = pd.DataFrame({'group':['Accounting','Engineering','HR'],
                    'supervisor':['Carly','Guido','Steve']})
print(df3);print(df4) #df3的group列有重复
print(pd.merge(df3,df4)) #保留

##3.pd.merge() 多对多连接 两边共同列都有重复值
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
print(df1);print(df5)
print(pd.merge(df1, df5))

##以上简单默认操作，对merge()有一个简单的了解，下面介绍其主要参数功能
###1.参数on 只有在两个dataframe有共同列名时可以使用
###用在有多个列名相同时，选择某一个。
print(df1);print(df2)
pd.merge(df1,df2,on='employee') 
###2.left_on right_on 没有相同列名，使用left_on指定左边
###使用right_on指定右边
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(df1);print(df3)
pd.merge(df1, df3, left_on="employee", right_on="name")
###获取的结果中两列都会保留 想去除可用drop()
pd.merge(df1,df3,left_on='employee',right_on='name').drop('name',axis=1)
###3.left_index和right_index,通过行索引进行合并
df1a = df1.set_index('employee') #将本列设置为索引
df2a = df2.set_index('employee')
print(df1a);print(df2a)
pd.merge(df1a,df2a,left_index=True,right_index=True)
###jion()方法按照索引进行数据合并
df1a.join(df2a) #同样效果
###left/right_on 或者 left/right_index 可以混合用
###4.数据连接的集合操作规则，当一个值出现在一列，却没有出现在另一列的时候，就需要考虑集合操作规则了
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']})
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink']) #加不加columns这里效果一样
print(df6);print(df7)
pd.merge(df6,df7) #默认inner内连接
pd.merge(df6,df7,how='outer') #外连接 缺失值NaN
pd.merge(df6,df7,how='left') #左连接 
pd.merge(df6,df7,how='right')

###5.重复列名 suffixes参数
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
print(df8);print(df9)
pd.merge(df8, df9, on="name")
pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]) #加后缀
##案例美国各州的统计
pop = pd.read_csv('./data/state-population.csv')
areas = pd.read_csv('./data/state-areas.csv')
abbrevs = pd.read_csv('./data/state-abbrevs.csv')
pop.head()
print(areas.head()) #加不加print()区别也没有out：
print(abbrevs.head())

##目的：美国各州人口密度，pop中有人口，但是州名为缩写
##areas中为全称，这样我们需要先将pop和abbrevs合并找到缩写
##代表的全称，再与areas合并获取人口密度
merged = pd.merge(pop,abbrevs,how='outer',
                  left_on='state/region',right_on='abbreviation') #列名不一样啊
merged.head() #对应起来删除重复的列
merged = merged.drop('abbreviation',axis=1)
merged.head()
##逐行检查一下是否有缺失值
merged.isnull().any()
##结果显示部分population state有缺失值
###population 掩码索引
merged[merged['population'].isnull()].head()
###state #显示索引
merged.loc[merged['state'].isnull(),'state/region'].unique()
###州名缩写缺失弥补
merged.loc[merged['state/region']=='PR','state'] = 'Puerto Rico'
merged.loc[merged['state/region']=='USA','state'] = 'United States'

merged.isnull().any() #state没有缺失值
##合并areas
final = pd.merge(merged,areas,on='state',how='left')
final.head()

final.isnull().any() #area有缺失值
final[final['area (sq. mi)'].isnull()]['state'].unique() #United States
###全国面积没有，不影响，各州加一起就行，这里删除
final.dropna(inplace=True)
final.isnull().any() #any()就每个位置都显示 加any()以列来看
final.head()
##2000年人口总数
data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()
##计算人口密度，首先索引重置，再计算
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']
##从大到小排序
density.sort_values(ascending=False, inplace=True)
density.head()
##密度最低的几个州
density.tail()

#累计和分组
##行星数据
import numpy as np
import pandas as pd
import seaborn as sns;sns.set()
planets = sns.load_dataset('planets')
planets.shape
planets.head()
##1.简单累计功能
###Series
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5)) 
ser
ser.sum()
ser.mean()
ser.describe()
###dataframe 优先按列统计
df = pd.DataFrame({'A':rng.rand(5),
                   'B':rng.rand(5)})
df.mean()
df.describe()
df.mean(axis=1) #按行统计 

planets.isnull().any() #存在缺失值
planets.describe() 
planets.dropna().describe() #删除缺失值
'''
pandas内置累计方法：
count() 计数；first() last() 第一项和最后一项；mean() median()
min() max() std() var(); mad() 均值绝对偏差 ;prod() 所有项乘积；sum()
'''
##2.分割split 应用apply 组合combine
##2.通过groupBy完全这一系列操作
df = pd.DataFrame({'key':['A','B','C','A','B','C'],
                   'data':range(6)},columns=['key','data'])
df
##应用groupby()只需将列名传递进去
df.groupby('key') #返回DataFrameGroupBy对象，在没有应用累积函数之前不计算
##应用累积函数
df.groupby('key').sum()
df.groupby('key').describe()
df.groupby('key').mean()
##下面详细介绍GroupBy的参数设置 应用行星数据
planets.columns
###1.按列取值:orbital_period列
planets.groupby('method')['orbital_period'].median()
###2.按组迭代
for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method,group.shape))
###调用方法
planets.groupby('method')['year'].describe().unstack()

##同groupby一期使用的其他方法：累积aggregate 过滤filter
##转换transfrom 应用apply
rng = np.random.RandomState(0)
df = pd.DataFrame({'key':['A','B','C','A','B','C'],
                   'data1':range(6),
                   'data2':rng.randint(0,10,6)},columns=['key','data1','data2'])
df
###1.累积aggregate
df.groupby('key').aggregate(['min',np.median,max]) #三种方法
###指定不同列不同方法 以下三种结果一样
df.groupby('key').aggregate({'data1':'min','data2':'max'})
df.groupby('key').aggregate({'data1':'min','data2':np.max})
df.groupby('key').aggregate({'data1':min,'data2':np.max})

###2.过滤filter 过滤掉不满足你设定的条件的值 返回缩减后的数据
###保留标准差大于4的列
def filter_func(x):
    return x['data2'].std() > 4
print(df);df.groupby('key').std()
df.groupby('key').filter(filter_func) #A组不满足 都过滤掉 返回B C

###3.转换transfrom #对全部数据转换，返回全新的全部数据
df.groupby('key').transform(lambda x: x-x.mean())

###4.应用apply #对每个组应用任意方法
###自定义方法
def norm_by_data2(x):
    #x 已经分好组
    x['data1'] /= x['data2']
    return x
print(df)
df.groupby('key').apply(norm_by_data2)

##设置分割的键，前面一直使用列名分割
###1.将列表 数组 Series和索引作为分组的键 分组建可以是任意与DataFrame
###相匹配的Serues和列表
L = [0,1,0,1,2,0]
df.groupby(L).sum()

###2.用字典或者Series将索引映射到分组名称
df2 = df.set_index('key')
mapping = {'A':'vo','B':'con','C':'con'}
print(df2)
df2.groupby(mapping).sum()

###3.任意函数 函数将映射到索引
df2.groupby(str.lower).mean() #大写变小写 此时传入函数不需加()

###4.多个有效键的组合 以列表方式传入
df2.groupby([str.lower, mapping]).mean()

###分组案例
decade = 10*(planets['year'] // 10) #返回整十年代
decade = decade.astype(str)+'s'
decade.name = 'decade'
planets.groupby(['method',decade])['number'].sum().unstack().fillna(0)
planets.groupby(['method',decade]).head()

#向量化字符串操作：字符串处理方便
##1.Pandas方便之处
import numpy as np
import pandas as pd

data = ['peter','Paul','MAry','guIdo']
data*2 #复制
#格式化首字母大写：capitalize()
data.capitalize() #AttributeError: 'list' object has no attribute 'capitalize'
#通过for
[s.capitalize() for s in data] #Out: ['Peter', 'Paul', 'Mary', 'Guido']
#pd格式
names = pd.Series(data)
names.str.capitalize() #可以直接调用 存在None也没有问题
##2.Pandas字符串方法 str.方法名
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
'''
前面均加上str.可调用相应方法
len() lower() translate() islower() 
ljust() upper() startswith() isupper() 
rjust() find() endswith() isnumeric() 
center() rfind() isalnum() isdecimal() 
zfill() index() isalpha() split() 
strip() rindex() isdigit() rsplit() 
rstrip() capitalize() isspace() partition() 
lstrip() swapcase() istitle() rpartition() 
'''
monte.str.lower() #小写 返回字符串
monte.str.len() #每个字符串长度  返回数值
monte.str.startswith('T') #是否以T开头 返回布尔值
monte.str.split() #以空格分割 返回复合值

##3.应用正则
'''
Pandas与Python中re的对应 
str.match(): re.match() on each element, returning a boolean. 
str.extract(): re.match() 返回匹配正则的字符串组
str.findall(): re.findall() on each element 
str.replace(): 有正则模式代替字符串
str.contains(): Call re.search() on each element, returning a boolean 
str.count(): 计算符合正则的数量
str.split(): Equivalent to str.split(), but accepts regexps 
str.rsplit(): Equivalent to str.rsplit(), but accepts regexps
'''
#元素前面连续的字母作为名字（first name）
monte.str.extract('([A-Za-a+])') #只提取第一个符合元素
monte.str.extract('([A-Za-z]+)', expand=False) #提取到有不符合为止
#所有开头和结果都是辅音字母的名字 (^):开头 ($):结尾
monte.str.findall(r'^[^AEIOU].*[^aeiou]$')

##4.其他字符串方法
'''
str.
get() 获取元素索引位置的值
slice() 切片取值
slice_replace() 切片替换
cat() 连接字符串
repeat() 重复元素
normalize() 将字符串转化为Unicode规范格式
pad() 在字符串左边 右边 或者两边加空格
wrap() 按照指定列宽换行
join() 用指定分隔符连接
get_dummies() extract dummy variables as a dataframe 
'''
monte.str[3]
monte.str.get(3) #和上面同样效果

monte.str[0:3] #对每个字符串操作
monte.str.slice(0,3) #和上面同样效果
#获取last name
monte.str.split().str.get(-1)
###get_dummies() 按照指定的分隔符 将分割出来的指标 独热编码
###例如A=美国 B=英国 C=喜欢奶酪 D=喜欢午餐肉
full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C',
                                    'B|D', 'B|C', 'B|C|D']})
full_monte
###现在将info独热编码 one-hot
full_monte['info'].str.get_dummies('|')
'''
Out: 
   A  B  C  D
0  0  1  1  1
1  0  1  0  1
2  1  0  1  0
3  0  1  0  1
4  0  1  1  0
5  0  1  1  1
'''

#数据透视表 处理时间序列数据 高性能 eval() query()
