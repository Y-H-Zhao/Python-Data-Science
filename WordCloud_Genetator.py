# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:56:18 2018

@author: ZYH
"""
import os 
os.chdir('D:/pythonshell/WordCloud')

import numpy as np
from PIL import Image #加载图片
import matplotlib.pyplot as plt #绘图
from wordcloud import WordCloud 

#频率表示方法
text1 = {"近似推断":10,"随机模拟":20,"采样方法":35,"蒙特卡洛方法":50}
#字符串表示方法
text2 = "近似推断,随机模拟,采样方法,蒙特卡洛方法" 
#字符串会根据出现次数决定字体大小
text2 = "近似推断,随机模拟,采样方法,采样方法,蒙特卡洛方法,蒙特卡洛方法,蒙特卡洛方法,蒙特卡洛方法"
#加载背景图片 搜索wordcloud背景图片下载相关图片
cloud_mask = np.array(Image.open("bool.jpg"))
wc = WordCloud(
    background_color="white", #背景颜色
    max_words=200, #显示最大词数
    font_path="msyh.ttf",  #使用字体 需要加载中文字体
    mask=cloud_mask, #背景图片
    min_font_size=15,
    max_font_size=50, 
    width=400,  #图幅宽度
    )
#应用频率生成词云
wc.generate_from_frequencies(text1)
# 显示图片
plt.figure()
plt.imshow(wc)
plt.axis("off")
plt.show()
wc.to_file("pic1.png") #保存图片
#应用字符串生成词云
wc.generate(text2)
plt.figure()
plt.imshow(wc)
plt.axis("off")
plt.show()
wc.to_file("pic2.png") #保存图片

#结巴分词使用
'''
import jieba
text="李小璐给王思聪买了微博热搜"
#强调特殊名词
jieba.suggest_freq(('微博'), True)
jieba.suggest_freq(('热搜'), True)
#分词
segs=jieba.cut(text)
mytext_list=[]

#读取标点符号库
f=open("utils/stopwords.txt","r")
stopwords={}.fromkeys(f.read().split("\n"))
f.close()

#文本清洗 去除停用词 空格 长度为1的词
for seg in segs:
    if seg not in stopwords and seg!=" " and len(seg)!=1:
        mytext_list.append(seg.replace(" ",""))
cloud_text=",".join(mytext_list)
cloud_text #结果字符串
'''
