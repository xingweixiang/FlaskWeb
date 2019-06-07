# webMachineLearning
web机器学习
=========
## 运行环境
     python 3.7.*
## requirements
    numpy
    pandas
    matplotlib
## 目录
* [web机器学习](#web机器学习)
	* [一、web框架](#一web框架)
		* [1、常用Web框架](#1常用Web框架)
	* [二、无监督机器学习](#二无监督机器学习)
		* [1、聚类算法](#1聚类算法)
	* [三、有监督机器学习](#三有监督机器学习)
		* [1、常见监督式学习算法](#1常见监督式学习算法)
### 一、web框架
- web框架是表示一组库和模块的开发框架，用来支持动态网站、网络应用程序及网络服务的开发。
### 1、常用Web框架
- Flask是一个基于python的，微型web框架。基于Werkzeug WSGI工具箱和Jinja2 模板引擎。<br>
之所以被称为微型是因为其核心非常简单，同时具有很强的扩展能力。  
- Django：是一个高级别的PythonWeb框架，它鼓励快速开发和干净、实用的设计。<br>
它是由经验丰富的开发人员构建的，它处理了Web开发中的许多麻烦，因此您可以专注于编写应用程序，而无需重新发明方向盘。<br>
它是免费的，开源的。 
## 二、[无监督机器学习](/example/chapter_2)
输入数据没有被标记，也没有确定的结果。样本数据类别未知，需要根据样本间的相似性对样本集进行分类。<br>
利用聚类结果，可以提取数据集中隐藏信息，对未来数据进行分类和预测。应用于数据挖掘，模式识别，图像处理等。
### 1、聚类算法
- 聚类分析计算方法主要有如下几种：划分法、层次法、密度算法、图论聚类法、网格算法和模型算法。
```
import numpy as np
from sklearn import mixture
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
# generate two clusters: a with 100 points, b with 50:
np.random.seed(4711)  # for repeatability 
c1 = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])#生成一个多元正态分布矩阵
l1 = np.zeros(100)
l2 = np.ones(100)
c2 = np.random.multivariate_normal([0, 10], [[3, 1], [1, 4]], size=[100,])#生成一个多元正态分布矩阵
print (c1.shape)
#add noise:
np.random.seed(1)  # for repeatability 设随机数种子，生成的随机数结果相同
noise1x = np.random.normal(0,2,100)
noise1y = np.random.normal(0,8,100)
noise2 = np.random.normal(0,8,100)
c1[:,0] += noise1x
c1[:,1] += noise1y
c2[:,1] += noise2

#
fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(111)#1×1网格，第一子图
ax.set_xlabel('x',fontsize=30)
ax.set_ylabel('y',fontsize=30)
fig.suptitle('classes',fontsize=30)
labels = np.concatenate((l1,l2),)#数组拼接
X = np.concatenate((c1, c2),)
pp1= ax.scatter(c1[:,0], c1[:,1],cmap='prism',s=50,color='r')#散点图
pp2= ax.scatter(c2[:,0], c2[:,1],cmap='prism',s=50,color='g')
ax.legend((pp1,pp2),('class 1', 'class2'),fontsize=35)#legend，把多个图放一起
fig.savefig('tmp/classes.png')


#start figure
fig.clf()#reset plt 清除整个当前数字。
fig, ((axis1, axis2), (axis3, axis4)) = plt.subplots(2, 2, sharex='col', sharey='row')

#k-means 聚类算法,又称为k-均值算法
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
pred_kmeans = kmeans.labels_
#axis1 = fig.add_subplot(211)
print ('kmeans:',np.unique(kmeans.labels_))
print ('kmeans:',metrics.homogeneity_completeness_v_measure(labels,pred_kmeans))
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='prism')  # plot points with cluster dependent colors
axis1.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='prism')
#axis1.set_xlabel('x',fontsize=40)
axis1.set_ylabel('y',fontsize=40)
axis1.set_title('k-means(K均值)',fontsize=20)
#plt.show()


#mean-shift 均值漂移,是一种基于密度的聚类算法
ms = MeanShift(bandwidth=7)
ms.fit(X)
pred_ms = ms.labels_
axis2.scatter(X[:,0], X[:,1], c=pred_ms, cmap='prism')
axis2.set_title('mean-shift(均值漂移)',fontsize=20)
print ('ms:',metrics.homogeneity_completeness_v_measure(labels,pred_ms))#评价
print ('ms:',np.unique(ms.labels_))#去重
#gaussian
#g = mixture.GMM(n_components=2)
g = mixture.GaussianMixture(n_components=2)#高斯混合模型
g.fit(X)
print (g.means_ )
pred_gmm = g.predict(X)
print ('gmm:',metrics.homogeneity_completeness_v_measure(labels,pred_gmm))
axis3.scatter(X[:,0], X[:,1], c=pred_gmm, cmap='prism')
axis3.set_xlabel('x',fontsize=40)
axis3.set_ylabel('y',fontsize=40)
axis3.set_title('gaussian mixture(高斯混合模型)',fontsize=20)

#hierarchical
# generate the linkage matrix
Z = linkage(X, 'ward')#层次聚类
max_d = 20
pred_h = fcluster(Z, max_d, criterion='distance')
print ('clusters:',np.unique(pred_h))
k=2
fcluster(Z, k, criterion='maxclust')
print ('h:',metrics.homogeneity_completeness_v_measure(labels,pred_h))
axis4.scatter(X[:,0], X[:,1], c=pred_h, cmap='prism')
axis4.set_xlabel('x',fontsize=40)
#axis4.set_ylabel('y',fontsize=40)
axis4.set_title('hierarchical(层次算法ward链接)',fontsize=20)

fig.set_size_inches(18.5,10.5)
fig.savefig('tmp/comp_clustering.png', dpi=100)

fig.clf()#reset plt
fig = plt.figure(figsize=(20,15))
plt.title('Hierarchical Clustering Dendrogram(层次聚类树状图)',fontsize=30)
plt.xlabel('data point index (or cluster index)',fontsize=30)
plt.ylabel('distance (ward)',fontsize=30)
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
fig.savefig('tmp/dendrogram.png')
#plt.show()
```
![聚类算法](/example/chapter_2/tmp/comp_clustering.png)
![层次聚类树状图](/example/chapter_2/tmp/dendrogram.png)
```
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#line y = 2*x
x = np.arange(1,101,1).astype(float)
y = 5*np.arange(1,101,1).astype(float)
#add noise
noise = np.random.normal(0, 10, 100)
y += noise
fig = plt.figure(figsize=(10,10))
#save
ax = fig.add_subplot(111)
ax.axis([0,102, -20,220])
ax.set_xlabel('x',fontsize=40)
ax.set_ylabel('y',fontsize=40)
#plot
plt.plot(x,y,'ro')
plt.axis([0,102, -20,220])
plt.quiver(60, 100,10-0, 20-0, scale_units='xy', scale=1)
plt.arrow(60, 100,10-0, 20-0,head_width=2.5, head_length=2.5, fc='k', ec='k')
plt.text(70, 110, r'$v^1$', fontsize=20)
#plt.show()
fig.suptitle('主成分分析',fontsize=40)
fig.savefig('tmp/pca_data.png')
```
![聚类算法](/example/chapter_2/tmp/pca_data.png)
## 三、[有监督机器学习](/example/chapter_3)
在监督式学习下，输入数据被称为“训练数据”，每组训练数据有一个明确的标识或结果，如对防垃圾邮件系统中“垃圾邮件”“非垃圾邮件”，对手写数字识别中的“1“，”2“，”3“，”4“等。<br>
在建立预测模型的时候，监督式学习建立一个学习过程，将预测结果与“训练数据”的实际结果进行比较，不断的调整预测模型，直到模型的预测结果达到一个预期的准确率。<br>
监督式学习的常见应用场景如分类问题和回归问题。
### 1、常见监督式学习算法
- 有决策树学习(ID3,C4.5等)，朴素贝叶斯分类，最小二乘回归，逻辑回归（Logistic Regression），支持向量机，集成方法以及反向传递神经网络（Back Propagation Neural Network）等等