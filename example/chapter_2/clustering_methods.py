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


#measures:
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import silhouette_score

res = homogeneity_completeness_v_measure(labels,pred_kmeans)
print ('kmeans measures, homogeneity:',res[0],' completeness:',res[1],' v-measure:',res[2],' silhouette score:',silhouette_score(X,pred_kmeans))
res = homogeneity_completeness_v_measure(labels,pred_ms)
print ('mean-shift measures, homogeneity:',res[0],' completeness:',res[1],' v-measure:',res[2],' silhouette score:',silhouette_score(X,pred_ms))
res = homogeneity_completeness_v_measure(labels,pred_gmm)
print ('gaussian mixture model measures, homogeneity:',res[0],' completeness:',res[1],' v-measure:',res[2],' silhouette score:',silhouette_score(X,pred_gmm))
res = homogeneity_completeness_v_measure(labels,pred_h)
print ('hierarchical (ward) measures, homogeneity:',res[0],' completeness:',res[1],' v-measure:',res[2],' silhouette score:',silhouette_score(X,pred_h))
