import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

# 生成数据集
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本4个特征，
# 共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2,0.2]
X, y = make_blobs(n_samples=1000, n_features=2,
                  centers=[[2,0], [4,1], [2,2],[4,4]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2],
                  random_state =9)
# 生成数据散点图
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

#用Calinski-Harabasz Index评估的聚类分数
print("聚集两类的分数判断:",metrics.calinski_harabaz_score(X, y_pred))

from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

print("聚集三类的分数判断:",metrics.calinski_harabaz_score(X, y_pred))

from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

print("聚集四类的分数判断:",metrics.calinski_harabaz_score(X, y_pred))


from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=5, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

print("聚集五类的分数判断:",metrics.calinski_harabaz_score(X, y_pred))


from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=6, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

print("聚集六类的分数判断:",metrics.calinski_harabaz_score(X, y_pred))

from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=7, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

print("聚集七类的分数判断:",metrics.calinski_harabaz_score(X, y_pred))
