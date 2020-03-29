import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

###################################上网开始时间的聚类##########################################
# 数据集的处理
mac2id = dict()
onlinetimes = []


f = open('TestData.txt', encoding='utf-8')
for line in f:
    mac = line.split(',')[2]
    onlinetime = int(line.split(',')[6])
    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])
    if mac not in mac2id:
        mac2id[mac] = len(onlinetimes)
        onlinetimes.append((starttime, onlinetime))
    else:
        onlinetimes[mac2id[mac]] = [(starttime, onlinetime)]
print(mac2id)
print(onlinetimes)

# 生成开始上网时间和在线时长的数据散点图
dataSet=np.array(onlinetimes)
fig, ax =plt.subplots()
ax.plot(dataSet[:, 0], dataSet[:, 1], 'o')
plt.xlabel("starttime")
plt.ylabel("onlinetime")
plt.show()



# 利于DBSCAN对数据散点图进行基于密度的聚类
real_X = np.array(onlinetimes).reshape((-1, 2))
X = real_X[:, 0:1]
db = skc.DBSCAN(eps=0.01, min_samples=20).fit(X)
labels = db.labels_


# 输出每个数据的簇标签，-1则表示为噪声点
print('Labels:')
print(labels)

raito = len(labels[labels[:] == -1]) / len(labels)
print('Noise raito:', format(raito, '.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

for i in range(n_clusters_):
    print('Cluster ', i, ':')
    print(list(X[labels == i].flatten()))

plt.hist(X, 24)
plt.show()
###################################上网开始时间的聚类##########################################