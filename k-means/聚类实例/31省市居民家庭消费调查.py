import numpy as np
from sklearn.cluster import KMeans
import sklearn.cluster as skc
import matplotlib.pyplot as plt


def loadData(filePath):
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName


if __name__ == '__main__':
    data, cityName = loadData('city.txt')
    # 查看数据属性
    print(np.array(data).shape)
    print(cityName)

    fig, ax = plt.subplots()
    ax.plot(data[:], 'o')
    plt.ylabel("元",fontproperties="SimHei")
    plt.xlabel("属性",fontproperties="SimHei")
    # plt.title(cityName[:],fontproperties="SimHei")
    plt.show()






    print("k-means聚类:")
    # 利用k-means聚类
    km = KMeans(n_clusters=4)
    label = km.fit_predict(data)

    # 输出分类的标签，每一个数据集都标记好了其分类的簇编号
    print(label)
    expenses = np.sum(km.cluster_centers_, axis=1)

    CityCluster = [[], [], [], []]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])

    print("DBSCAN聚类")
    db = skc.DBSCAN(eps=0.1, min_samples=2).fit(data)
    labels = db.labels_
    print(labels)