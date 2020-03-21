import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def loadData(filePath):
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData , retCityName


if __name__ == '__main__':
    data,cityName = loadData('city.txt')
    # 生成数据散点图
    print(data)
    print(cityName)
    # 生成数据散点图
    dataSet=np.array(data)
    print(dataSet[:,0])
    plt.scatter(dataSet[:, 0], dataSet[:, 1], marker='o')
    plt.show()


    km = KMeans(n_clusters=4)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_,axis=1)

    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=label)
    plt.show()


    CityCluster=[[],[],[],[]]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])