import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



if __name__ == '__main__':
    data=[[3,4],
          [3,6],
          [7,3],
          [4,7],
          [3,8],
          [8,5],
          [4,5],
          [4,1],
          [7,4],
          [5,5]];
    cityName=['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'];


    km = KMeans(n_clusters=3)
    label = km.fit_predict(data)
    print(km.cluster_centers_)
    expenses = np.sum(km.cluster_centers_, axis=1)

    CityCluster = [[], [] ,[]]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])