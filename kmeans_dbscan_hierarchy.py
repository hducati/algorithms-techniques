import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn import datasets


def main():
    x, y = datasets.make_moons(n_samples=1500, noise=0.089)
    plt.scatter(x[:, 0], x[:, 1], s=5)
    
    cores = np.array(['red', 'blue'])
    
    kmeans = KMeans(n_clusters=2)
    predict = kmeans.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], s=5, color=cores[predict])
    
    hc = AgglomerativeClustering()
    predict = hc.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], s=5, color=cores[predict])
    
    dbscan = DBSCAN(eps=0.1)
    predict = dbscan.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], s=5, color=cores[predict])
    
    
if __name__ == '__main__':
    main()