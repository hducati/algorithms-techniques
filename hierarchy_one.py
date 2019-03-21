import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

x = [20, 27, 21, 37, 46, 53, 55, 47, 52, 32, 39, 41, 39, 48, 48]
y = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500]

base = np.array([[20, 1000], [27, 1200], [21, 2900], [37, 1850], [46, 900], [53, 950]])
scaler = StandardScaler()
base = scaler.fit_transform(base)

dendograma = dendrogram(linkage(base, method='ward'))
plt.title('Dendograma')
plt.xlabel('Pessoas')
plt.ylabel('Distancia Euclidiana')
    
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
predict = hc.fit_predict(base)

plt.scatter(base[predict == 0, 0], base[predict == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(base[predict == 1, 0], base[predict == 1, 1], s=100, c='green', label='Cluster 2')
plt.scatter(base[predict == 2, 0], base[predict == 2, 1], s=100, c='blue', label='Cluster 3')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.legend()