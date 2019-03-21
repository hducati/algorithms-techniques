import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    data = pd.read_csv('credit-card-clients.csv', header=1)
    data['BILL_TOTAL'] = data['BILL_AMT1'] + data['BILL_AMT2'] + data['BILL_AMT3'] + \
    data['BILL_AMT4'] +  data['BILL_AMT6'] + data['BILL_AMT6']
    x = data.iloc[:, [1, 25]].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.xlabel('Clusters')
    plt.ylabel('WCSS')
    
    kmeans = KMeans(n_clusters=4, random_state=0)
    predict = kmeans.fit_predict(x)
    
    plt.scatter(x[predict == 0, 0], x[predict == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(x[predict == 1, 0], x[predict == 1, 1], s=100, c='orange', label='Cluster 2')
    plt.scatter(x[predict == 2, 0], x[predict == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(x[predict == 3, 0], x[predict == 3, 1], s=100, c='blue', label='Cluster 4')
    plt.xlabel('Limite')
    plt.ylabel('Gastos')
    plt.legend()
    # add a new column with the values from the predict variable
    client_list = np.column_stack((data, predict))
    # sort values
    client_list = client_list[client_list[:, 26].argsort()]
    

if __name__ == '__main__':
    main()
