import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def main():
    data = pd.read_csv('credit-card-clients.csv', header=1)
    data['BILL_TOTAL'] = data['BILL_AMT1'] + data['BILL_AMT2'] + data['BILL_AMT3'] + \
    data['BILL_AMT4'] +  data['BILL_AMT6'] + data['BILL_AMT6']
    
    x = data.iloc[:, [1, 25]].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    dbscan = DBSCAN(eps=0.35, min_samples=8)
    predict = dbscan.fit_predict(x)
    unique, n = np.unique(predict, return_counts=True)
    
    plt.scatter(x[predict == 0, 0], x[predict == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(x[predict == 1, 0], x[predict == 1, 1], s=100, c='orange', label='Cluster 2')
    plt.scatter(x[predict == 2, 0], x[predict == 2, 1], s=100, c='blue', label='Cluster 3')
    plt.xlabel='Idade'
    plt.ylabel('Gasto')
    plt.legend()
    
    client_list = np.column_stack((data, predict))
    client_list = client_list[client_list[:, 26].argsort()]
    
if __name__ == '__main__':
    main()