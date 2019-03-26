import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('credit-data.csv')
    data = data.dropna()
    data.loc[data.age < 0, 'age'] = 40.92
    
    # verificando outliers dos registros
    plt.scatter(data.iloc[:, 1], data.iloc[:, 2])
    plt.scatter(data.iloc[:, 1], data.iloc[:, 3])
    plt.scatter(data.iloc[:, 2], data.iloc[:, 3])
    
    
if __name__ == '__main__':
    main()