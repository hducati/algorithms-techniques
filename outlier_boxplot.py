import pandas as pd
import matplotlib.pyplot as plt
    

def main():
    base = pd.read_csv('credit-data.csv')
    base = base.dropna()
    # verificando outliers
    plt.boxplot(base.iloc[:, 2])
    outliers_age = base[(base.age < 0)]
    
    plt.boxplot(base.iloc[:, 3])
    outliers_loan = base[(base.loan > 13400)]