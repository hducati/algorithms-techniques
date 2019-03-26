import pandas as pd
from pyod.models.knn import KNN


def main():
    data = pd.read_csv('credit-data.csv')
    data = data.dropna()
    
    detector = KNN()
    detector.fit(data.iloc[:, 1:4])
    
    # detector.labels_ irá retornar 0 caso não tenha outlier e 1 caso tenha
    predict = detector.labels_
    predict_trust = detector.decision_scores_
    print(str(predict))
    print(str(predict_trust))
    
    outlier = [i for i in range(len(predict)) if predict[i] == 1]
    print(str(outlier))
    
    df_outlier = data.iloc[outlier, :]
    print(df_outlier)
    
    
if __name__ == '__main__':
    main()