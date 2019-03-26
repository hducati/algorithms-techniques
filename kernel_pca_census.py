import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier


def main():
    data = pd.read_csv('census.csv')
    previsores = data.iloc[:, 0:14].values
    classe = data.iloc[:, 14].values
    
    list_prev = [1, 3, 5, 6, 7, 8, 9, 13]
    for i in list_prev:
        labelencoder_prev = LabelEncoder()
        previsores[:, i] = labelencoder_prev.fit_transform(previsores[:, i])

    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)
    
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = \
        train_test_split(previsores, classe, test_size=0.15, random_state=0)
    # utiliza-se kernelPCA quando os problemas não são
    # linearmente separáveis
    kpca = KernelPCA(n_components=6, kernel='rbf')
    previsores_treinamento = kpca.fit_transform(previsores_treinamento)
    previsores_teste = kpca.transform(previsores_teste)

    classifier = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
    classifier.fit(previsores_treinamento, classe_treinamento)
    predict = classifier.predict(previsores_teste)
    
    accuracy = accuracy_score(classe_teste, predict)
    print(str(accuracy))


if __name__ == '__main__':
    main()