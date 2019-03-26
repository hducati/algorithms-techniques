import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def main():
    data = pd.read_csv('census.csv')
    previsores = data.iloc[:, 0:14].values
    classe = data.iloc[:, 14].values
    
    # transformando variáveis categóricas em variávies numéricas
    list_prev = [1, 3, 5, 6, 7, 8, 9, 13]
    for i in list_prev:
        labelencoder_prev = LabelEncoder()
        previsores[:, i] = labelencoder_prev.fit_transform(previsores[:, i])

    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)
    
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = \
        train_test_split(previsores, classe, test_size=0.15, random_state=0)
    
    pca = PCA(n_components=6)
    previsores_treinamento = pca.fit_transform(previsores_treinamento)
    previsores_teste = pca.transform(previsores_teste)
    # valores do coeficiente
    components = pca.explained_variance_ratio_
    print(str(components))
    predict_list = []
    
    for i in range(0, 30):
        classifier = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=i)
        classifier.fit(previsores_treinamento, classe_treinamento)
        predict = classifier.predict(previsores_teste)
        accuracy = accuracy_score(classe_teste, predict)
        predict_list.append(accuracy)

    print(max(predict_list))
