import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Charger les données à partir des fichiers CSV nettoyés
chemin_fichier_2018 = '/home/yvanoide/iadev-python/EAU/Data_2018.csv'
chemin_fichier_2019 = '/home/yvanoide/iadev-python/EAU/Data_2019.csv'
chemin_fichier_2020 = '/home/yvanoide/iadev-python/EAU/Data_2020.csv'

donnees_2018 = pd.read_csv(chemin_fichier_2018)
donnees_2019 = pd.read_csv(chemin_fichier_2019)
donnees_2020 = pd.read_csv(chemin_fichier_2020)

# Concaténer les données de toutes les années
donnees_combinees = pd.concat([donnees_2018, donnees_2019, donnees_2020], ignore_index=True)

# Encodage de la colonne "Classification"
label_encoder = LabelEncoder()
donnees_combinees['Classification'] = label_encoder.fit_transform(donnees_combinees['Classification'])

# Supprimer les lignes avec valeurs manquantes
donnees_combinees = donnees_combinees.dropna()

# Diviser les données en ensembles d'entraînement et de test
X = donnees_combinees.drop('Classification', axis=1)
y = donnees_combinees['Classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les caractéristiques numériques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création et entraînement du modèle KNN
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_scaled, y_train)
y_pred_knn = knn_classifier.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Exactitude du modèle KNN:", accuracy_knn)

# Création et entraînement du modèle Random Forest
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train_scaled, y_train)
y_pred_rf = random_forest_classifier.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Exactitude du modèle Random Forest:", accuracy_rf)
