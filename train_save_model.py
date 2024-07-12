# train_save_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from joblib import dump
from sklearn.metrics import accuracy_score

# Membaca dataset
iris = pd.read_csv('iris.csv')  # ganti dengan path yang sesuai
iris.drop('Id', axis=1, inplace=True)

# Memisahkan atribut dan label
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

# Membagi data menjadi Training dan Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

# Menggunakan MLPClassifier untuk Neural Network
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=123)

# Melakukan training model
mlp_model.fit(X_train, y_train)

# Menyimpan model
dump(mlp_model, 'model/iris_model.joblib')


# Melakukan prediksi pada data uji
y_pred = mlp_model.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)
