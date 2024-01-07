import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#baca data
songs = pd.read_csv("Top-50-musicality-global.csv")

def eda():
    pop_scores = []

    for i in songs["Popularity"]:
        if i > 80:
            pop_scores.append("Very popular")
        elif i > 60:
            pop_scores.append("Popular")
        elif i > 40:
            pop_scores.append("Middle popularity")
        else:
            pop_scores.append("Unpopular")

    songs["Popularity rank"] = pop_scores

    print(songs[["Energy", "Liveness", "Acousticness", "Popularity", "Popularity rank"]].tail(30))

def model():
    X = songs[["Energy", "Liveness", "Acousticness"]]
    y = songs["Popularity rank"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17, stratify=y)

    n_range = np.arange(3, 20)

    train_accuracies = []
    test_accuracies = []

    for z in n_range:
        knn = KNeighborsClassifier(n_neighbors=z)
        knn.fit(X_train, y_train)
        train_accuracies.append(knn.score(X_train, y_train))
        test_accuracies.append(knn.score(X_test, y_test))


def sample():
    X = songs[["Energy", "Liveness", "Acousticness"]]
    y = songs["Popularity rank"]

    knn = KNeighborsClassifier(n_neighbors=5)  # contoh dengan 5 tetangga
    knn.fit(X, y)

    # Buat DataFrame sample untuk prediksi
    sample = pd.DataFrame({'Energy': [Energy], 'Liveness': [Liveness], 'Acousticness': [Acousticness]})

    # Lakukan prediksi dengan model yang sudah dilatih
    predicted_rank = knn.predict(sample)

    print(f'Peringkat prediksi: {predicted_rank[0]}')

    return predicted_rank[0]

st.title("Soundly")
Energy = st.number_input("masukan Energy (0 - 0.9):")
Liveness = st.number_input("masukan Liveness (0 - 0.9) : ")
Acousticness = st.number_input("masukan Acousticness (0 - 0.9) : ")
if st.button("predict"):
    eda()
    model()
    sample()
    st.write(sample())
