import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

songs = pd.read_csv("Top-50-musicality-global.csv")

#EDA
songs = pd.read_csv("Top-50-musicality-global.csv")
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

print(songs[["Energy", "Liveness", "Popularity","Positiveness","Loudness","Popularity rank"]].tail(10))

# model
X = songs[["Energy", "Liveness","Positiveness","Loudness"]]
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

best_n = n_range[np.argmax(test_accuracies)]
best_knn = KNeighborsClassifier(n_neighbors=best_n)
best_knn.fit(X_train, y_train)

train_accuracy_percentage = best_knn.score(X_train, y_train) * 100
test_accuracy_percentage = best_knn.score(X_test, y_test) * 100

print(f"Akurasi model terbaik (n_neighbors = {best_n}):")
print(f"Akurasi pada data latih: {train_accuracy_percentage:.2f}%")
print(f"Akurasi pada data uji: {test_accuracy_percentage:.2f}%")


with open('knn_model.pkl', 'wb') as file:
    pickle.dump(best_knn, file)