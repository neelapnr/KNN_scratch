from dataloader import load_normalized_data
from knn import KNN

# chargement des données
normalized_features, labels, scaler = load_normalized_data("bienetre.csv")

# conversion en listes
features_list = normalized_features.tolist()
labels_list = labels.tolist()

# KNN
print("=" * 40)
print("KNN")
print("=" * 40)
knn_model = KNN(n_neighbors=3)
knn_model.fit(features_list, labels_list)
knn_result = knn_model.grid_search(
    "n_neighbors",
    range(1, 20, 2),
    features_list,
    labels_list
)
print(knn_result)
