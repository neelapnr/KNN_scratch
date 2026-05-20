from dataloader import load_normalized_data
from knn import KnnFs
 
normalized_features, labels, scaler = load_normalized_data("bienetre.csv")
 
# conversion en listes 
features_list = normalized_features.tolist()
labels_list = labels.tolist()
 

knn_model = KnnFs(n_neighbors=3)
 
# recherche du meilleur k avec cross validation
best_k, best_score = knn_model.grid_search(features_list, labels_list, k_folds=5)
