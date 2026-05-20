import math
 
 
class KnnFs:
    def __init__(self, n_neighbors=3):
        # je stocke le nombre de voisins
        self.n_neighbors = n_neighbors
        self.train_features = None
        self.train_labels = None
 
    def fit(self, train_features, train_labels):
        # je mémorise les données d'entraînement
        self.train_features = train_features
        self.train_labels = train_labels
 
    def euclidean_distance(self, point_a, point_b):
        # je calcule la distance entre deux points
        distance = 0
        for value_a, value_b in zip(point_a, point_b):
            distance += (value_a - value_b) ** 2
        return math.sqrt(distance)
 
    def predict(self, sample):
        # je prédit la classe d'un seul point
        distances = []
        for sample_index, train_sample in enumerate(self.train_features):
            distance = self.euclidean_distance(train_sample, sample)
            distances.append((distance, self.train_labels[sample_index]))
 
        # je trie par distance et je prends les k plus proches
        distances.sort(key=lambda item: item[0])
        neighbors = distances[:self.n_neighbors]
 
        # vote majoritaire
        labels = [label for _, label in neighbors]
        prediction = max(labels, key=labels.count)
        return prediction
 
    def evaluate(self, test_features, test_labels):
        # je calcule le f1 score
        true_positive = 0
        false_positive = 0
        false_negative = 0
 
        for sample, actual_label in zip(test_features, test_labels):
            prediction = self.predict(sample)
            if prediction == 1 and actual_label == 1:
                true_positive += 1
            elif prediction == 1 and actual_label != 1:
                false_positive += 1
            elif prediction != 1 and actual_label == 1:
                false_negative += 1
 
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
 
        if precision + recall == 0:
            return 0
 
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score
 
    def cross_validation(self, features, labels, k_folds):
        # je fais une validation croisée
        fold_size = len(features) // k_folds
        scores = []
 
        for fold_index in range(k_folds):
            start = fold_index * fold_size
            end = start + fold_size
 
            test_features = features[start:end]
            test_labels = labels[start:end]
            train_features = features[:start] + features[end:]
            train_labels = labels[:start] + labels[end:]
 
            self.fit(train_features, train_labels)
            score = self.evaluate(test_features, test_labels)
            scores.append(score)
 
        mean_score = sum(scores) / len(scores)
        return mean_score
 
    def grid_search(self, features, labels, k_folds):
        # je cherche le meilleur k
        best_k = 1
        best_score = 0
 
        for k_value in range(1, 20, 2):
            self.n_neighbors = k_value
            score = self.cross_validation(features, labels, k_folds)
            print(f"k = {k_value} | F1-score = {score:.4f}")
 
            if score > best_score:
                best_score = score
                best_k = k_value
 
        print("\nMeilleur k :", best_k)
        print("Meilleur score :", round(best_score, 4))
        self.n_neighbors = best_k
        return best_k, best_score
