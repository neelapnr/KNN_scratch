from evaluate_and_opti_ml_model import EvalOpti


class KNN(EvalOpti):

    def __init__(self, n_neighbors=None):

        self.n_neighbors = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):

        # Sauvegarde les données d'entraînement
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x):

        # Calcul des distances
        distances = []

        for x_train_point in self.x_train:

            distance = 0

            for i in range(len(x)):

                distance += (
                    x[i] - x_train_point[i]
                ) ** 2

            distance = distance ** 0.5

            distances.append(distance)

        # Trie les distances
        sorted_distances = sorted(
            enumerate(distances),
            key=lambda distance: distance[1]
        )

        # Récupère les voisins les plus proches
        nearest_neighbors = sorted_distances[:self.n_neighbors]

        # Récupère les labels des voisins
        nearest_labels = []

        for neighbor in nearest_neighbors:

            neighbor_index = neighbor[0]

            nearest_labels.append(
                self.y_train[neighbor_index]
            )

        # Compte les labels
        label_distribution = {}

        for label in nearest_labels:

            if label in label_distribution:

                label_distribution[label] += 1

            else:

                label_distribution[label] = 1

        # Retourne le label le plus fréquent
        return max(
            label_distribution,
            key=lambda label: label_distribution[label]
        )
