from evaluate_and_opti_ml_model import EvalOpti


class Node:

    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):

        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree(EvalOpti):

    def __init__(self, max_depth=None):

        self.max_depth = max_depth
        self.root = None

    def fit(self, x_train, y_train):

        # Construit l'arbre
        self.root = self._build_tree(
            x_train,
            y_train,
            depth=0
        )

    def _build_tree(self, x_train, y_train, depth):

        # Arrête la construction de l'arbre
        if (
            depth == self.max_depth
            or len(set(y_train)) == 1
        ):

            leaf_value = max(
                set(y_train),
                key=y_train.count
            )

            return Node(value=leaf_value)

        # Recherche le meilleur split
        best_feature_index, best_threshold = (
            self._best_split(x_train, y_train)
        )

        # Création des groupes gauche et droite
        x_left = []
        y_left = []

        x_right = []
        y_right = []

        for i in range(len(x_train)):

            if (
                x_train[i][best_feature_index]
                <= best_threshold
            ):

                x_left.append(x_train[i])
                y_left.append(y_train[i])

            else:

                x_right.append(x_train[i])
                y_right.append(y_train[i])

        # Construction récursive des branches
        left_node = self._build_tree(
            x_left,
            y_left,
            depth + 1
        )

        right_node = self._build_tree(
            x_right,
            y_right,
            depth + 1
        )

        # Retourne le nœud
        return Node(
            feature_index=best_feature_index,
            threshold=best_threshold,
            left=left_node,
            right=right_node
        )

    def _best_split(self, x_train, y_train):

        best_gini = float("inf")

        best_feature_index = None
        best_threshold = None

        # Parcourt chaque variable
        for feature_index in range(len(x_train[0])):

            feature_values = []

            for x in x_train:

                feature_values.append(
                    x[feature_index]
                )

            sorted_feature_values = sorted(
                set(feature_values)
            )
