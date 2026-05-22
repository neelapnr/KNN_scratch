class EvalOpti:

    def evaluate(self, x_test, y_test):

        # Récupère toutes les classes présentes
        labels = list(set(y_test))

        # Liste des prédictions
        y_pred = []

        # Fait une prédiction pour chaque donnée
        for x in x_test:
            prediction = self.predict(x)
            y_pred.append(prediction)

        # Liste des scores F1
        f1_scores = []

        # Calcul du F1-score pour chaque classe
        for label in labels:

            TP = 0
            FP = 0
            FN = 0

            for i in range(len(y_pred)):

                # Bonne prédiction
                if y_pred[i] == label and y_test[i] == label:
                    TP += 1

                # Le modèle prédit la classe alors que c'est faux
                elif y_pred[i] == label and y_test[i] != label:
                    FP += 1

                # Le modèle rate la bonne classe
                elif y_pred[i] != label and y_test[i] == label:
                    FN += 1

            # Calcul de la précision
            precision = TP / (TP + FP)

            # Calcul du recall
            recall = TP / (TP + FN)

            # Calcul du F1-score
            f1 = (2 * precision * recall) / (precision + recall)

            f1_scores.append(f1)

        # Moyenne des F1-scores
        f1_macro = sum(f1_scores) / len(f1_scores)

        return {
            "f1_macro": f1_macro
        }

    def grid_search(
        self,
        setting_model,
        setting_values,
        x_test,
        y_test
    ):

        best_setting_model = None
        best_f1_score_macro = 0

        # Teste chaque valeur du paramètre
        for value in setting_values:

            # Change la valeur du paramètre
            setattr(self, setting_model, value)

            # Évalue le modèle
            metrics = self.evaluate(x_test, y_test)

            # Garde le meilleur score
            if metrics["f1_macro"] > best_f1_score_macro:

                best_f1_score_macro = metrics["f1_macro"]
                best_setting_model = value

        return {
            "best_setting_model": best_setting_model,
            "best_f1_score_macro": best_f1_score_macro
        }
