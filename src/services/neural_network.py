from typing import List
from utils.mnist import *
import numpy as np

class NeuralNetwork:

    # Constantes
    # NB_PARAM = None
    MAX_GRAD_NORM = 1.0                    

    learning_rate: float = 0.1
    initial_learning_rate: float = 0.01  # Learning rate initial
    min_learning_rate: float = 0.0001  # Learning rate minimum
    learning_rate_decay: float = 0.95  # Facteur de décroissance par époque
    patience: int = 5  # Nombre d'époques sans amélioration avant de réduire le LR

    def __init__(self):
        # Init neural layer
        self.layers = []
        self.z_values = []
        self.weights_mats = []
        self.bias_mats = []

        self.layers.append(np.zeros((1, 784)))
        self.layers.append(np.zeros((1, 256)))  # Couche cachée 1 : 256 neurones
        self.layers.append(np.zeros((1, 128)))  # Couche cachée 2 : 128 neurones
        self.layers.append(np.zeros((1, 10)))

        # Init weights and bias matrix avec initialisation de Xavier/Glorot
        # Variance = 2 / (n_input + n_output) pour la sigmoïde
        # Architecture : 784 → 256 → 128 → 10
        self.weights_mats.append(np.random.randn(784, 256) * np.sqrt(2.0 / (784 + 256)))  # De 784 vers 256
        self.weights_mats.append(np.random.randn(256, 128) * np.sqrt(2.0 / (256 + 128)))    # De 256 vers 128
        self.weights_mats.append(np.random.randn(128, 10) * np.sqrt(2.0 / (128 + 10)))    # De 128 vers 10

        self.bias_mats.append(np.zeros((1, 256)))  # Biais pour la première couche cachée (256 neurones)
        self.bias_mats.append(np.zeros((1, 128)))  # Biais pour la deuxième couche cachée (128 neurones)
        self.bias_mats.append(np.zeros((1, 10)))  # Biais pour la couche de sortie

        # Init des données d'entrainement
        temp = extract_training(None)
        self.training_images = temp[0]
        self.training_labels = temp[1]


    # Calcule la sortie pour une image en entrée donnée
    def forward_prop(self):
        self.z_values = []
        # Layer 0 -> Layer 1
        z1 = np.dot(self.layers[0], self.weights_mats[0]) + self.bias_mats[0]
        self.z_values.append(z1)
        self.layers[1] = self.sigmoid(z1)

        # Layer 1 -> Layer 2
        z2 = np.dot(self.layers[1], self.weights_mats[1]) + self.bias_mats[1]
        self.z_values.append(z2)
        self.layers[2] = self.sigmoid(z2)

        # Layer 2 -> Layer 3
        z3 = np.dot(self.layers[2], self.weights_mats[2]) + self.bias_mats[2]
        self.z_values.append(z3)
        self.layers[3] = self.sigmoid(z3)


    def back_prop(self, output_target: np.array):
        output_model = self.layers[3]
        derivated_cost_output = 2 * (output_model - output_target)

        # calcule de l'erreur de la couche 3 sortie du model
        error_output = derivated_cost_output * self.derivated_sigmoid(self.z_values[2])

        # calcule du gradient pour la derniere couche (2 -> 3)
        grad_weights_2 = np.dot(self.layers[2].T, error_output)
        grad_bias_2 = error_output
        # propagation de l'erreur vers la couche 2
        error_layer_2 = np.dot(error_output, self.weights_mats[2].T) * self.derivated_sigmoid(self.z_values[1])

        # calcule du gradient pour la 2e couche (1 -> 2)        
        grad_weights_1 = np.dot(self.layers[1].T, error_layer_2)
        grad_bias_1 = error_layer_2
        # propagation de l'erreur vers la couche 1
        error_layer_1 = np.dot(error_layer_2, self.weights_mats[1].T) * self.derivated_sigmoid(self.z_values[0])

        # calcule du gradient pour la 1e couche (0 -> 1)
        grad_weights_0 = np.dot(self.layers[0].T, error_layer_1)
        grad_bias_0 = error_layer_1

        grad_weights = [grad_weights_0, grad_weights_1, grad_weights_2]
        grad_bias = [grad_bias_0, grad_bias_1, grad_bias_2]

        return grad_weights, grad_bias
        
    def update_weights(self, grad_weight: List):
        for i in range(len(self.weights_mats)):
            self.weights_mats[i] = self.weights_mats[i] - (self.learning_rate * grad_weight[i])

    def update_bias(self, grad_bias: List):
        for i in range(len(self.bias_mats)):
            self.bias_mats[i] = self.bias_mats[i] - (self.learning_rate * grad_bias[i])
    
    # Calcule le cout pour une sortie d'entrainement
    def cost(self, output_model, output_target) -> float:
        # Calculer la somme des carrés des différences
        # output_model et output_target sont de forme (1, 10)
        cost = np.sum(np.square(output_model - output_target))
        return float(cost)
    
    # Retourne le coup moyen d'une liste de coup pour un batch donné
    # Param:
    #   - cost ->  list de cost
    #   - size_costs -> la taille du tableau de coup afin de faire une moyenne
    def calculate_average_cost(self, costs: List, size_costs: int) -> float:
        res: float = 0.0

        for cost in costs:
            res += cost
        return res/size_costs

    def training(self, epochs: int = 10):
        """
        Entraîne le modèle sur les données d'entraînement avec learning rate adaptatif.
        
        Args:
            epochs: Nombre d'époques (passes complètes sur les données)
        """
        batch_size = 1000
        total_batches = (len(self.training_images) + batch_size - 1) // batch_size
        
        # Réinitialiser le learning rate et les variables de suivi
        self.learning_rate = self.initial_learning_rate
        best_cost = float('inf')
        epochs_without_improvement = 0
        
        print(f"Démarrage de l'entraînement sur {epochs} époques ({len(self.training_images)} images, {total_batches} batches/époque)")
        print(f"Learning rate initial: {self.learning_rate:.6f}\n")
        
        epoch_costs = []  # Pour stocker le coût moyen de chaque époque
        best_cost = float('inf')  # Meilleur coût observé
        epochs_without_improvement = 0  # Compteur d'époques sans amélioration
        
        # Boucle sur les époques
        for epoch in range(epochs):
            epoch_batch_costs = []
            
            # Parcourir les images par batch
            for batch_start in range(0, len(self.training_images), batch_size):
                batch_end = min(batch_start + batch_size, len(self.training_images))
                batch_costs = []
                
                # Initialiser les accumulateurs de gradients
                batch_grad_weights = [np.zeros_like(w) for w in self.weights_mats]
                batch_grad_bias = [np.zeros_like(b) for b in self.bias_mats]
                
                # Pour chaque image du batch
                for i in range(batch_start, batch_end):
                    # Préparer l'image (convertir en numpy array de forme (1, 784))
                    self.layers[0] = np.array(self.training_images[i]).reshape(1, 784)
                    
                    # Convertir le label en one-hot
                    target_label = self.label_to_one_hot(self.training_labels[i])
                    
                    # Forward propagation
                    self.forward_prop()
                    
                    # Calculer le coût (pour monitoring)
                    cost_value = self.cost(self.layers[3], target_label)
                    batch_costs.append(cost_value)
                    
                    # Backward propagation (récupérer les gradients)
                    grad_weights, grad_bias = self.back_prop(target_label)
                    
                    # Accumuler les gradients
                    for j in range(len(batch_grad_weights)):
                        batch_grad_weights[j] += grad_weights[j]
                        batch_grad_bias[j] += grad_bias[j]
                
                # Moyenner les gradients sur le batch
                actual_batch_size = batch_end - batch_start
                for j in range(len(batch_grad_weights)):
                    batch_grad_weights[j] /= actual_batch_size
                    batch_grad_bias[j] /= actual_batch_size
                
                # Mettre à jour les poids et biais
                self.update_weights(batch_grad_weights)
                self.update_bias(batch_grad_bias)
                
                # Calculer le coût moyen du batch
                avg_cost = self.calculate_average_cost(batch_costs, len(batch_costs))
                epoch_batch_costs.append(avg_cost)
                
                # Afficher la progression (réécrit sur la même ligne)
                batch_num = batch_start // batch_size + 1
                progress = (batch_num / total_batches) * 100
                print(f"\rÉpoque {epoch + 1}/{epochs} | Batch {batch_num}/{total_batches} ({progress:.1f}%) | Coût: {avg_cost:.4f}", end='', flush=True)
            
            # Calculer le coût moyen de l'époque
            epoch_avg_cost = self.calculate_average_cost(epoch_batch_costs, len(epoch_batch_costs))
            epoch_costs.append(epoch_avg_cost)
            
            # Learning rate adaptatif : vérifier si amélioration
            if epoch_avg_cost < best_cost:
                best_cost = epoch_avg_cost
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Réduire le learning rate si pas d'amélioration depuis 'patience' époques
            if epochs_without_improvement >= self.patience:
                old_lr = self.learning_rate
                self.learning_rate = max(self.learning_rate * self.learning_rate_decay, self.min_learning_rate)
                if self.learning_rate != old_lr:
                    print(f"\rÉpoque {epoch + 1}/{epochs} terminée | Coût: {epoch_avg_cost:.4f} | LR réduit: {old_lr:.6f} → {self.learning_rate:.6f} {' ' * 10}")
                    epochs_without_improvement = 0  # Reset après réduction
                else:
                    print(f"\rÉpoque {epoch + 1}/{epochs} terminée | Coût: {epoch_avg_cost:.4f} | LR: {self.learning_rate:.6f} (min) {' ' * 20}")
            else:
                # Décroissance exponentielle normale
                if epoch > 0:  # Ne pas réduire à la première époque
                    self.learning_rate = max(self.learning_rate * self.learning_rate_decay, self.min_learning_rate)
                print(f"\rÉpoque {epoch + 1}/{epochs} terminée | Coût: {epoch_avg_cost:.4f} | LR: {self.learning_rate:.6f} {' ' * 30}")
        
        # Résumé final de l'entraînement
        print(f"\n{'='*60}")
        print(f"ENTRAÎNEMENT TERMINÉ")
        print(f"{'='*60}")
        print(f"Nombre d'époques: {epochs}")
        print(f"Coût initial (époque 1): {epoch_costs[0]:.4f}")
        print(f"Coût final (époque {epochs}): {epoch_costs[-1]:.4f}")
        print(f"Amélioration: {((epoch_costs[0] - epoch_costs[-1]) / epoch_costs[0] * 100):.2f}%")
        print(f"{'='*60}\n")

    def testing(self):
        """
        Teste le modèle sur les données de test et calcule la précision.
        """
        temp = extract_testing(None)
        test_images = format_images(temp[0])
        test_labels = temp[1]
        
        correct_predictions = 0
        total_images = len(test_images)
        
        print(f"Test en cours sur {total_images} images...", end='', flush=True)
        
        # Tester chaque image
        for i in range(total_images):
            # Préparer l'image (convertir en numpy array de forme (1, 784))
            self.layers[0] = np.array(test_images[i]).reshape(1, 784)
            
            # Forward propagation
            self.forward_prop()
            
            # Obtenir la prédiction (l'indice du neurone avec la valeur la plus élevée)
            prediction = np.argmax(self.layers[3])
            
            # Comparer avec le label réel
            true_label = test_labels[i]
            
            if prediction == true_label:
                correct_predictions += 1
            
            # Afficher la progression tous les 1000 images
            if (i + 1) % 1000 == 0 or i == total_images - 1:
                progress = ((i + 1) / total_images) * 100
                print(f"\rTest en cours... {i + 1}/{total_images} ({progress:.1f}%)", end='', flush=True)
        
        # Calculer la précision
        accuracy = (correct_predictions / total_images) * 100
        
        # Résumé du test
        print(f"\r{' ' * 60}")  # Effacer la ligne de progression
        print(f"{'='*60}")
        print(f"RÉSULTATS DU TEST")
        print(f"{'='*60}")
        print(f"Images testées: {total_images}")
        print(f"Prédictions correctes: {correct_predictions}")
        print(f"Prédictions incorrectes: {total_images - correct_predictions}")
        print(f"PRÉCISION: {accuracy:.2f}%")
        print(f"{'='*60}\n")
        
        return accuracy

    # Convertit le label: training target, vers un vect pour calculer la fonction Cost du model
    def label_to_one_hot(self, label: int) -> np.array:
        one_hot = np.zeros((1, 10)) # renvoie un vecteur de dim(1, 10)
        one_hot[0, label] = 1.0
        return one_hot
    
    # Fonction sigmoid qui compresse la droite des réelles entre 0 et 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # retourne la dérivié de la sigmoid
    def derivated_sigmoid(self, z) -> np.array:
        s = self.sigmoid(z)
        return s * (1 - s)