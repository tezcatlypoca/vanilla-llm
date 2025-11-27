from typing import List
from utils.mnist import *
import numpy as np

class NeuralNetwork:

    # Constantes
    # NB_PARAM = None

    # Init neural layer
    layers = []
    weights_mats = []
    bias_mats = []

    layers.append(np.zeros((1, 784)))
    layers.append(np.zeros((1, 16)))
    layers.append(np.zeros((1, 16)))
    layers.append(np.zeros((1, 10)))

    # Init weights and bias matrix
    # Pour np.dot(current, w) où current est (1, 784), w doit être (784, 16)
    weights_mats.append(np.random.randn(...) * 0.1)  # De 784 vers 16
    weights_mats.append(np.random.randn(...) * 0.1)    # De 16 vers 16
    weights_mats.append(np.random.randn(...) * 0.1)    # De 16 vers 10

    bias_mats.append(np.zeros((1, 16)))  # Biais pour la première couche cachée
    bias_mats.append(np.zeros((1, 16)))  # Biais pour la deuxième couche cachée
    bias_mats.append(np.zeros((1, 10)))  # Biais pour la couche de sortie

    training_images = [] # Contient les images 28x28 pour l'entrainement
    training_labels = [] # Contient les labels associés aux images d'entrainements

    def __init__(self):
        # Init des données d'entrainement
        temp = extract_training(None)
        self.training_images = temp[0]
        self.training_labels = temp[1]


    # Calcule la sortie pour une image en entrée donnée
    # param:
    # - entry => layers[0], un vect de dim(1, 784)
    # - weights => weights_mats, l'ensemble des matrice de poids
    # - bias => bias_mats, l'ensemble des matrice de biais
    def forward_prop(self, entry: np.array, weights: List, bias: List):
        # Retourne le vecteur de sortie de dim(1, 10)
        current = entry 
        for w, b in zip(weights, bias):
            current = self.sigmoid(np.dot(current, w) + b)
        return current


    def back_prop(self):
        pass
    
    # Calcule le cout pour une sortie d'entrainement
    def cost(self, output_model: [], output_target: []) -> float:
        cost: float = 0.0

        for i in range(len(output_model)):
            cost += np.square((output_model[i] - output_target[i]))
        return cost

    def training(self):
        pass

    # Convertit le label: training target, vers un vect pour calculer la fonction Cost du model
    def label_to_one_hot(self, label: int) -> []:
        one_hot = np.zeros(10) # renvoie un vecteur de dim(1, 10)
        one_hot[label] = 1.0
        return one_hot

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))