from src.neural_network import NeuralNetwork
import torch, os, datetime

def get_metadata(llm: NeuralNetwork):
    metadata = {}

    # Date de la sauvegarde
    metadata["horodatage"] = datetime.datetime()
    # Nombre de hidden layers
    metadata['nb_layer'] = len(llm.layers) -2
    # Récupération de la taille de chaque layer (entrée, hidden layers, sortie)
    metadata['layer_size'] = {
        'layer_0': llm.layers[0].shape,
        'layer_1': llm.layers[1].shape,
        'layer_2': llm.layers[2].shape,
        'layer_3': llm.layers[3].shape,
    }
    # Nombre d'epochs effectuées lors de l'entrainement
    metadata['epochs'] = llm.MAX_EPOCHS
    # Taille des batchs utilisés
    metadata['batch_size'] = llm.BATCH_SIZE
    # Durée de l'entrainement
    metadata['training_time'] = llm.total_time