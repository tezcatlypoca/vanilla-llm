from utils.mnist import extract_training, extract_testing
from threading import Thread
import numpy as np
import torch, time

device = []
device.append(torch.device("cpu"))
if torch.cuda.is_available():
    nb_devices_available = torch.cuda.device_count()
    print(f"Nombre de GPU disponible: {nb_devices_available}")
    for i in range(nb_devices_available):
        print(f"\tNom du GPU {i}: {torch.cuda.get_device_name(i)}")
        device.append(torch.device(f"cuda:{i}"))
    # device = [torch.device("cuda:0"), torch.device("cuda:1")]
else:
    print("Pas de GPU disponible !")

class NeuralNetwork():

    INITIAL_RATE = 0.01
    DECAY_RATE = 0.95

    output_test = []

    # Params:
    #   - Layering: 
    #       * la taille du tableau représente le nombre de couche du NN
    #       * chaque représente le nombre de neurones de la couche
    def __init__(self):
        self.relu = torch.nn.ReLU() # Var contenant la fonction de ReLU, à utiliser comme une fonction
        self.learning_rate = self.INITIAL_RATE

        # init de toutes les matrices
        self.init_layers()
        self.init_weights()
        self.init_bias()
        self.init_z_values()

        # init des images et labels d'entrainement
        self.training_images, self.training_labels, _ = extract_training()

        self.testing_images, self.testing_labels, _ = extract_testing()

    def run(self, mode: str):
        BATCH_SIZE = 500
        EPOCHS = 10
        if mode == "TRAINING":

            print("Début de l'entrainement")
            start_time = time.time()

            for e_index in range(EPOCHS):
                for i_index in range(0, len(self.training_images), BATCH_SIZE):
                    batch_0 = self.training_images[i_index: i_index+(BATCH_SIZE//2)] # Première moitié du batch
                    batch_1 = self.training_images[i_index+(BATCH_SIZE//2): i_index+BATCH_SIZE] # Deuxième moitié du batch

                    labels_0 = self.training_labels[i_index: i_index+(BATCH_SIZE//2)]
                    labels_1 = self.training_labels[i_index+(BATCH_SIZE//2): i_index+BATCH_SIZE]

                    thread_gpu_0 = Thread(target=self.training, args=(e_index, batch_0, labels_0), kwargs={"device_id": 1}) # Utilisation du GPU 0
                    thread_gpu_1 = Thread(target=self.training, args=(e_index, batch_1, labels_1), kwargs={"device_id": 2}) # Utilisation du GPU 1

                    thread_gpu_0.start()
                    thread_gpu_1.start()

                    thread_gpu_0.join()
                    thread_gpu_1.join()

            print(f"Fin de l'entrainement en {(time.time() - start_time)/60:.1f}")
        elif mode == "TESTING":
            self.output_test = []
            print("Début du test")
            start_time = time.time()

            for i_index in range(0, len(self.testing_images), BATCH_SIZE):
                batch_0 = self.testing_images[i_index: i_index+(BATCH_SIZE//2)] # Première moitié du batch
                batch_1 = self.testing_images[i_index+(BATCH_SIZE//2): i_index+BATCH_SIZE] # Deuxième moitié du batch

                thread_gpu_0 = Thread(target=self.testing, args=(batch_0, 1)) # Utilisation du GPU 0
                thread_gpu_1 = Thread(target=self.testing, args=(batch_1, 2)) # Utilisation du GPU 1

                thread_gpu_0.start()
                thread_gpu_1.start()

                thread_gpu_0.join()
                thread_gpu_1.join()
            
            print(f"Fin du test en {(time.time() - start_time)/60:.1f}")

            # Matrice représentant l'ensemble des sortie du test dim(total_size, 10)
            if len(self.output_test) > 0:
                self.output_test = [o.to(device[0]) for o in self.output_test]
                output_model_test = torch.cat(self.output_test, dim=0)
                output_target_test = self.label_to_vect(self.testing_labels)

                precision = self._compute_precision(output_model_test, output_target_test)
                print(f"Précision de {precision:.2f}")
    # END FUNCTION

    def training(self, epoch: int, batch: list, labels: list, device_id: int = 0):
        batch_tensor = torch.tensor(batch)
        output_model = self.forward_prop(batch_tensor, device_id)
        output_target = self.label_to_vect(labels)

        gradients = self.back_prop(output_model, output_target, device_id)
        self._compute_gradient(gradients, epoch, device_id)        

    def testing(self, batch: list, device_id: int = 0):
        self.output_test.append(self.forward_prop(batch, device_id))     

    def forward_prop(self, batch: torch.Tensor | list, gpu_id: int = 0):
        if isinstance(batch, torch.Tensor):
            layer_0 = batch.to(device[gpu_id])
            if len(layer_0.shape) == 1:
                layer_0 = layer_0.unsqueeze(0)
        elif isinstance(batch, list):
            layer_0 = torch.tensor(batch, device=device[gpu_id], dtype=torch.float32)
            if len(layer_0.shape) == 1:
                layer_0 = layer_0.unsqueeze(0)
        else:
            raise ValueError("Le batch doit être de type torch.Tensor ou list.")
        
        # Vérification/adjustement de la shape si nécessaire
        if len(layer_0.shape) == 2 and layer_0.shape[1] != 784:
            # Si shape est (784, batch_size), transposer
            if layer_0.shape[0] == 784:
                layer_0 = layer_0.T  # (784, batch_size) -> (batch_size, 784)

        ## Forward prop for layer_0 -> layer_1

        # Transformation des weights, bias et z_values en tensor
        weights_gpu =  [w.to(device[gpu_id]) for w in self.weights]
        bias_gpu =  [b.to(device[gpu_id]) for b in self.bias]


        # Calcule de z1 = Somme des activation de layer_0 pondérée par les weights
        z1 = torch.addmm(bias_gpu[0], layer_0, weights_gpu[0]) # = W @ layer_0 + B
        # Sauvegarde de z1
        self.z_values[0] = z1
        # Calcule de a1 grace au ReLU
        a1 = self.relu(z1)
        # Sauvegarde de a1 dans layer_1
        self.layers[1] = a1

        ## Forward prop for layer_1 -> layer_2

        # Calcule de z2 = Somme des activation de layer_1 pondérée par les weights
        z2 = torch.addmm(bias_gpu[1], a1, weights_gpu[1]) # = W @ layer_0 + B
        # Sauvegarde de z1
        self.z_values[1] = z2
        # Calcule de a1 grace au ReLU
        a2 = self.relu(z2)
        self.layers[2] = a2

        ## Forward prop for layer_2 -> layer_3

        # Calcule de z2 = Somme des activation de layer_1 pondérée par les weights
        z3 = torch.addmm(bias_gpu[2], a2, weights_gpu[2]) # = W @ layer_0 + B
        # Sauvegarde de z1
        self.z_values[2] = z3
        # Calcule de a1 grace au ReLU
        a3 = torch.softmax(z3, dim=1)
        self.layers[3] = a3 

        return a3
    # END FUNCTION

    def back_prop(self, batch: torch.Tensor, output_target: torch.Tensor, gpu_id: int = 0):
        if batch.device !=  device[gpu_id]:
            output_model = batch.to(device[gpu_id])
        else:
            output_model = batch
        if output_target.device != device[gpu_id]:
            output_target = output_target.to(device[gpu_id])

        weights_gpu = [w.to(device[gpu_id]) for w in self.weights]
        layers_gpu = [l.to(device[gpu_id]) for l in self.layers]
        z_values_gpu = [z.to(device[gpu_id]) for z in self.z_values]

        # === Couche 3 (sortie) ===
        output_error = output_model - output_target # Dérivée du coup par rapport à la sortie du modèle


        error_output_3 = output_error
        
        gradient_weight_2 = torch.matmul(layers_gpu[2].T, error_output_3) # dz(l)/w(l) @ da(l)/z @ dC/a => règle de la chaine
        gradient_bias_2 = torch.mean(error_output_3, dim=0, keepdim=True) # dim(1, 10)

        # === Propagation vers couche 2 ===
        derivated_z2 = self.derivated_relu(z_values_gpu[1])
        error_layer_2 = torch.matmul(error_output_3, weights_gpu[2].T)
        error_output_2 = error_layer_2 * derivated_z2

        gradient_weight_1 = torch.matmul(layers_gpu[1].T, error_output_2)
        gradient_bias_1 = torch.mean(error_output_2, dim=0, keepdim=True) # dim(1, 128)

        # === Propagation vers couche 1
        derivated_z1 = self.derivated_relu(z_values_gpu[0])
        error_layer_1 = torch.matmul(error_output_2, weights_gpu[1].T)
        error_output_1 = error_layer_1 * derivated_z1

        gradient_weight_0 = torch.matmul(layers_gpu[0].T, error_output_1)
        gradient_bias_0 = torch.mean(error_output_1, dim=0, keepdim=True) # dim(1, 256)

        return [gradient_weight_0, gradient_weight_1, gradient_weight_2, \
                gradient_bias_0, gradient_bias_1, gradient_bias_2]
    # END FUNCTION


    def _compute_gradient(self, gradients: list, epoch: int, device_id: int = 0):
        self._update_lr(epoch)

        # Check quel device possède les data poids/biais
        # Chargement de celle ci sur la device ciblée si nécessaire
        if len(self.weights) > 0 and self.weights[0].device != device[device_id]:
            self.weights = [w.to(device[device_id]) for w in self.weights]
        if len(self.bias) > 0 and self.bias[0].device != device[device_id]:
            self.bias = [b.to(device[device_id]) for b in self.bias]

        # Split du tableau pour récupérer les gradients de poids et biais séparés
        if gradients is not None and len(gradients) > 0:
            gradient_weight = gradients[:3]
            gradient_bias = gradients[3:]

            # Chargement des gradients sur la device ciblée s'ils sont des Tensor objects
            gradient_weight = [g.to(device[device_id]) if isinstance(g, torch.Tensor) else g for g in gradient_weight]
            gradient_bias = [g.to(device[device_id]) if isinstance(g, torch.Tensor) else g for g in gradient_bias]

            for i in range(3):
                # MaJ des poids
                self.weights[i] = self.weights[i] - (self.learning_rate * gradient_weight[i])
                # MaJ des biais
                self.bias[i] = self.bias[i] - (self.learning_rate * gradient_bias[i])
        else:
            raise IndexError("Le tableau des gradients ne peut être None ou vide.")

    def _update_lr(self, epoch: int):
        self.learning_rate = self.learning_rate * pow(self.DECAY_RATE, epoch)

    def _compute_precision(self, output_test: torch.Tensor, output_target: torch.Tensor):
        total_test = len(output_test)

        predictions = torch.argmax(output_test, dim=1)
        true_labels = torch.argmax(output_target, dim=1)

        correct = (predictions == true_labels).sum().item()

        return (correct/total_test) * 100 # Pourcentage
    # END FUNCTION

    def init_layers(self):
        # tableau comprenant les différentes couches du NN
        self.layers = []
        self.layers.append(torch.tensor(np.zeros((1, 784)), device=device[0], dtype=torch.float32)) # Matrice d'entré représentant l'image 28x28: couche 0
        self.layers.append(torch.tensor(np.zeros((1, 256)), device=device[0], dtype=torch.float32)) # layers couche 1
        self.layers.append(torch.tensor(np.zeros((1, 128)), device=device[0], dtype=torch.float32)) # layers couche 2
        self.layers.append(torch.tensor(np.zeros((1, 10)), device=device[0], dtype=torch.float32)) # Matrice de sortie (résultat): couche 3
    
    def init_weights(self):
        # tableau des poids et des biais du NN
        self.weights = []
        self.weights.append(torch.tensor(np.random.randn(784, 256), device=device[0], dtype=torch.float32)) # Poids entre la couche (0, 1)
        self.weights.append(torch.tensor(np.random.randn(256, 128), device=device[0], dtype=torch.float32)) # Poids entre la couche (1, 2)
        self.weights.append(torch.tensor(np.random.randn(128, 10), device=device[0], dtype=torch.float32)) # Poids entre la couche (2, 3)

    def init_bias(self):
        self.bias = []
        self.bias.append(torch.tensor(np.random.randn(1, 256), device=device[0], dtype=torch.float32)) # Biais entre la couche (0, 1)
        self.bias.append(torch.tensor(np.random.randn(1, 128), device=device[0], dtype=torch.float32)) # Biais entre la couche (1, 2)
        self.bias.append(torch.tensor(np.random.randn(1, 10), device=device[0], dtype=torch.float32)) # Biais entre la couche (2, 3)

    def init_z_values(self):
        # tableau contenant la somme des activations de la couche précédentes pondérée des poids et biais
        self.z_values = []
        self.z_values.append(torch.tensor(np.random.randn(1, 256), device=device[0], dtype=torch.float32)) # Z_values en sortie de la couche 0 => couche 1
        self.z_values.append(torch.tensor(np.random.randn(1, 128), device=device[0], dtype=torch.float32)) # Z_values en sortie de la couche 1 => couche 2
        self.z_values.append(torch.tensor(np.random.randn(1, 10), device=device[0], dtype=torch.float32)) # Z_values en sortie de la couche 2 => couche 3

    def label_to_vect(self, labels: list) ->  torch.Tensor:
        batch_label = len(labels)

        one_hot = torch.zeros((batch_label, 10), dtype=torch.float32)

        for i, label in enumerate(labels):
            one_hot[i][label] = 1.0

        return one_hot
    
    def derivated_relu(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float() # Convertit True/False en 1.0/0.00