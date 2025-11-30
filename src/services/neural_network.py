from src.utils.mnist import extract_training, extract_testing
import numpy as np
import torch

device = [torch.device("cpu")]
if torch.cuda.is_available():
    nb_devices_available = torch.cuda.device_count()
    print(f"Nombre de GPU disponible: {nb_devices_available}")
    for i in range(nb_devices_available):
        print(f"\tNom du GPU {i}: {torch.cuda.get_device_name(i)}")
    device = [torch.device("cuda:0"), torch.device("cuda:1")]
else:
    print("Pas de GPU disponible !")

class NeuralNetwork():

    INITIAL_RATE = 0.01
    DECAY_RATE = 0.95

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
        self.training_images, self.training_labels = extract_training()
        

    def forward_prop(self, batch: torch.Tensor | np.ndarray, gpu_id: int = 0):
        if isinstance(batch, torch.Tensor):
            layer_0 = batch.to(device[gpu_id])
        elif isinstance(batch, np.ndarray):
            layer_0 = torch.tensor(batch, device=device[gpu_id])
        else:
            raise ValueError("Le batch doit être de type torch.Tensor ou np.ndarray.")

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
        a3 = self.relu(z3)
        self.layers[3] = a3 

        return a3
    # END FUNCTION

    def back_prop(self, batch: np.ndarray | torch.Tensor, output_target: torch.Tensor, gpu_id: int = 0):
        if batch.device !=  device[gpu_id]:
            output_model = batch.to(device[gpu_id])
        if output_target.device != device[gpu_id]:
            output_target = output_target.to(device[gpu_id])

        weights_gpu = [w.to(device[gpu_id]) for w in self.weights]
        layers_gpu = [l.to(device[gpu_id]) for l in self.layers]
        z_values_gpu = [z.to(device[gpu_id]) for z in self.z_values]

        # === Couche 3 (sortie) ===
        output_error = 2 * (output_model - output_target) # Dérivée du coup par rapport à la sortie du modèle

        derivated_z3 = self.derivated_relu(z_values_gpu[2])

        error_output_3 = output_error * derivated_z3
        
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

    def training(self):
        batch_size = 500 
        epochs = 10

        for i in range(epochs):
            for batch in range(len(self.training_images), batch_size):
                batch_images = self.training_image[batch:batch+batch_size]

                output_model = self.forward_prop(batch_images)
                output_target = self.label_to_vect(self.training_labels[batch:batch+batch_size])

                gradient_weights = self.back_prop(output_model, output_target)
                pass

        

    def testint(self):
        pass

    def _compute_gradient(self, gradient_weights: torch.Tensor, gradient_bias: torch.Tensor):
        # MaJ des poids
        self.weights[0] = self.weights[0] - (self.learning_rate * gradient_weights[0])
        self.weights[1] = self.weights[1] - (self.learning_rate * gradient_weights[1])
        self.weights[2] = self.weights[2] - (self.learning_rate * gradient_weights[2])

        # MaJ des biais
        self.bias[0] = self.bias[0] - (self.learning_rate * gradient_bias[0])
        self.bias[1] = self.bias[1] - (self.learning_rate * gradient_bias[1])
        self.bias[2] = self.bias[2] - (self.learning_rate * gradient_bias[2])

    def _update_lr(self, epoch: int):
        self.learning_rate = self.learning_rate * pow(self.DECAY_RATE, epoch)

    def init_layers(self):
        # tableau comprenant les différentes couches du NN
        self.layers = []
        self.layers.append(torch.tensor(np.zeros((1, 784)), device=device[0])) # Matrice d'entré représentant l'image 28x28: couche 0
        self.layers.append(torch.tensor(np.zeros((1, 256)), device=device[0])) # layers couche 1
        self.layers.append(torch.tensor(np.zeros((1, 128)), device=device[0])) # layers couche 2
        self.layers.append(torch.tensor(np.zeros((1, 10)), device=device[0])) # Matrice de sortie (résultat): couche 3
    
    def init_weights(self):
        # tableau des poids et des biais du NN
        self.weights = []
        self.weights.append(torch.tensor(np.random.randn(784, 256), device=device[0])) # Poids entre la couche (0, 1)
        self.weights.append(torch.tensor(np.random.randn(256, 128), device=device[0])) # Poids entre la couche (1, 2)
        self.weights.append(torch.tensor(np.random.randn(128, 10), device=device[0])) # Poids entre la couche (2, 3)

    def init_bias(self):
        self.bias = []
        self.bias.append(torch.tensor(np.random.randn(1, 256), device=device[0])) # Biais entre la couche (0, 1)
        self.bias.append(torch.tensor(np.random.randn(1, 128), device=device[0])) # Biais entre la couche (1, 2)
        self.bias.append(torch.tensor(np.random.randn(1, 10), device=device[0])) # Biais entre la couche (2, 3)

    def init_z_values(self):
        # tableau contenant la somme des activations de la couche précédentes pondérée des poids et biais
        self.z_values = []
        self.z_values.append(torch.tensor(np.random.randn(1, 256), device=device[0])) # Z_values en sortie de la couche 0 => couche 1
        self.z_values.append(torch.tensor(np.random.randn(1, 128), device=device[0])) # Z_values en sortie de la couche 1 => couche 2
        self.z_values.append(torch.tensor(np.random.randn(1, 10), device=device[0])) # Z_values en sortie de la couche 2 => couche 3

    def label_to_vect(self, x: int):
        vect = torch.new_zeros(1,10)
        vect[x] = 1.0
        return vect
    
    def derivated_relu(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float() # Convertit True/False en 1.0/0.00