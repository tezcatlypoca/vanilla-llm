from utils.mnist import extract_training, extract_testing
from threading import Thread, Lock
import numpy as np
import torch, time
import os
from datetime import datetime

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
    # print(device)

class NeuralNetwork():

    INITIAL_RATE = 0.01
    MIN_RATE = 0.001
    DECAY_RATE = 0.95
    BATCH_SIZE = 500
    MAX_EPOCHS = 10

    output_test = []
    gradients_thread = []

    # Params:
    #   - Layering: 
    #       * la taille du tableau représente le nombre de couche du NN
    #       * chaque représente le nombre de neurones de la couche
    def __init__(self):
        self.lock = Lock()

        self.relu = torch.nn.ReLU() # Var contenant la fonction de ReLU, à utiliser comme une fonction
        self.learning_rate = self.INITIAL_RATE
        self.precision = None  # Précision calculée après test
        self.total_time = None  # Temps total d'entraînement

        # init de toutes les matrices
        self.init_layers()
        self.init_weights()
        self.init_bias()
        self.init_z_values()

        # init des images et labels d'entrainement
        self.training_images, self.training_labels, _ = extract_training()

        self.testing_images, self.testing_labels, _ = extract_testing()

    def run(self, mode: str):
        if mode == "TRAINING":
            total_batches = (len(self.training_images) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
            
            print("=" * 70)
            print("ENTRAÎNEMENT")
            print("=" * 70)
            print(f"Dataset: {len(self.training_images)} images")
            print(f"Batch size: {self.BATCH_SIZE} | Batches par epoch: {total_batches}")
            print(f"Epochs: {self.MAX_EPOCHS} | Learning rate initial: {self.INITIAL_RATE}")
            print("=" * 70)
            
            start_time = time.time()
            epoch_times = []

            for e_index in range(self.MAX_EPOCHS):
                epoch_start = time.time()
                batch_count = 0

                self._update_lr(e_index)
                
                for i_index in range(0, len(self.training_images), self.BATCH_SIZE):
                    batch_0 = self.training_images[i_index: i_index+(self.BATCH_SIZE//2)] # Première moitié du batch
                    batch_1 = self.training_images[i_index+(self.BATCH_SIZE//2): i_index+self.BATCH_SIZE] # Deuxième moitié du batch

                    labels_0 = self.training_labels[i_index: i_index+(self.BATCH_SIZE//2)]
                    labels_1 = self.training_labels[i_index+(self.BATCH_SIZE//2): i_index+self.BATCH_SIZE]

                    # Déterminer les device_id selon les devices disponibles
                    # Si 2+ GPU disponibles, utiliser GPU 0 et GPU 1 (device[1] et device[2])
                    # Sinon, utiliser CPU pour les deux (device[0])
                    if len(device) >= 3:  # Au moins CPU + 2 GPU
                        device_id_0 = 1  # GPU 0
                        device_id_1 = 2  # GPU 1
                    else:
                        device_id_0 = 0  # CPU
                        device_id_1 = 0  # CPU

                    thread_gpu_0 = Thread(target=self.training, args=(e_index, batch_0, labels_0), kwargs={"device_id": device_id_0})
                    thread_gpu_1 = Thread(target=self.training, args=(e_index, batch_1, labels_1), kwargs={"device_id": device_id_1})

                    thread_gpu_0.start()
                    thread_gpu_1.start()

                    thread_gpu_0.join()
                    thread_gpu_1.join()

                    averaged_grad = self._average_gradients(self.gradients_thread)
                    self._compute_gradient(averaged_grad)
                    self.gradients_thread = []
                    
                    batch_count += 1

                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
                elapsed_time = time.time() - start_time
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = self.MAX_EPOCHS - (e_index + 1)
                estimated_remaining = avg_epoch_time * remaining_epochs
                
                # Affichage après chaque epoch (pas de calculs lourds)
                with self.lock:
                    current_lr = self.learning_rate
                
                print(f"Epoch {e_index+1}/{self.MAX_EPOCHS} | "
                      f"Temps: {epoch_time:.1f}s | "
                      f"LR: {current_lr:.6f} | "
                      f"Total: {elapsed_time/60:.1f}min | "
                      f"Restant: ~{estimated_remaining/60:.1f}min")

            self.total_time = time.time() - start_time
            print("=" * 70)
            print(f"Entraînement terminé en {self.total_time/60:.1f} minutes ({self.total_time:.1f}s)")
            print(f"Temps moyen par epoch: {sum(epoch_times)/len(epoch_times):.1f}s")
            print("=" * 70)
            
            # Sauvegarde automatique du modèle après l'entraînement
            saved_path = self.save_model()
            print(f"Modèle sauvegardé automatiquement: {saved_path}")
        elif mode == "TESTING":
            self.output_test = []
            total_test_batches = (len(self.testing_images) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
            
            print("=" * 70)
            print("TEST")
            print("=" * 70)
            print(f"Dataset: {len(self.testing_images)} images")
            print(f"Batch size: {self.BATCH_SIZE} | Batches: {total_test_batches}")
            print("=" * 70)
            
            start_time = time.time()
            batch_count = 0

            for i_index in range(0, len(self.testing_images), self.BATCH_SIZE):
                batch_0 = self.testing_images[i_index: i_index+(self.BATCH_SIZE//2)] # Première moitié du batch
                batch_1 = self.testing_images[i_index+(self.BATCH_SIZE//2): i_index+self.BATCH_SIZE] # Deuxième moitié du batch

                # Déterminer les device_id selon les devices disponibles
                if len(device) >= 3:  # Au moins CPU + 2 GPU
                    device_id_0 = 1  # GPU 0
                    device_id_1 = 2  # GPU 1
                else:
                    device_id_0 = 0  # CPU
                    device_id_1 = 0  # CPU

                thread_gpu_0 = Thread(target=self.testing, args=(batch_0, device_id_0))
                thread_gpu_1 = Thread(target=self.testing, args=(batch_1, device_id_1))

                thread_gpu_0.start()
                thread_gpu_1.start()

                thread_gpu_0.join()
                thread_gpu_1.join()
                
                batch_count += 1
                if batch_count % 10 == 0 or batch_count == total_test_batches:
                    elapsed = time.time() - start_time
                    progress = (batch_count / total_test_batches) * 100
                    print(f"Progression: {batch_count}/{total_test_batches} batches ({progress:.1f}%) | "
                          f"Temps: {elapsed:.1f}s", end='\r')
            
            test_time = time.time() - start_time
            print()  # Nouvelle ligne après la progression
            
            # Matrice représentant l'ensemble des sortie du test dim(total_size, 10)
            if len(self.output_test) > 0:
                print("Calcul de la précision...")
                self.output_test = [o.to(device[0]) for o in self.output_test]
                output_model_test = torch.cat(self.output_test, dim=0)
                output_target_test = self.label_to_vect(self.testing_labels)

                precision = self._compute_precision(output_model_test, output_target_test)
                self.precision = precision  # Stocker la précision pour la sauvegarde
                correct = (torch.argmax(output_model_test, dim=1) == torch.argmax(output_target_test, dim=1)).sum().item()
                total = len(output_model_test)
                
                print("=" * 70)
                print("RÉSULTATS DU TEST")
                print("=" * 70)
                print(f"Précision: {precision:.2f}%")
                print(f"Prédictions correctes: {correct}/{total}")
                print(f"Temps de test: {test_time:.1f}s ({test_time/60:.2f} min)")
                print(f"Images par seconde: {len(self.testing_images)/test_time:.1f}")
                print("=" * 70)
                
                # Si le modèle a été entraîné avant, mettre à jour la sauvegarde avec la précision
                if self.total_time is not None:
                    # Trouver le dernier modèle sauvegardé et le mettre à jour
                    models_dir = "models"
                    if os.path.exists(models_dir):
                        # Récupérer le dernier fichier de modèle
                        model_files = [f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".pt")]
                        if model_files:
                            # Trier par date de modification (le plus récent en dernier)
                            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
                            latest_model = os.path.join(models_dir, model_files[-1])
                            # Charger, mettre à jour la précision et resauvegarder
                            try:
                                save_dict = torch.load(latest_model, map_location=device[0])
                                if 'metadata' in save_dict:
                                    save_dict['metadata']['precision'] = precision
                                    torch.save(save_dict, latest_model)
                                    print(f"Modèle mis à jour avec la précision: {latest_model}")
                            except Exception as e:
                                print(f"Attention: Impossible de mettre à jour le modèle avec la précision: {e}")
    # END FUNCTION

    def training(self, epoch: int, batch: list, labels: list, device_id: int = 0):
        batch_tensor = torch.tensor(batch, dtype=torch.float32)
        output_model = self.forward_prop(batch_tensor, device_id)
        output_target = self.label_to_vect(labels)

        gradients = self.back_prop(output_model, output_target, device_id)
        with self.lock:
            self.gradients_thread.append(gradients)
        # self._compute_gradient(gradients, epoch, device_id)        

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

        ## Forward prop for layer_0 -> layer_1

        # Transformation des weights, bias et z_values en tensor
        weights_gpu =  [w.to(device[gpu_id]) for w in self.weights]
        bias_gpu =  [b.to(device[gpu_id]) for b in self.bias]

        self.layers[0] = layer_0


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
        
        gradient_weight_2 = torch.matmul(layers_gpu[2].T, error_output_3) / error_output_3.shape[0] # dz(l)/w(l) @ da(l)/z @ dC/a => règle de la chaine
        gradient_bias_2 = torch.mean(error_output_3, dim=0, keepdim=True) # dim(1, 10)

        # === Propagation vers couche 2 ===
        derivated_z2 = self.derivated_relu(z_values_gpu[1])
        error_layer_2 = torch.matmul(error_output_3, weights_gpu[2].T)
        error_output_2 = error_layer_2 * derivated_z2

        gradient_weight_1 = torch.matmul(layers_gpu[1].T, error_output_2) / error_output_2.shape[0]
        gradient_bias_1 = torch.mean(error_output_2, dim=0, keepdim=True) # dim(1, 128)

        # === Propagation vers couche 1
        derivated_z1 = self.derivated_relu(z_values_gpu[0])
        error_layer_1 = torch.matmul(error_output_2, weights_gpu[1].T)
        error_output_1 = error_layer_1 * derivated_z1

        gradient_weight_0 = torch.matmul(layers_gpu[0].T, error_output_1) / error_output_1.shape[0]
        gradient_bias_0 = torch.mean(error_output_1, dim=0, keepdim=True) # dim(1, 256)

        return [gradient_weight_0, gradient_weight_1, gradient_weight_2, \
                gradient_bias_0, gradient_bias_1, gradient_bias_2]
    # END FUNCTION


    def _compute_gradient(self, gradients: list, device_id: int = 0):
        with self.lock:
            
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
    # END FUNCTION

    def _update_lr(self, epoch: int):
        old_rate = self.learning_rate
        self.learning_rate = self.INITIAL_RATE * (1 - epoch / self.MAX_EPOCHS)
        # self.learning_rate = self.INITIAL_RATE * max(0.01, 1 - self.DECAY_RATE * epoch)
        if self.learning_rate < self.MIN_RATE:
            self.learning_rate = old_rate

    def _compute_precision(self, output_test: torch.Tensor, output_target: torch.Tensor):
        total_test = len(output_test)

        predictions = torch.argmax(output_test, dim=1)
        true_labels = torch.argmax(output_target, dim=1)

        correct = (predictions == true_labels).sum().item()

        return (correct/total_test) * 100 # Pourcentage
    # END FUNCTION

    def _average_gradients(self, grad_list: list):
        if len(grad_list) < 2:
            raise ValueError("grad_list doit contenir au moins 2 éléments (un par thread)")
    
        averaged_grad = []
        # Extraire les gradients de chaque thread
        grad_thread_0 = grad_list[0]  # Liste de 6 tensors (3 poids, 3 biais)
        grad_thread_1 = grad_list[1]  # Liste de 6 tensors
        
        # Pour chaque gradient (poids ou biais), faire la moyenne
        for g0, g1 in zip(grad_thread_0, grad_thread_1):
            # Déplacer les deux gradients sur le CPU
            g0_cpu = g0.to(device[0]) if isinstance(g0, torch.Tensor) else torch.tensor(g0, device=device[0])
            g1_cpu = g1.to(device[0]) if isinstance(g1, torch.Tensor) else torch.tensor(g1, device=device[0])
            
            # Moyenner les deux gradients
            averaged = torch.mean(torch.stack([g0_cpu, g1_cpu], dim=0), dim=0)
            averaged_grad.append(averaged)
        
        return averaged_grad

    def init_layers(self):
        # tableau comprenant les différentes couches du NN
        self.layers = []
        self.layers.append(torch.tensor(np.zeros((self.BATCH_SIZE, 784)), device=device[0], dtype=torch.float32)) # Matrice d'entré représentant l'image 28x28: couche 0
        self.layers.append(torch.tensor(np.zeros((self.BATCH_SIZE, 256)), device=device[0], dtype=torch.float32)) # layers couche 1
        self.layers.append(torch.tensor(np.zeros((self.BATCH_SIZE, 128)), device=device[0], dtype=torch.float32)) # layers couche 2
        self.layers.append(torch.tensor(np.zeros((self.BATCH_SIZE, 10)), device=device[0], dtype=torch.float32)) # Matrice de sortie (résultat): couche 3
    
    def init_weights(self):
        # tableau des poids et des biais du NN
        self.weights = []
        self.weights.append(torch.tensor(np.random.randn(784, 256) * np.sqrt(2.0 / 784), device=device[0], dtype=torch.float32)) # Poids entre la couche (0, 1)
        self.weights.append(torch.tensor(np.random.randn(256, 128) * np.sqrt(2.0 / 256), device=device[0], dtype=torch.float32)) # Poids entre la couche (1, 2)
        self.weights.append(torch.tensor(np.random.randn(128, 10) * np.sqrt(2.0 / 128), device=device[0], dtype=torch.float32)) # Poids entre la couche (2, 3)

    def init_bias(self):
        self.bias = []
        self.bias.append(torch.zeros((1, 256), device=device[0], dtype=torch.float32)) # Biais entre la couche (0, 1)
        self.bias.append(torch.zeros((1, 128), device=device[0], dtype=torch.float32)) # Biais entre la couche (1, 2)
        self.bias.append(torch.zeros((1, 10), device=device[0], dtype=torch.float32)) # Biais entre la couche (2, 3)

    def init_z_values(self):
        # tableau contenant la somme des activations de la couche précédentes pondérée des poids et biais
        self.z_values = []
        self.z_values.append(torch.zeros((1, 256), device=device[0], dtype=torch.float32)) # Z_values en sortie de la couche 0 => couche 1
        self.z_values.append(torch.zeros((1, 128), device=device[0], dtype=torch.float32)) # Z_values en sortie de la couche 1 => couche 2
        self.z_values.append(torch.zeros((1, 10), device=device[0], dtype=torch.float32)) # Z_values en sortie de la couche 2 => couche 3

    def label_to_vect(self, labels: list) ->  torch.Tensor:
        batch_label = len(labels)

        one_hot = torch.zeros((batch_label, 10), dtype=torch.float32)

        for i, label in enumerate(labels):
            one_hot[i][label] = 1.0

        return one_hot
    
    def derivated_relu(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float() # Convertit True/False en 1.0/0.00
    
    def save_model(self, filepath: str = None) -> str:
        """
        Sauvegarde le modèle (poids, biais et métadonnées) dans un fichier .pt
        
        Args:
            filepath: Chemin du fichier de sauvegarde. Si None, génère un nom avec horodatage.
        
        Returns:
            Chemin du fichier sauvegardé
        """
        # Créer le dossier models/ s'il n'existe pas
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Générer le nom de fichier avec horodatage si non fourni
        if filepath is None or "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(models_dir, f"model_{timestamp}.pt")
        else:
            # S'assurer que le chemin est dans le dossier models/
            if not filepath.startswith(models_dir):
                filepath = os.path.join(models_dir, filepath)
        
        # Déplacer les poids et biais sur CPU pour compatibilité
        weights_cpu = [w.to(device[0]) for w in self.weights]
        bias_cpu = [b.to(device[0]) for b in self.bias]
        
        # Préparer les métadonnées
        metadata = {
            'architecture': [layer.shape[1] for layer in self.layers],  # Tailles des couches
            'epochs': self.MAX_EPOCHS,
            'initial_lr': self.INITIAL_RATE,
            'final_lr': self.learning_rate,
            'batch_size': self.BATCH_SIZE,
            'training_time': self.total_time if self.total_time is not None else None,
            'precision': self.precision,  # Peut être None si test non effectué
            'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Créer le dictionnaire de sauvegarde
        save_dict = {
            'weights': weights_cpu,
            'bias': bias_cpu,
            'metadata': metadata
        }
        
        # Sauvegarder
        torch.save(save_dict, filepath)
        print(f"Modèle sauvegardé: {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> dict:
        """
        Charge un modèle sauvegardé depuis un fichier .pt
        
        Args:
            filepath: Chemin du fichier à charger
        
        Returns:
            Dictionnaire contenant les métadonnées du modèle chargé
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            RuntimeError: Si le fichier est corrompu ou incompatible
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
        
        try:
            # Charger le fichier
            save_dict = torch.load(filepath, map_location=device[0])
            
            # Vérifier la structure
            if 'weights' not in save_dict or 'bias' not in save_dict:
                raise RuntimeError("Format de fichier invalide: poids ou biais manquants.")
            
            # Restaurer les poids et biais
            self.weights = [w.to(device[0]) for w in save_dict['weights']]
            self.bias = [b.to(device[0]) for b in save_dict['bias']]
            
            # Afficher les métadonnées
            metadata = save_dict.get('metadata', {})
            print("=" * 70)
            print("MODÈLE CHARGÉ")
            print("=" * 70)
            if metadata:
                print(f"Architecture: {metadata.get('architecture', 'N/A')}")
                print(f"Epochs: {metadata.get('epochs', 'N/A')}")
                print(f"Learning rate initial: {metadata.get('initial_lr', 'N/A')}")
                print(f"Learning rate final: {metadata.get('final_lr', 'N/A')}")
                print(f"Batch size: {metadata.get('batch_size', 'N/A')}")
                if metadata.get('training_time'):
                    print(f"Temps d'entraînement: {metadata.get('training_time'):.1f}s ({metadata.get('training_time')/60:.1f} min)")
                if metadata.get('precision') is not None:
                    print(f"Précision: {metadata.get('precision'):.2f}%")
                print(f"Sauvegardé le: {metadata.get('saved_at', 'N/A')}")
            print("=" * 70)
            
            return metadata
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {str(e)}")