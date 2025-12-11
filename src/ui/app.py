import sys, os, glob, random
from pathlib import Path
from neural_network import NeuralNetwork

# Ajouter src/ au PYTHONPATH pour les imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from PyQt5 import *
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
import numpy as np


class MyApp(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        self.nn = NeuralNetwork()
        self.model_loaded = False  # Flag pour savoir si le modèle est chargé
        self.current_image = None  # Image MNIST actuellement affichée (normalisée)
        self.current_label = None  # Label réel de l'image actuelle

        # Charger le modèle au démarrage (une seule fois)
        self.load_model_if_available()

        self.set_layout()
        self.init_image_display()
        self.init_prediction_display()
        self.init_new_btn()
        # Ajouter le container du bouton après l'avoir créé
        self.main_layout.addWidget(self.button_container)
    # END FUNCTION

    def run(self):
        self.exec()
    # END FUNCTION

    def set_layout(self):
        self.window = QMainWindow()
        # Créer le widget central avec un layout vertical
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)  # Marges autour du layout
        self.main_layout.setSpacing(10)  # Espacement entre les widgets
        self.central_widget.setLayout(self.main_layout)
        # Ajout du layout a la window
        self.window.setCentralWidget(self.central_widget)
        # Force size de la fenêtre (560 pour l'image + ~100 pour la prédiction + ~60 pour le bouton + marges)
        self.window.setMinimumSize(QSize(600, 750))
        self.window.show()
    # END FUNCTION

    def init_image_display(self):
        # Label pour afficher l'image MNIST (taille fixe)
        self.image_label = QLabel()
        self.image_label.setFixedSize(560, 560)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        self.image_label.setText("Cliquez sur 'New' pour charger une image MNIST")
        # Ajouter le label avec un alignement centré
        self.main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
    # END FUNCTION

    def init_prediction_display(self):
        # Label pour afficher la prédiction
        self.prediction_label = QLabel("Prédiction: -")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        self.main_layout.addWidget(self.prediction_label, alignment=Qt.AlignCenter)
    # END FUNCTION

    def init_new_btn(self):
        # Container pour le bouton centré en bas
        self.button_container = QWidget()
        self.button_container.setFixedHeight(50)  # Hauteur fixe pour le container
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)  # Pas de marges dans le container
        self.button_container.setLayout(self.button_layout)

        self.button_layout.addStretch()
        self.new_button = QPushButton("New")
        self.new_button.setMinimumSize(QSize(100, 40))  # Taille minimale du bouton
        self.new_button.pressed.connect(self.load_new_image)
        self.button_layout.addWidget(self.new_button)
        self.button_layout.addStretch()
    # END FUNCTION

    def load_new_image(self):
        """Charge une image MNIST test aléatoire et fait une prédiction."""
        if not self.model_loaded:
            # Afficher un message d'erreur si le modèle n'est pas disponible
            msg = QMessageBox(self.window)
            msg.setWindowTitle("Erreur")
            msg.setText("Aucun modèle disponible. Veuillez d'abord entraîner le modèle.")
            msg.exec()
            return

        # Charger les images de test (déjà chargées dans nn.testing_images)
        if len(self.nn.testing_images) == 0:
            msg = QMessageBox(self.window)
            msg.setWindowTitle("Erreur")
            msg.setText("Aucune image de test disponible.")
            msg.exec()
            return

        # Choisir une image aléatoire
        random_index = random.randint(0, len(self.nn.testing_images) - 1)
        # Les images de test ne sont PAS normalisées (0-255), il faut les normaliser
        image_raw = self.nn.testing_images[random_index]
        self.current_label = self.nn.testing_labels[random_index]
        
        # Normaliser l'image (0-255 -> 0-1)
        self.current_image = [pixel / 255.0 for pixel in image_raw]

        # Afficher l'image dans le label (utiliser l'image normalisée pour l'affichage)
        self.display_mnist_image(self.current_image)

        # Faire la prédiction
        predicted_label, probabilities = self.nn.run("USER", user_image=self.current_image)
        
        # Afficher la prédiction
        self.update_prediction_display(predicted_label, self.current_label, probabilities)
    # END FUNCTION

    def display_mnist_image(self, image_normalized):
        """
        Affiche une image MNIST normalisée (liste de 784 pixels 0-1) dans le QLabel.
        
        Args:
            image_normalized: Liste de 784 pixels normalisés entre 0 et 1
        """
        # Convertir la liste en array numpy
        img_array = np.array(image_normalized, dtype=np.float32)
        
        # Reshape en 28x28
        img_2d = img_array.reshape(28, 28)
        
        # Convertir de 0-1 à 0-255 (uint8)
        # Dans MNIST normalisé : 0.0 = noir (chiffre), 1.0 = blanc (fond)
        # Pour l'affichage, on inverse : 0.0 -> 255 (blanc), 1.0 -> 0 (noir)
        img_uint8 = (1.0 - img_2d) * 255.0
        img_uint8 = img_uint8.astype(np.uint8)
        
        # Créer une QImage depuis le array numpy
        height, width = img_uint8.shape
        qimage = QImage(img_uint8.data, width, height, QImage.Format_Grayscale8)
        
        # Redimensionner pour l'affichage (560x560)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(560, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Afficher dans le label
        self.image_label.setPixmap(pixmap)
    # END FUNCTION

    def update_prediction_display(self, predicted_label, true_label, probabilities):
        """
        Met à jour l'affichage de la prédiction.
        
        Args:
            predicted_label: Label prédit par le modèle
            true_label: Label réel de l'image
            probabilities: Array de probabilités pour chaque chiffre
        """
        # Créer le texte de prédiction
        result_text = f"Prédiction: <b>{predicted_label}</b>"
        if true_label is not None:
            result_text += f" | Vrai label: {true_label}"
            if predicted_label == true_label:
                result_text += " ✓"
            else:
                result_text += " ✗"
        
        self.prediction_label.setText(result_text)
    # END FUNCTION

    def load_model_if_available(self):
        """
        Charge le modèle une seule fois au démarrage si disponible.
        """
        models_dir = "models"
        model_files = []
        
        if os.path.exists(models_dir):
            # Chercher tous les fichiers model_*.pt
            model_files = glob.glob(os.path.join(models_dir, "model_*.pt"))
        
        if model_files:
            # Trier par date de modification (le plus récent en dernier)
            model_files.sort(key=lambda x: os.path.getmtime(x))
            latest_model = model_files[-1]
            
            print(f"Modèle sauvegardé trouvé: {latest_model}")
            print("Chargement du modèle...")
            
            # Charger le modèle
            self.nn.load_model(latest_model)
            self.model_loaded = True
            print("Modèle chargé avec succès.")
        else:
            print("Aucun modèle sauvegardé trouvé. Le modèle sera entraîné au premier envoi.")
            self.model_loaded = False
    # END FUNCTION
        
    def prediction_dlg(self, label: int, probabilities):
        dlg = QDialog(self.window)
        dlg.setWindowTitle("Prédiction")
        dlg.setMinimumSize(QSize(300, 200))
        
        layout = QVBoxLayout()
        
        # Afficher le label prédit
        label_text = QLabel(f"Chiffre prédit: <b>{label}</b>")
        label_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_text)
        
        # Afficher les probabilités
        prob_text = "Probabilités:\n"
        for i, prob in enumerate(probabilities):
            prob_text += f"{i}: {prob*100:.2f}%\n"
        
        prob_label = QLabel(prob_text)
        prob_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(prob_label)
        
        # Bouton OK
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dlg.accept)
        layout.addWidget(ok_button)
        
        dlg.setLayout(layout)
        dlg.exec()
    # END FUNCTION