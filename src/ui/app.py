import sys, os, glob
from pathlib import Path
import numpy as np
import torch
from neural_network import NeuralNetwork

# Ajouter src/ au PYTHONPATH pour les imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from PyQt5 import *
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import *
from ui.qwidget import DrawingWidget


class MyApp(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        self.nn = NeuralNetwork()
        self.model_loaded = False  # Flag pour savoir si le modèle est chargé

        # Charger le modèle au démarrage (une seule fois)
        self.load_model_if_available()

        self.set_layout()
        self.init_canva()
        self.init_send_btn()
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
        # Force size de la fenêtre (560 pour le canvas + ~60 pour le bouton + marges)
        self.window.setMinimumSize(QSize(600, 650))
        self.window.show()
    # END FUNCTION

    def init_canva(self):
        # Canvas en haut (taille fixe)
        self.canva = DrawingWidget()
        self.canva.setFixedSize(560, 560)
        # Ajouter le canvas avec un alignement centré
        self.main_layout.addWidget(self.canva, alignment=Qt.AlignCenter)
    # END FUNCTION

    def init_send_btn(self):
        # Container pour le bouton centré en bas
        self.button_container = QWidget()
        self.button_container.setFixedHeight(50)  # Hauteur fixe pour le container
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)  # Pas de marges dans le container
        self.button_container.setLayout(self.button_layout)

        self.button_layout.addStretch()
        self.reset_button = QPushButton("Send")
        self.reset_button.setMinimumSize(QSize(100, 40))  # Taille minimale du bouton
        self.reset_button.pressed.connect(self.send_image)
        self.button_layout.addWidget(self.reset_button)
        self.button_layout.addStretch()
    # END FUNCTION

    def send_image(self):
        draw = self.canva.save_draw()  # Utiliser save_draw() au lieu de resize()
        # Stocker l'image pour pouvoir la réutiliser pour l'apprentissage
        self.current_user_image = draw
        self.canva.refresh()

        if self.model_loaded:
            # run("USER") retourne déjà (predicted_label, probabilities)
            predicted_label, probabilities = self.nn.run("USER", user_image=draw)
            self.prediction_dlg(predicted_label, probabilities, draw)
        else:
            # Afficher un message d'erreur si le modèle n'est pas disponible
            msg = QMessageBox(self.window)
            msg.setWindowTitle("Erreur")
            msg.setText("Aucun modèle disponible. Veuillez d'abord entraîner le modèle.")
            msg.exec()
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
            self.nn.run("TRAINING")
            self.model_loaded = False
    # END FUNCTION
        
    def prediction_dlg(self, predicted_label: int, probabilities, user_image: list):
        dlg = QDialog(self.window)
        dlg.setWindowTitle("Prédiction")
        dlg.setMinimumSize(QSize(350, 300))
        
        layout = QVBoxLayout()
        
        # Afficher le label prédit
        label_text = QLabel(f"Chiffre prédit: <b>{predicted_label}</b>")
        label_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_text)
        
        # Afficher les probabilités (top 3 seulement pour ne pas surcharger)
        prob_text = "Probabilités (top 3):\n"
        top_indices = np.argsort(probabilities)[::-1][:3]
        for idx in top_indices:
            prob_text += f"{idx}: {probabilities[idx]*100:.2f}%\n"
        
        prob_label = QLabel(prob_text)
        prob_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(prob_label)

        # Question de validation
        question_label = QLabel("Le résultat est-il correct ?")
        question_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(question_label)
        
        # Boutons Yes/No
        buttons_layout = QHBoxLayout()
        yes_btn = QPushButton("Oui")
        no_btn = QPushButton("Non")
        buttons_layout.addWidget(yes_btn)
        buttons_layout.addWidget(no_btn)
        layout.addLayout(buttons_layout)

        # Champ pour la réponse correcte (caché initialement)
        answer_label = QLabel("Quel est le bon chiffre ? (0-9):")
        answer_label.setVisible(False)
        user_answer = QLineEdit()
        user_answer.setPlaceholderText("Entrez un chiffre entre 0 et 9")
        user_answer.setVisible(False)
        user_answer.setMaxLength(1)
        
        # Bouton OK pour valider la réponse
        ok_button = QPushButton("Valider")
        ok_button.setVisible(False)
        
        layout.addWidget(answer_label)
        layout.addWidget(user_answer)
        layout.addWidget(ok_button)
        
        # Variables pour stocker le résultat
        result = {"correct": None, "true_label": None}
        
        def on_yes():
            result["correct"] = True
            result["true_label"] = predicted_label
            dlg.accept()
        
        def on_no():
            answer_label.setVisible(True)
            user_answer.setVisible(True)
            ok_button.setVisible(True)
            yes_btn.setVisible(False)
            no_btn.setVisible(False)
            user_answer.setFocus()
        
        def on_ok():
            try:
                correct_number = int(user_answer.text())
                if 0 <= correct_number <= 9:
                    result["correct"] = False
                    result["true_label"] = correct_number
                    dlg.accept()
                else:
                    QMessageBox.warning(dlg, "Erreur", "Veuillez entrer un chiffre entre 0 et 9.")
            except ValueError:
                QMessageBox.warning(dlg, "Erreur", "Veuillez entrer un nombre valide.")
        
        yes_btn.clicked.connect(on_yes)
        no_btn.clicked.connect(on_no)
        ok_button.clicked.connect(on_ok)
        user_answer.returnPressed.connect(on_ok)  # Entrée pour valider
        
        dlg.setLayout(layout)
        dlg.exec()
        
        # Traiter le résultat après la fermeture de la dialog
        if result["correct"] is not None:
            self.handle_user_feedback(result["correct"], result["true_label"], user_image, predicted_label)
    # END FUNCTION

    def handle_user_feedback(self, is_correct: bool, true_label: int, user_image: list, predicted_label: int):
        """
        Gère le retour de l'utilisateur et effectue l'apprentissage en ligne si nécessaire.
        
        Args:
            is_correct: True si la prédiction était correcte, False sinon
            true_label: Le label correct (soit predicted_label si correct, soit la réponse de l'utilisateur)
            user_image: L'image dessinée par l'utilisateur
            predicted_label: Le label prédit par le modèle
        """
        if is_correct:
            print(f"✓ Prédiction correcte: {predicted_label}")
            # Faire un apprentissage en ligne pour renforcer le comportement positif
            # avec un learning rate encore plus faible pour juste renforcer légèrement
            self.online_learning(user_image, true_label, reinforcement=True)
        else:
            print(f"✗ Prédiction incorrecte: prédit {predicted_label}, correct: {true_label}")
            # Faire un apprentissage en ligne avec cette image pour corriger
            self.online_learning(user_image, true_label, reinforcement=False)
    # END FUNCTION

    def online_learning(self, user_image: list, correct_label: int, reinforcement: bool = False):
        """
        Effectue un apprentissage en ligne avec une seule image.
        Utilise un learning rate beaucoup plus faible pour éviter le surapprentissage.
        
        Args:
            user_image: L'image à utiliser pour l'apprentissage (liste de 784 pixels)
            correct_label: Le label correct pour cette image
            reinforcement: True si c'est pour renforcer une prédiction correcte, False pour corriger une erreur
        """
        if reinforcement:
            print(f"Renforcement positif: image avec label {correct_label}")
        else:
            print(f"Correction: image avec label {correct_label}")
        
        # Déterminer le device à utiliser
        from neural_network import device
        if len(device) >= 3:  # Au moins CPU + 2 GPU
            device_id = 1  # GPU 0
        else:
            device_id = 0  # CPU
        
        # Sauvegarder le learning rate actuel
        original_lr = self.nn.learning_rate
        
        # Utiliser un learning rate adapté selon le type d'apprentissage
        if reinforcement:
            # Pour le renforcement positif : learning rate très faible pour juste renforcer légèrement
            online_lr = 0.00001  # 1000x plus faible (renforcement très léger)
        else:
            # Pour la correction d'erreur : learning rate un peu plus élevé mais toujours faible
            online_lr = 0.0001  # 100x plus faible que le learning rate initial (0.01)
        
        self.nn.learning_rate = online_lr
        print(f"  Learning rate: {online_lr} (original: {original_lr})")
        
        try:
            # Créer un batch avec une seule image
            batch = [user_image]
            labels = [correct_label]
            
            # Calculer forward prop
            batch_tensor = torch.tensor(batch, dtype=torch.float32)
            output_model = self.nn.forward_prop(batch_tensor, device_id)
            output_target = self.nn.label_to_vect(labels)
            
            # Calculer les gradients avec backprop
            gradients = self.nn.back_prop(output_model, output_target, device_id)
            
            # Appliquer directement les gradients (pas besoin de moyenne pour une seule image)
            if gradients is not None:
                self.nn._compute_gradient(gradients, device_id)
                if reinforcement:
                    print(f"✓ Renforcement positif terminé pour le label {correct_label}")
                else:
                    print(f"✓ Correction terminée pour le label {correct_label}")
            else:
                print("✗ Erreur: aucun gradient calculé")
        finally:
            # Restaurer le learning rate original
            self.nn.learning_rate = original_lr
    # END FUNCTION
