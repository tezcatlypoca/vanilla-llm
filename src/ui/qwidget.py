import sys
from pathlib import Path

# Ajouter src/ au PYTHONPATH pour les imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget

from utils.resize_image import *

class DrawingWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.last_point = None  # Pour stocker le point précédent
        self.lines = []  # Pour stocker toutes les lignes à dessiner

        # Modif de la couleur de background
        self.setProperty('active', True)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

    def mousePressEvent(self, event):
        self.last_point = event.pos()  # Enregistrer le point de départ

    def mouseReleaseEvent(self, event):
        self.last_point = None  # Réinitialiser quand on relâche

    def mouseMoveEvent(self, event):
        # Dessiner seulement si le bouton est maintenu (pressed)
        if event.buttons() & Qt.LeftButton and self.last_point is not None:
            current_point = event.pos()
            # Ajouter la ligne à la liste
            self.lines.append((self.last_point, current_point))
            self.last_point = current_point
            self.update()  # Déclencher un repaint

    def paintEvent(self, event):
        # Cette méthode est appelée automatiquement pour redessiner le widget
        painter = QPainter(self)
        
        # Dessiner le fond blanc d'abord pour effacer les anciens dessins
        painter.fillRect(self.rect(), QColor(Qt.white))
        
        # Ensuite dessiner les lignes
        painter.setPen(QPen(QColor(Qt.black), 12, Qt.SolidLine))
        for start_point, end_point in self.lines:
            painter.drawLine(start_point, end_point)

    def save_draw(self):
        # Debug : vérifier combien de lignes sont dessinées
        print(f"\nDEBUG save_draw: {len(self.lines)} lignes à dessiner")
        
        # Forcer un repaint pour s'assurer que tout est dessiné
        self.update()
        self.repaint()
        
        # Attendre un peu pour que le repaint soit terminé
        import time
        time.sleep(0.01)  # 10ms pour s'assurer que le rendu est terminé
        
        # Utiliser grab() pour capturer le widget (inclut tout le contenu dessiné)
        pixmap = self.grab()
        # Convertir QPixmap en QImage
        image = pixmap.toImage()
        resized_img = resize_image(image)
        
        # Debug : vérifier les valeurs de l'image normalisée
        import numpy as np
        img_array = np.array(resized_img)
        print(f"DEBUG Image capturée:")
        print(f"  Min: {img_array.min():.4f}, Max: {img_array.max():.4f}, Mean: {img_array.mean():.4f}")
        print(f"  Pixels < 0.1 (noir): {(img_array < 0.1).sum()}, Pixels > 0.9 (blanc): {(img_array > 0.9).sum()}")
        # Hash de l'image pour vérifier si elle change
        img_hash = hash(tuple(resized_img[:50]))  # Hash des 50 premiers pixels
        print(f"  Hash (50 premiers pixels): {img_hash}")
        
        # Afficher l'image en ASCII art (C'EST L'IMAGE CENTRÉE QUI EST AFFICHÉE)
        print("\n" + "=" * 30)
        print("Image capturée et CENTRÉE (28x28) - Celle envoyée au modèle:")
        print("=" * 30)
        
        # Solution 1 : Utiliser display_image_ascii (fonction custom)
        print(display_image_ascii(resized_img))
        
        print("=" * 30 + "\n")
        
        return resized_img

    def refresh(self):
        # Réinitialiser complètement le canvas
        self.lines = []
        self.last_point = None
        # Forcer un repaint complet pour effacer visuellement
        self.update()
        self.repaint()