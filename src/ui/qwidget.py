from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget


class DrawingWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.last_point = None  # Pour stocker le point précédent
        self.lines = []  # Pour stocker toutes les lignes à dessiner

    def mousePressEvent(self, event):
        print("Pressed !")
        self.last_point = event.pos()  # Enregistrer le point de départ

    def mouseReleaseEvent(self, event):
        print("Release !")
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
        painter.setPen(QPen(QColor(Qt.black), 2, Qt.SolidLine))
        
        # Dessiner toutes les lignes stockées
        for start_point, end_point in self.lines:
            painter.drawLine(start_point, end_point)

    def refresh(self):
        self.lines = []