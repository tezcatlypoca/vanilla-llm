from PyQt5 import *
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout, QHBoxLayout
from qwidget import DrawingWidget
import sys


app = QApplication(sys.argv)

window = QMainWindow()

# Créer le widget central avec un layout vertical
central_widget = QWidget()
main_layout = QVBoxLayout()
central_widget.setLayout(main_layout)

# Canvas en haut (prend tout l'espace disponible)
canva = DrawingWidget()
canva.setFixedSize(560, 560)
main_layout.addWidget(canva)

# Container pour le bouton centré en bas
button_container = QWidget()
button_layout = QHBoxLayout()
button_container.setLayout(button_layout)

# Ajouter un espace flexible à gauche et à droite pour centrer le bouton
button_layout.addStretch()
reset_button = QPushButton("Reset")
reset_button.pressed.connect(canva.refresh)
button_layout.addWidget(reset_button)
button_layout.addStretch()

# Ajouter le container du bouton en bas
main_layout.addWidget(button_container)

window.setCentralWidget(central_widget)

window.setMinimumSize(QSize(600, 600))
window.show()

app.exec()

