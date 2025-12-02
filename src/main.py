from neural_network import NeuralNetwork
import os
import glob

if __name__ == "__main__":
    nn = NeuralNetwork()
    
    # Vérifier s'il existe des modèles sauvegardés
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
        nn.load_model(latest_model)
        
        # Lancer uniquement le test
        nn.run("TESTING")
    else:
        # Aucun modèle sauvegardé, lancer l'entraînement puis le test
        print("Aucun modèle sauvegardé trouvé. Lancement de l'entraînement...")
        nn.run("TRAINING")
        nn.run("TESTING")