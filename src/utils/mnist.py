from mnist import MNIST
import kagglehub, os, random
from pathlib import Path


def get_data_dir():
    """
    Trouve automatiquement le dossier 'data' en remontant depuis le fichier actuel.
    Retourne le chemin absolu vers le dossier 'data' à la racine du projet.
    """
    current_file = Path(__file__).resolve()
    # Remonter depuis src/utils/mnist.py jusqu'à la racine du projet
    project_root = current_file.parent.parent.parent
    data_dir = project_root / "data"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Le dossier 'data' n'a pas été trouvé. Cherché dans: {data_dir}")
    
    return str(data_dir)


def download_mnist():
    # Download latest version
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print("Path to dataset files:", path)
    return path


def extract_training(path: str = None):
    if path is None or path == "":
        data_dir = get_data_dir()
        path = os.path.join(data_dir, "mnist_data", "train")
    
    mndata = MNIST(path)
    images, labels = mndata.load_training()
    return (images, labels, mndata)

def extract_testing(path: str = None):
    if path is None or path == "":
        data_dir = get_data_dir()
        path = os.path.join(data_dir, "mnist_data", "test")
    
    mndata = MNIST(path)
    images, labels = mndata.load_testing()
    return (images, labels, mndata)


def display_image(images, labels=None, mndata=None):
    """
    Affiche une image aléatoire du dataset MNIST.
    
    Args:
        images: Liste des images
        labels: Liste des labels (optionnel)
        mndata: Instance MNIST pour l'affichage (optionnel)
    """
    index = random.randrange(0, len(images))
    
    if mndata is not None:
        # Utiliser la méthode display de l'instance MNIST
        print(mndata.display(images[index]))
    else:
        # Affichage simple si pas d'instance MNIST
        if labels is not None:
            print(f"Image {index}, Label: {labels[index]}")
        else:
            print(f"Image {index}")
        # Afficher les premières valeurs de l'image
        img = images[index]
        print(f"Taille de l'image: {len(img)} pixels")
        print(f"Premières valeurs: {img[:10]}")


