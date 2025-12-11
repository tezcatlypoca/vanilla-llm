from scipy.ndimage import zoom, shift, binary_dilation
from PyQt5.QtGui import QImage
import numpy as np


def display_image_ascii(image_normalized, width=28, threshold=0.5):
    """
    Affiche une image normalisée (0-1) en ASCII art, similaire à mndata.display()
    
    Args:
        image_normalized: Liste de 784 pixels normalisés entre 0 et 1
        width: Largeur de l'image (28 pour MNIST)
        threshold: Seuil pour déterminer si un pixel est noir ou blanc (0.5 = 127.5/255)
    
    Returns:
        String ASCII représentant l'image
    """
    render = ''
    for i, pixel in enumerate(image_normalized):
        if i % width == 0:
            render += '\n'
        # Dans MNIST normalisé : 0.0 = noir (chiffre), 1.0 = blanc (fond)
        # On inverse pour l'affichage : si pixel < threshold, c'est du noir (chiffre) -> '@'
        # Si pixel >= threshold, c'est du blanc (fond) -> '.'
        if pixel < threshold:
            render += '@'  # Pixel sombre (chiffre)
        else:
            render += '.'  # Pixel clair (fond)
    return render

def resize_image(image: QImage):
    # Grayscale convertion
    resized_image = image.convertToFormat(QImage.Format_Grayscale8)

    # Convertion numpy array
    width = resized_image.width()
    height = resized_image.height()
    ptr = resized_image.bits()
    ptr.setsize(resized_image.byteCount())
    arr = np.array(ptr).reshape(height, width)
    
    # Debug : vérifier les valeurs avant redimensionnement
    print(f"Image originale: {width}x{height}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
    print(f"Pixels noirs (<50): {(arr < 50).sum()}, Pixels blancs (>200): {(arr > 200).sum()}")

    # Resize de l'image a l'aide de min
    factor = height // 28
    arr_reshape = arr.reshape(28, factor, 28, factor)
    arr_28 = arr_reshape.min(axis=(1, 3)).astype(np.uint8)
    
    # Debug : vérifier les valeurs après sous-échantillonnage
    print(f"Image 28x28: min={arr_28.min()}, max={arr_28.max()}, mean={arr_28.mean():.1f}")
    print(f"Pixels noirs (<50): {(arr_28 < 50).sum()}, Pixels blancs (>200): {(arr_28 > 200).sum()}")

    # Normaliser entre 0 et 1 (comme format_images dans mnist.py)
    # Dans MNIST : chiffres noirs (0) -> 0.0, fond blanc (255) -> 1.0
    arr_28_normalized = arr_28.astype(np.float32) / 255.0

    # Améliorer le contraste : forcer les valeurs vers 0 (noir) ou 1 (blanc)
    # Cela simule mieux le contraste élevé des images MNIST
    # Pour des images très claires (mean > 0.8), utiliser un threshold plus agressif
    mean_val = arr_28_normalized.mean()
    if mean_val > 0.8:
        # Image très claire : threshold plus bas pour mieux détecter les traits fins
        # On utilise un threshold encore plus bas pour capturer plus de pixels
        threshold_contrast = 0.5  # Threshold plus élevé pour capturer les pixels gris
    else:
        # Image normale : threshold adaptatif
        threshold_contrast = mean_val * 0.7
    
    arr_28_normalized = np.where(arr_28_normalized < threshold_contrast, 0.0, 1.0)
    pixels_noirs_avant = (arr_28_normalized == 0.0).sum()
    print(f"DEBUG Contraste: mean={mean_val:.3f}, threshold={threshold_contrast:.3f}, pixels noirs={pixels_noirs_avant}, pixels blancs={(arr_28_normalized == 1.0).sum()}")
    
    # Dilatation morphologique pour épaissir les traits (simule mieux les traits épais de MNIST)
    # Structure 3x3 complète pour épaissir plus agressivement
    structure = np.ones((3, 3), dtype=bool)  # Structure complète 3x3 pour plus d'épaississement
    arr_28_binary = (arr_28_normalized == 0.0)  # Convertir en booléen (True = pixel noir)
    # 2 itérations pour épaissir davantage et se rapprocher de MNIST (500-600 pixels noirs)
    arr_28_dilated = binary_dilation(arr_28_binary, structure=structure, iterations=2)
    arr_28_normalized = np.where(arr_28_dilated, 0.0, 1.0)  # Reconvertir en 0.0/1.0
    pixels_noirs_apres = (arr_28_normalized == 0.0).sum()
    mean_apres = arr_28_normalized.mean()
    print(f"DEBUG Dilatation: pixels noirs avant={pixels_noirs_avant}, après={pixels_noirs_apres}, gain={pixels_noirs_apres - pixels_noirs_avant}")
    print(f"  Mean après: {mean_apres:.3f} (objectif: ~0.10-0.20 comme MNIST, actuel: {mean_apres:.3f})")

    # Debug : afficher l'image avant centrage
    arr_flat_before = arr_28_normalized.flatten().tolist()
    arr_before_np = np.array(arr_flat_before)
    hash_before = hash(tuple(arr_flat_before))
    print(f"DEBUG Avant centrage: hash complet={hash_before}")
    print(f"  Pixels < 0.1: {(arr_before_np < 0.1).sum()}, Pixels > 0.9: {(arr_before_np > 0.9).sum()}")
    
    # Centre l'image pour se rapprocher du format MNIST
    arr_centered = center_image(arr_flat_before)
    
    # Debug : afficher l'image après centrage
    arr_after_np = np.array(arr_centered)
    hash_after = hash(tuple(arr_centered))
    diff_total = np.abs(arr_before_np - arr_after_np).sum()
    print(f"DEBUG Après centrage: hash complet={hash_after}, changé={'OUI' if hash_before != hash_after else 'NON'}")
    print(f"  Différence totale: {diff_total:.6f} (devrait être > 0 si centrage effectif)")
    print(f"  Pixels < 0.1: {(arr_after_np < 0.1).sum()}, Pixels > 0.9: {(arr_after_np > 0.9).sum()}")

    return arr_centered  # Retourner une liste comme format_images()

def center_image(image: list):
    cx, cy = find_centroide(image, 28)
    offset = compute_translation(28, cx, cy)
    
    # Debug : afficher les informations de centrage
    print(f"DEBUG Centrage: centroïde=({cx:.2f}, {cy:.2f}), offset=({offset[0]:.2f}, {offset[1]:.2f})")
    
    centered_image = apply_translation(image, offset, 28)
    
    return centered_image

def find_centroide(image: list, width: int) -> tuple:
    arr = np.array(image).reshape(width, width)
    
    # Threshold adaptatif : utiliser la moyenne comme seuil pour mieux détecter les pixels sombres
    # Pour des images très claires, un threshold fixe de 0.5 peut ne pas fonctionner
    mean_val = arr.mean()
    threshold = min(0.5, mean_val * 0.8)  # Utiliser 80% de la moyenne ou 0.5 max
    
    mask = arr < threshold
    
    # Debug : afficher les informations de détection
    num_pixels_chiffre = mask.sum()
    print(f"DEBUG Centroïde: threshold={threshold:.3f}, pixels du chiffre détectés={num_pixels_chiffre}")

    if not mask.any():
        print("DEBUG Centroïde: Aucun pixel détecté, retour du centre par défaut")
        return width // 2, width // 2
    
    y_coords, x_coords = np.where(mask)
    
    cx = np.mean(x_coords)
    cy = np.mean(y_coords)

    return cx, cy

def compute_translation(width: int, cx: float, cy: float) -> tuple:
    offset_x =  (width//2) - cx
    offset_y =  (width//2) - cy

    return offset_x, offset_y

def apply_translation(image: list, offset: tuple, width: int) -> list:
    arr = np.array(image).reshape(width, width)
    
    # Debug : vérifier l'image avant translation
    print(f"DEBUG Translation: offset=({offset[0]:.2f}, {offset[1]:.2f})")
    print(f"  Image avant: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    print(f"  Pixels < 0.1: {(arr < 0.1).sum()}, Pixels > 0.9: {(arr > 0.9).sum()}")
    
    offset_y, offset_x = offset[1], offset[0]
    
    # Si l'offset est très petit, on l'applique quand même mais avec order=0 (pas d'interpolation)
    # pour éviter les artefacts d'interpolation sur des images binaires
    if abs(offset_y) < 0.1 and abs(offset_x) < 0.1:
        print(f"  Offset très petit (< 0.1), pas de translation")
        return image  # Pas de translation si l'offset est négligeable
    
    # Utiliser order=0 (plus proche voisin) pour les images binaires (0 ou 1)
    # Cela évite les valeurs intermédiaires créées par l'interpolation
    translated = shift(arr, (offset_y, offset_x), mode='constant', cval=1.0, order=0)
    
    # Rebinariser l'image après translation (au cas où l'interpolation créerait des valeurs intermédiaires)
    translated = np.where(translated < 0.5, 0.0, 1.0)
    
    # Debug : vérifier l'image après translation
    print(f"  Image après: min={translated.min():.4f}, max={translated.max():.4f}, mean={translated.mean():.4f}")
    print(f"  Pixels < 0.1: {(translated < 0.1).sum()}, Pixels > 0.9: {(translated > 0.9).sum()}")
    
    # Vérifier si l'image a vraiment changé
    diff = np.abs(arr - translated).sum()
    print(f"  Différence totale: {diff:.6f} (devrait être > 0 si translation effective)")

    return translated.flatten().tolist()