from scipy.ndimage import zoom
from PyQt5.QtGui import QImage
import numpy as np

def resize_image(image: QImage):
    # Grayscale convertion
    resized_image = image.convertToFormat(QImage.Format_Grayscale8)

    # Convertion numpy array
    width = resized_image.width()
    height = resized_image.height()
    ptr = resized_image.bits()
    ptr.setsize(resized_image.byteCount())
    arr = np.array(ptr).reshape(height, width)

    arr_28 = zoom(arr, (28/height, 28/width), order=1).astype(np.uint8)

    print(arr_28)

    return arr_28

