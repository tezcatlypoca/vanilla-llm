from utils.mnist import *
import os


res = extract_training()
images, labels, mndata = res
display_image(images, labels, mndata)