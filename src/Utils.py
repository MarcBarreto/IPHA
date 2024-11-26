import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot(x, y = None, title = None, xlabel = None, ylabel = None, save = False, path = None):
    plt.figure()

    if y is None:
        plt.plot(np.arange(1, len(x) + 1), x)

    else:
        plt.plot(np.arange(1, x), y)
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    
    if ylabel is not None:
        plt.ylabel(ylabel)
        
    if title is not None:
        plt.title(title)
        
    if save:
        plt.savefig(path)
        
    plt.show()
    plt.close()

def run_image(path, ipha, class2idx):
    image = Image.open(path)
    label = class2idx[path.split('/')[-2]]
    x_important, x_non_important = ipha(image, label)
    ipha.compare_images(image, label, x_important, x_non_important)

    return x_important, x_non_important