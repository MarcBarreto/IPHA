import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def plot(x, y = None, title = None, xlabel = None, ylabel = None, save = False, path = None):
    """
    Plots a graph based on the provided x and y data, with optional customization for labels, title, and saving.

    This function creates a line plot for the given `x` and `y` values. If `y` is not provided, the function will plot 
    `x` against a simple range from 1 to the length of `x`. Additionally, the function allows setting the title, 
    xlabel, and ylabel, and optionally saving the plot to a specified path.

    Parameters:
    - x: The data for the x-axis. This should be a sequence (e.g., list or array).
    - y: The data for the y-axis. This is optional; if not provided, the function will plot `x` against a range from 1 to `len(x)`.
    - title: The title of the plot. This is optional.
    - xlabel: The label for the x-axis. This is optional.
    - ylabel: The label for the y-axis. This is optional.
    - save: A boolean that determines whether the plot should be saved as an image file. Default is `False`.
    - path: The file path where the plot should be saved if `save` is `True`. This is optional.
    """
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

def run_image(path, ipha, label):
    """
    Processes and analyzes an image using the given model and label, and compares important vs non-important features.

    This function loads an image from the specified path, then uses the `ipha` model to generate important and 
    non-important features based on the provided label. It also compares the original image with the important and 
    non-important features using the `compare_images` method of the `ipha` model.

    Parameters:
    - path: The file path to the image that will be processed.
    - ipha: The model or method that processes the image, which generates important and non-important features.
    - label: The label corresponding to the image that is used for feature generation and analysis.

    Returns:
    - x_important: The important features of the image as determined by the model.
    - x_non_important: The non-important features of the image as determined by the model.
    """
    image = Image.open(path)
    x_important, x_non_important = ipha(image, label)
    ipha.compare_images(image, label, x_important, x_non_important)

    return x_important, x_non_important

def run_images(folder_path, ipha, label_constant, class2idx, save_path='./'):
    """
    Processes a folder of images and extracts important and non-important features for each image.

    This function loads and processes each image in the specified folder, applying the given `ipha` model to extract 
    important and non-important features for each image based on the provided `label_constant` and `class2idx` mapping. 
    The results, including scores and feature information, are saved into a CSV file.

    Parameters:
    - folder_path: The directory containing the images to be processed.
    - ipha: The model or method that processes each image, extracting important and non-important features.
    - label_constant: The constant value used for comparison (e.g., a pre-defined constant that influences the analysis).
    - class2idx: A dictionary mapping class names or labels (from folder names) to integer indices used by the model.
    - save_path: The directory where the results CSV file will be saved. Defaults to the current working directory.

    Returns:
    - None: This function saves the results as a CSV file but does not return any value.

    Notes:
    - Images in the specified folder are assumed to have `.jpg` or `.png` extensions and are processed in a loop.
    - The function processes each image by loading it, extracting features using `ipha(image, label)`, and then calling 
    `ipha.compare_images()` to compare the extracted features.
    - The results are stored in a dictionary and then converted to a Pandas DataFrame, which is saved as `results.csv` 
    in the specified `save_path`.
    - The `class2idx` mapping uses the folder structure to determine the label for each image.
    - The progress of processing is tracked and displayed using `tqdm`, showing how many images have been processed.
    """

    images = glob.glob(f'{folder_path}/*.[jp][pn]g')

    results = {'image_path': [],
            'features_important': [],
            'features_non_important': [],
            'original_score': [],
            'important_score': [],
            'non_important_score': [],
            'fi_important': [],
            'fi_non_important': [],
            'pr_important': [],
            'pr_non_important': [],
            'constant': []}

    for _, path in enumerate(tqdm(images, desc="Processing Image", unit="image", ncols=100)):
        results['image_path'].append(path)
        
        image = Image.open(path)
        
        label = class2idx[path.split('/')[-2]]
        
        x_important, x_non_important = ipha(image, label)

        results['features_important'].append(x_important)

        results['features_non_important'].append(x_non_important)
        
        org_scr, impt_scr, non_impt_scr, fi_impt, fi_non_impt, pr_impt, pr_non_impt = ipha.compare_images(image, label, x_important, x_non_important, show = False)

        results['original_score'].append(org_scr)
        results['important_score'].append(impt_scr)
        results['non_important_score'].append(non_impt_scr)
        results['fi_important'].append(fi_impt)
        results['fi_non_important'].append(fi_non_impt)
        results['pr_important'].append(pr_impt)
        results['pr_non_important'].append(pr_non_impt)
        results['constant'].append(label_constant)

    df = pd.DataFrame(results)
    df.to_csv(os.join.path(save_path, 'results.csv'), index=False)