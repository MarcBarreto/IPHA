import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

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

def run_images(folder_path, ipha, label_constant, class2idx, save_path='./'):
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