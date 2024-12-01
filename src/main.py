import os
import sys
import Utils
import warnings
from IPHA import IPHA_GA
from Resnet import get_resnet_cifar10
from torchvision import transforms as T

def main():
    warnings.filterwarnings("ignore")
    Utils.reproducibility()

    optional = None

    if len(sys.argv) == 5:
        type = int(sys.argv[1])
        model_path = sys.argv[2]
        noise_type = sys.argv[3]
        img_id = int(sys.argv[4])

    elif len(sys.argv) > 5:
        type = int(sys.argv[1])
        model_path = sys.argv[2]
        noise_type = sys.argv[3]
        img_id = int(sys.argv[4])
        optional = sys.argv[5]
    else:
        raise ValueError("Insufficient arguments. Please provide at least the Type (0 for processing one image or 1 for processing multiple images), model path, noise type (black, white, gaussian, or norm_mean), image ID (0, 1, ... 9999) or the number of images to process, and the save path only if you wish to save the image.")

    resnet = get_resnet_cifar10(model_path)

    testdata = Utils.get_cifar10_testset('../')

    ipha = IPHA_GA(resnet, Utils.get_noise(noise_type), 1000, 50, select = 10, cf = 0.001)

    if type == 0:
        if optional:
            Utils.run_image(resnet, ipha, testdata, img_id, noise_type, save = True, save_path = optional)
        else:
            Utils.run_image(resnet, ipha, testdata, img_id, noise_type, save = False)
    elif type == 1:
        Utils.run_images(resnet, ipha, testdata, img_id, noise_type)        

if __name__ == '__main__':
    main()