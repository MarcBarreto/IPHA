import os
import sys
import Utils
from Resnet import ResNet
from IPHA import IPHA_GA
from torchvision import transforms as T

def main():
    save_path = '../'

    if len(sys.argv) > 4:
        image_path = sys.argv[1]
        model_path = sys.argv[2]
        label_constant = sys.argv[3]
        save_path = sys.argb[4]
    elif len(sys.argv) == 3:
        image_path = sys.argv[1]
        model_path = sys.argv[2]
        label_constant = sys.argv[3]
    else:
        raise ValueError("Insufficient arguments. Please provide at least the image path, model path, the constant type (black, white, gaussian, or norm), and optionally the save path.")

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    resnet = ResNet(name = 'resnet18', pt = True, transform = transform)

    resnet.load_model(model_path)

    class2idx = {'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }

    if label_constant == 'black':
        constant = 0
    elif label_constant == 'white':
        constant = 255
    elif label_constant in ['gaussian', 'norm']:
        constant = None
    else:
        raise ValueError(f"Invalid label_constant: {label_constant}")

    ipha = IPHA_GA(resnet, constant, 1000, 50, select = 10)

    if os.path.isdir(image_path):
        Utils.run_images(image_path, ipha, label_constant, class2idx, save_path)
    elif os.path.isfile(image_path):
        Utils.run_image(image_path, ipha, class2idx)
    else:
        raise ValueError(f"The specified image path '{image_path}' is neither a valid file nor a directory.")


if __name__ == '__main__':
    main()