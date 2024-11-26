import Utils
from PIL import Image
from Resnet import ResNet
from IPHA import IPHA_HC, IPHA_GA
from torchvision import transforms as T

def main():
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    resnet = ResNet(name = 'resnet18', pt = True, transform = transform)

    resnet.load_model(f'/kaggle/input/resnet18/pytorch/default/1/{resnet.name}.pt')

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
    
    image_path = '/kaggle/input/cifar10-pngs-in-folders/cifar10/test/dog/0001.png'
    ipha = IPHA_GA(resnet, 0, 1000, 100, select = 10)
    
    Utils.run_images(image_path, ipha, class2idx)
    
if __name__ == '__main__':
    main()