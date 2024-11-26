from PIL import Image
from Resnet import ResNet
from IPHA import IPHA_HC, IPHA_GA
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def main(train_path, test_path):
    train_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomHorizontalFlip()
    ])

    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root = train_path, transform = train_transform)
    test_dataset = ImageFolder(root = test_path, transform = test_transform)

    valid_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - valid_size 

    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_subset, batch_size= 64, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size= 32)
    test_loader = DataLoader(test_dataset, batch_size = 32)

    resnet = ResNet(name = 'resnet18', pt = True, transform = test_transform)
    
    resnet.fit(train_loader, valid_loader, path = f'/kaggle/working/{resnet.name}.pt', epochs = 10)

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

    image = Image.open(image_path)
    label = class2idx[image_path.split('/')[-2]]
    x_important, x_non_important = ipha(image, label)
    ipha.compare_images(image, label, x_important, x_non_important)
    
if __name__ == '__main__':
    main()