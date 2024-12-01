import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from IPHA import IPHA_GA
from Resnet import infer
from torchvision import datasets, transforms

def reproducibility():
    torch.manual_seed(50)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(50)

def cpu(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cpu()
    return ob

def cuda(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cuda()
    return ob

def batch_sublogits(torch_model, batch, target):
    batch_logits = torch_model(batch)
    batch_probs = F.softmax(batch_logits)
    batch_features = torch_model.features.detach().data
    list_features = torch.split(batch_features, 1)
    
    list_sublogits = [(torch_model.linear.weight*feature)[target,:].cpu().detach().data.numpy() for feature in list_features]
    
    list_sublogits = [feature.astype(np.float32) for feature in list_sublogits] 
    batch_logits = batch_logits.cpu().detach().data.numpy().astype(np.float32)
    batch_probs = batch_probs.cpu().detach().data.numpy().astype(np.float32)

    return batch_logits, batch_probs, list_sublogits

def batch_layer(torch_model, batch):
    batch_logits = torch_model(batch)
    batch_probs = F.softmax(batch_logits)
    batch_features = torch_model.features.detach().data
    fmaps1 = torch_model.out1.cpu().detach().data
    fmaps2 = torch_model.out2.cpu().detach().data
    fmaps3 = torch_model.out3.cpu().detach().data

    return fmaps1, fmaps2, fmaps3

def get_noise(noise_type):
    img_noise = None
    
    if noise_type == "black":
        img_noise = np.zeros((3, 32, 32), dtype=np.float32)
    elif noise_type == "white":
        img_noise = np.ones((3, 32, 32), dtype=np.float32)
    elif noise_type == "gaussian":
        img_noise = np.ones((3, 32, 32), dtype=np.float32)
        for i in range(32):
            for j in range(32):
                img_noise[0, i, j] = np.random.normal(loc=0.4914, scale=0.2471, size=1)
                img_noise[1, i, j] = np.random.normal(loc=0.4822, scale=0.2435, size=1)
                img_noise[2, i, j] = np.random.normal(loc=0.4465, scale=0.2616, size=1)
    elif noise_type == "norm_mean":
        img_noise = np.concatenate([
            np.ones((1, 32, 32), dtype=np.float32) * 0.4914, 
            np.ones((1, 32, 32), dtype=np.float32) * 0.4822, 
            np.ones((1, 32, 32), dtype=np.float32) * 0.4465], 0)

    return img_noise

def get_cifar10_testset(dir_, batch_size=1):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    num_workers = 2
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return list(test_loader)

def normalize_0_1(img):
    img1 = img.copy()
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    
    return img1

def run_image(model, ipha, testdata, img_id, noise_type, save = True, save_path = None):
    x, y = testdata[img_id]

    img = x
    target = y
    
    topk_high_imgs = []
    topk_high_conf = []
    
    topk_low_imgs = []
    topk_low_conf = []
    
    img_normed = normalize_0_1(np.transpose(img.squeeze().cpu().numpy(), (1, 2, 0)))

    node1, node2, noise = ipha(img_normed, target)

    new_img1 = node1 * img.detach().cpu().numpy() + ((1 - node1)*noise)
    new_img2 = node2 * img.detach().cpu().numpy() + ((1 - node2)* noise)
             
    topk_high_imgs.append(normalize_0_1(np.transpose(new_img1[0], (1, 2, 0))))
    topk_high_conf.append(round(float(infer(model, new_img1, target)), 4))
    
    topk_low_imgs.append(normalize_0_1(np.transpose(new_img2[0], (1, 2, 0))))
    topk_low_conf.append(round(float(infer(model, new_img2, target)), 4))
    
    num_imgs = len(topk_high_imgs)
    _, axs = plt.subplots(1, num_imgs + 1, figsize=((num_imgs + 1) * 5, 10))
    axs[0].imshow(img_normed)
    axs[0].set_title('Original Image')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    for i in range(num_imgs):
        axs[i+1].imshow(topk_high_imgs[i])
        axs[i+1].set_xticks([])
        axs[i+1].set_yticks([])
        axs[i+1].set_title('GA')
    
    if save:
        plt.savefig(os.path.join(save_path, f'image{img_id}_{noise_type}.jpg'), dpi=600, bbox_inches='tight')
    plt.show()

def run_images(model, ipha, testdata, len_imgs, noise_type):
    results = {'image_id': [],
        'features_important': [],
        'features_non_important': [],
        'original_score': [],
        'important_score': [],
        'non_important_score': [],
        'fi_important': [],
        'fi_non_important': [],
        'constant': []}
    
    ipha = IPHA_GA(model, get_noise(noise_type), 1000, 50, select = 10, cf = 0.001)
    
    for idx, _ in enumerate(tqdm(range(len_imgs), desc="Processing Image", unit="image", ncols=100)):
        x, y = testdata[idx]
    
        img = x
        target = y
    
        img_normed = normalize_0_1(np.transpose(img.squeeze().cpu().numpy(), (1, 2, 0)))

        results['image_id'].append(idx)
        
        node1, node2, noise = ipha(img_normed, target)

        x_important = np.array([node1 * img_normed.transpose(2, 0, 1) + (1 - node1) * noise])
        x_non_important = np.array([node2 * img_normed.transpose(2, 0, 1) + (1 - node2) * noise])

        results['features_important'].append(x_important)
    
        results['features_non_important'].append(x_non_important)
        
        org_scr, impt_scr, non_impt_scr, fi_impt, fi_non_impt = ipha.compare_images(img.detach().cpu().numpy(), target, x_important, x_non_important)
    
        results['original_score'].append(org_scr.item())
        results['important_score'].append(impt_scr.item())
        results['non_important_score'].append(non_impt_scr.item())
        results['fi_important'].append(fi_impt.item())
        results['fi_non_important'].append(fi_non_impt.item())
        results['constant'].append(noise_type)

    df = pd.DataFrame(results)
    df.to_csv(f'../{noise_type}_results.csv', index=False)