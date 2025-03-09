from torchvision.transforms import transforms
import torch.nn as nn
import os
import torch

transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            #transforms.CenterCrop(224), 
            #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.RandomGrayscale(p=0.1),
            #transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 1.5)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

train = []
dir = os.scandir("data/train")
for d in dir:
    if d.is_dir():
        train.append(len(os.listdir(d)))

train = torch.tensor(train, dtype=torch.float)
class_weights = 1.0 / train
class_weights /= class_weights.sum()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
val_loss_fn = nn.CrossEntropyLoss()