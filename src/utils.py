import os

from torchvision import transforms
import torch.nn as nn
import torch


transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)],
                p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(
                kernel_size=3, sigma=(0.3, 1.5))], p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1),
                                     ratio=(0.3, 3.3), value='random'),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])

transform_no_random = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2),
                            scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.2),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomPosterize(bits=4, p=0.5),
    transforms.RandomSolarize(threshold=128, p=0.5),
    transforms.RandomEqualize(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1),
                             ratio=(0.3, 3.3), value='random'),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_random = transforms.Compose([
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.2),
     transforms.RandomRotation(degrees=15),
     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                             scale=(0.9, 1.1)),
     transforms.ColorJitter(brightness=0.2, contrast=0.2,
                            saturation=0.2, hue=0.1),
     transforms.RandAugment(num_ops=2, magnitude=10),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.RandomErasing(p=0.3, scale=(0.02, 0.1),
                              ratio=(0.3, 3.3), value='random'),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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
class_weights = class_weights ** 0.5

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)


def entropy_loss(pred, target):
    """Return loss"""
    ce_loss = loss_fn(pred, target)
    return ce_loss


val_loss_fn = nn.CrossEntropyLoss()
