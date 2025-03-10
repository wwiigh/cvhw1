from torchvision.transforms import transforms
import torch.nn as nn
import os
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
    
transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)) ,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05))], p=0.4),
            #transforms.CenterCrop(224), 
            #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)],p=0.4), 
            #transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.5))],p=0.4),
            #transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 1.5)),
            
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
class_weights = class_weights ** 0.5

# 設定損失函數的權重
alpha_ce = 0.8  # CrossEntropy 的權重
alpha_focal = 0.2  # FocalLoss 的權重

# 初始化損失函數
focal_loss_fn = FocalLoss(alpha=class_weights,gamma=1)
ce_loss_fn = nn.CrossEntropyLoss()

def combined_loss(pred, target):
    ce_loss = ce_loss_fn(pred, target)
    focal_loss = focal_loss_fn(pred, target)
    return alpha_ce * ce_loss + alpha_focal * focal_loss  # 加權平均

#loss_fn = FocalLoss()
val_loss_fn = nn.CrossEntropyLoss()