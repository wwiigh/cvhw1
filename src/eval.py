import torch
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from model import get_model50
from dataset import get_val_dataloader
from utils import transform_val


def val(path):
    """Calculate the val acc"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model50().to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    valdir = "data/val"
    val_dataloader = get_val_dataloader(valdir, transform=transform_val,
                                        batch_size=1, shuffle=True)

    correct = 0
    total = len(val_dataloader)
    false = []
    for (image, label) in tqdm(val_dataloader):
        image = image.to(device)

        output = model(image)
        output = output.argmax(dim=1).item()
        if output == label:
            correct += 1
        else:
            false.append(label.item())

    print("points:", correct / total, " correct:", correct)

    print(Counter(false))


def matrix(path):
    """Draw the confusion matrix"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model50().to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    valdir = "data/val"
    val_dataloader = get_val_dataloader(valdir, transform=transform_val,
                                        batch_size=1, shuffle=True)

    true_label = []
    pred_label = []
    for (image, label) in tqdm(val_dataloader):
        image = image.to(device)

        output = model(image)
        output = output.argmax(dim=1).item()
        true_label.append(label)
        pred_label.append(output)

    cm = confusion_matrix(true_label, pred_label, labels=np.arange(100))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", square=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")
    plt.savefig("confusion matrix")


if __name__ == "__main__":
    val("model/strongbaseline/exp112_76_loss.pth")
