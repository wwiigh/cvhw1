import torch
from tqdm import tqdm
from model import get_model,get_model34, get_model100Update, get_model50
from dataset import get_train_dataloader, get_test_dataloader, get_val_dataloader
from utils import  transform_val
from collections import Counter

def val(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model50().to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    valdir = "data/val"
    val_dataloader = get_val_dataloader(valdir, transform=transform_val, batch_size=1, shuffle=True)
    
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

if __name__ == "__main__":
    val("model/exp72/exp72_3.pth")