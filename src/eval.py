import torch
from tqdm import tqdm
from model import get_model,get_model34
from dataset import get_train_dataloader, get_test_dataloader, get_val_dataloader
from utils import loss_fn, transform_val

def val(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model34().to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    valdir = "data/val"
    val_dataloader = get_val_dataloader(valdir, transform=transform_val, batch_size=1, shuffle=True)
    
    correct = 0
    total = len(val_dataloader)
    false = []
    for (image, label, img_path) in tqdm(val_dataloader):
        image = image.to(device)
        
        output = model(image)
        output = output.argmax(dim=1).item()
        if output == label:
            correct += 1
        else:
            false.append(label.item())

    print("points:", correct / total, " correct:", correct)
    false.sort()
    print(false)

if __name__ == "__main__":
    val("model/exp18/exp18_58.pth")