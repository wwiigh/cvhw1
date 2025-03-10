import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import get_train_dataloader, get_test_dataloader, get_val_dataloader
from utils import combined_loss, transform, transform_val, val_loss_fn
from model import get_model, get_model100, get_model34, get_model50, get_model100Update
from torchvision.transforms.functional import to_tensor
import numpy as np
import os

def mixup_data(x, y, alpha=1.0):
    """Applies Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y

def train():
    expdir = "exp40"
    if not os.path.exists(f"model/{expdir}"):
        os.makedirs(f"model/{expdir}")

    writer = SummaryWriter(f"logs/{expdir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)

    #改成128, learning rate改掉
    batch_size = 64
    epochs = 70
    learning_rate = 0.001
    weight_decay=5e-4
    setp_size = 10
    T_max = 100
    gamma = 0.8

    traindir = "data/train"
    valdir = "data/val"
    testdir = "data/test"

    train_dataloader = get_train_dataloader(traindir, transform=transform, batch_size=batch_size, shuffle=True)
    val_dataloader = get_val_dataloader(valdir, transform=transform_val, batch_size=1, shuffle=True)

    model = get_model100Update().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=5)
    

    bestcorrect = 0
    for epoch in range(epochs):
        
        runningloss = 0
        correct = 0

        model.train()
        for (image, label, img_path) in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):

            image = image.to(device)
            label = label.to(device)
            #image, label = mixup_data(image, label)
            optimizer.zero_grad()
            output = model(image)
            predictions = output.argmax(dim=1)
            correct += (predictions == label).sum().item()
            loss = combined_loss(output, label)
            loss.backward()
            optimizer.step()

            runningloss += loss.item()
            

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {runningloss/len(train_dataloader):.4f}, acc: {correct/len(train_dataloader.dataset)}")
        writer.add_scalar("Loss/epoch", loss, epoch)

        model.eval()
        valloss = 0
        correct = 0
        with torch.no_grad():
            for (image, label, img_path) in tqdm(val_dataloader, desc="val"):
                image = image.to(device)

                output = model(image)
                valloss += val_loss_fn(output, label.to(device)).item()
                output = output.argmax(dim=1).item()
                if output == label:
                    correct += 1

            if correct > bestcorrect:
                print(f"find best model in epoch:{epoch+1}, points:{correct/len(val_dataloader)}")
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),},
                            f"model/{expdir}/{expdir}_{epoch}.pth")
                bestcorrect = correct
            else:
                print(f"not find best model in epoch:{epoch+1}, points:{correct/len(val_dataloader)}")

        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {valloss/len(val_dataloader):.4f}")

        scheduler.step(valloss/len(val_dataloader))
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")

        writer.add_scalar("Learning Rate", current_lr, epoch)
        writer.add_scalar("eval/epoch", correct/len(val_dataloader), epoch)
        writer.add_scalar("val loss/epoch", valloss/len(val_dataloader), epoch)
    
    writer.close()

    print("save model")
    torch.save(model.state_dict(), f"model/{expdir}/{expdir}_final_model.pth")
    
def traincheckpoint():
    expdir = "exp38"
    if not os.path.exists(f"model/{expdir}"):
        os.makedirs(f"model/{expdir}")
    writer = SummaryWriter(f"logs/{expdir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)

    batch_size = 64
    epochs = 35
    learning_rate = 0.0001
    weight_decay=5e-4
    setp_size = 10
    gamma = 0.5

    traindir = "data/train"
    valdir = "data/val"
    testdir = "data/test"

    train_dataloader = get_train_dataloader(traindir, transform=transform, batch_size=batch_size, shuffle=True)
    val_dataloader = get_val_dataloader(valdir, transform=transform_val, batch_size=1, shuffle=True)

    model = get_model100Update().to(device)
    checkpoint = torch.load("model/exp36/exp36_63.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=setp_size, gamma=gamma)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #sscheduler.step_size = 15
    print("check setting")
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Scheduler last epoch: {scheduler.last_epoch}")

    bestcorrect = 0
    for epoch in range(epochs):
        
        runningloss = 0
        correct = 0

        model.train()
        for (image, label, img_path) in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):

            image = image.to(device)
            label = label.to(device)
            #image, label = mixup_data(image, label)
            optimizer.zero_grad()
            output = model(image)
            predictions = output.argmax(dim=1)
            correct += (predictions == label).sum().item()
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            runningloss += loss.item()
            

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {runningloss/len(train_dataloader):.4f}, acc: {correct/len(train_dataloader.dataset)}")
        writer.add_scalar("Loss/epoch", loss, epoch)

        model.eval()
        valloss = 0
        correct = 0
        with torch.no_grad():
            for (image, label, img_path) in tqdm(val_dataloader, desc="val"):
                image = image.to(device)

                output = model(image)
                valloss += val_loss_fn(output, label.to(device)).item()
                output = output.argmax(dim=1).item()
                if output == label:
                    correct += 1

            if correct > bestcorrect:
                print(f"find best model in epoch:{epoch+1}, points:{correct/len(val_dataloader)}")
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),},
                            f"model/{expdir}/{expdir}_{epoch}.pth")
                bestcorrect = correct
            else:
                print(f"not find best model in epoch:{epoch+1}, points:{correct/len(val_dataloader)}")

        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {valloss/len(val_dataloader):.4f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")

        writer.add_scalar("Learning Rate", current_lr, epoch)
        writer.add_scalar("eval/epoch", correct/len(val_dataloader), epoch)
        writer.add_scalar("val loss/epoch", valloss/len(val_dataloader), epoch)
    
    writer.close()

    print("save model")
    torch.save(model.state_dict(), f"model/{expdir}/{expdir}_final_model.pth")

if __name__ == "__main__":
    train()