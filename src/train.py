import os

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dataset import get_train_dataloader, get_val_dataloader
from utils import combined_loss, transform_val, val_loss_fn
from utils import transform_random, mixup_criterion
from model import get_model50


def train():
    """Start training"""
    exp_dir = "exp112"
    if not os.path.exists(f"model/{exp_dir}"):
        os.makedirs(f"model/{exp_dir}")

    writer = SummaryWriter(f"logs/{exp_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    batch_size = 64
    epochs = 100
    learning_rate = 5e-4
    weight_decay = 5e-4
    alpha = 0.2

    train_dir = "data/train"
    val_dir = "data/val"

    train_dataloader = get_train_dataloader(train_dir,
                                            transform=transform_random,
                                            batch_size=batch_size,
                                            shuffle=True)
    val_dataloader = get_val_dataloader(val_dir, transform=transform_val,
                                        batch_size=1, shuffle=True)

    model = get_model50().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=3)

    best_loss = 100
    best_correct = 0
    for epoch in range(epochs):

        running_loss = 0
        correct = 0

        model.train()
        for (image, label) in tqdm(train_dataloader,
                                   desc=f"Epoch {epoch+1}/{epochs}"):

            image = image.to(device)
            label = label.to(device)

            index = torch.randperm(image.size(0)).to(device)
            lam = np.random.beta(alpha, alpha)
            mixed_image = lam * image + (1 - lam) * image[index, :]
            label_a, label_b = label, label[index]
            mixed_label = lam * label_a + (1 - lam) * label_b

            optimizer.zero_grad()
            output = model(mixed_image)
            predictions = output.argmax(dim=1)
            correct += (predictions == mixed_label).sum().item()
            loss = mixup_criterion(combined_loss, output,
                                   label_a, label_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: \
              {running_loss/(len(train_dataloader)):.4f}, \
              acc: {correct/(len(train_dataloader.dataset))}")
        writer.add_scalar("Loss/epoch", running_loss/(len(train_dataloader)),
                          epoch)

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for (image, label) in tqdm(val_dataloader, desc="val"):
                image = image.to(device)

                output = model(image)
                val_loss += val_loss_fn(output, label.to(device)).item()
                output = output.argmax(dim=1).item()
                if output == label:
                    correct += 1

        if val_loss/len(val_dataloader) < best_loss:
            print(f"find best model in epoch:{epoch+1}, \
                  loss:{val_loss/len(val_dataloader)}")
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                },
                f"model/{exp_dir}/{exp_dir}_{epoch}_loss.pth"
            )
            best_loss = val_loss/len(val_dataloader)
            if correct > best_correct:
                best_correct = correct
        elif correct > best_correct:
            print(f"find best model in epoch:{epoch+1}, \
                  loss:{val_loss/len(val_dataloader)}")
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                },
                f"model/{exp_dir}/{exp_dir}_{epoch}_acc.pth"
            )
            best_correct = correct
        else:
            print(f"not find best model in epoch:{epoch+1}, \
                  loss:{val_loss/len(val_dataloader)}")

        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: \
              {val_loss/len(val_dataloader):.4f}, \
              points {correct/len(val_dataloader)}")

        scheduler.step(val_loss/len(val_dataloader))
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")

        writer.add_scalar("Learning Rate", current_lr, epoch)
        writer.add_scalar("eval/epoch", correct/len(val_dataloader), epoch)
        writer.add_scalar("val loss/epoch",
                          val_loss/len(val_dataloader), epoch)

    writer.close()

    print("save model")
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        },
        f"model/{exp_dir}/{exp_dir}_{epoch}_final.pth"
    )


if __name__ == "__main__":
    train()
