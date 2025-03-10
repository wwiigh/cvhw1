from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
import os

class TrainDatasets(Dataset):

    def __init__(self, imgdir, transform = None):
        self.data = []
        self.transform = transform
        dir = os.scandir(imgdir)
        for d in dir:
            if d.is_dir():
                path = os.path.join(imgdir, d.name)
                for f in os.listdir(path):
                    self.data.append((os.path.join(path, f), d.name) )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        label = self.data[idx][1]
        image = Image.open(img_path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(label), img_path

class valDatasets(Dataset):

    def __init__(self, imgdir, transform = None):
        self.data = []
        self.transform = transform
        dir = os.scandir(imgdir)
        for d in dir:
            if d.is_dir():
                path = os.path.join(imgdir, d.name)
                for f in os.listdir(path):
                    self.data.append((os.path.join(path, f), d.name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        label = self.data[idx][1]
        image = Image.open(img_path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(label), img_path

class TestDatasets(Dataset):

    def __init__(self, imgdir, transform = None):
        self.data = []
        self.transform = transform
        for f in os.listdir(imgdir):
            self.data.append((os.path.join(imgdir, f), f)) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        image = Image.open(img_path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.data[idx][1][:-4]

def get_train_dataloader(imgdir, transform = None, batch_size=1, shuffle = False):
    train_dataset = TrainDatasets(imgdir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,pin_memory=True)
    return train_dataloader

def get_val_dataloader(imgdir, transform = None, batch_size=1, shuffle = False):
    val_dataset = valDatasets(imgdir, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return val_dataloader

def get_test_dataloader(imgdir, transform = None, batch_size=1, shuffle = False):
    test_dataset = TestDatasets(imgdir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return test_dataloader

