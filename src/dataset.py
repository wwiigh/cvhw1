from torch.utils.data import Dataset
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
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

class TestDatasets(Dataset):

    def __init__(self, imgdir, transform = None):
        self.data = []
        self.transform = transform
        for f in os.listdir(imgdir):
            self.data.append(os.path.join(imgdir, f)) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path
