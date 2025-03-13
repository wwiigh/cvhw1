import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18()
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 100)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        output = self.model(x)
        output = self.dropout(output)
        return output


class Model100(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18()
        self.model.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(256, 100)
        )

    def forward(self, x):
        output = self.model(x)
        output = self.fc(output)
        return output

class Model100UpDate(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        self.model.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        output = self.model(x)
        output = self.fc(output)
        return output


class Model34(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34()
        self.model.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        output = self.model(x)
        output = self.fc(output)
        return output

class Model50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights="IMAGENET1K_V2")
        self.model.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 100),
        )

    def forward(self, x):
        output = self.model(x)
        output = self.fc(output)
        return output

class Model101(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(weights="IMAGENET1K_V2")
        self.model.fc = nn.Sequential(
                        nn.Linear(2048, 100)  # 最後的全連接層，這裡假設有 1000 類別
        )

    def forward(self, x):
        output = self.model(x)
        return output
    
class ModelNext(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnext50_32x4d(weights="IMAGENET1K_V2")
        self.model.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 100),
        )

    def forward(self, x):
        output = self.model(x)
        output = self.fc(output)
        return output

def get_model():
    return Model()

def get_model100():
    return Model100()

def get_model100Update():
    return Model100UpDate()

def get_model34():
    return Model34()

def get_model50():
    return Model50()

def get_model101():
    return Model101()

def get_modelNext():
    return ModelNext()

#print(models.resnext50_32x4d(weights="IMAGENET1K_V2"))
