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
        self.model = models.resnet18()
        self.model.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.Dropout(0.3),
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
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(256, 100)
        )

    def forward(self, x):
        output = self.model(x)
        output = self.fc(output)
        return output

class Model50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50()
        self.model.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(1024, 100)
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

