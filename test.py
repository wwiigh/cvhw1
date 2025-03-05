
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from src.dataset import TrainDatasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.utils import transform
# model = models.resnet18()
# print(model)

# print("now eval")
# model.eval()

# # 轉換圖片的 Transform
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 調整圖片大小
#     transforms.ToTensor(),  # 轉換成 Tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
# ])

# # 載入圖片
# image_path = "data/train/0/0f0aed51-5899-4336-98cd-03fc0516e2cd.jpg"  # 你的測試圖片
# image = Image.open(image_path).convert("RGB")  # 確保是 RGB 格式

# # 應用 transform
# input_tensor = transform(image).unsqueeze(0)  # 增加 batch 維度

# with torch.no_grad():
#     output = model(input_tensor)  # 模型輸出 logits

# # 取得預測類別
# predicted_class = output.argmax(dim=1).item()

# print(f"模型預測的類別: {predicted_class}")

def show_tensor_image(tensor):
    """將 PyTorch Tensor 轉換為可視化格式並顯示"""
    if tensor.ndimension() == 3 and tensor.shape[0] == 3:  # (C, H, W) 格式
        tensor = tensor.permute(1, 2, 0)  # 轉換為 (H, W, C)

    plt.imshow(tensor)  # Tensor 轉 NumPy
    plt.axis("off")





print("load dataset")
trainDataset = TrainDatasets("data/train", transform=transform)
print("print dataset len")
print(trainDataset.__len__())
train_loader = DataLoader(trainDataset, batch_size=1, shuffle=False)


