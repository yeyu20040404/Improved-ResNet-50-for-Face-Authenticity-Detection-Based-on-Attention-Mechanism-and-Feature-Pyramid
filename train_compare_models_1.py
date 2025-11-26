import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
import logging
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# -------------------------- 基础配置（无修改）--------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('train_baselines_weights.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 固定随机种子
os.environ['PYTHONHASHSEED'] = '2025'
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 数据集类
class FaceDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        for img_info in self.data:
            assert os.path.exists(img_info['file_path']), f"Missing file: {img_info['file_path']}"
            assert img_info['label'] in [0, 1], f"Invalid label: {img_info['label']}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_info = self.data[idx]
        image = Image.open(img_info['file_path']).convert('RGB')
        label = int(img_info['label'])
        if self.transform:
            image = self.transform(image)
        return image, label


# 数据预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 数据加载与划分
def load_and_split_data(json_file, train_ratio=0.8):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if not isinstance(data.get('images', []), list):
        raise ValueError("data['images'] should be a list")
    images = data['images']
    if len(images) == 0:
        raise ValueError("数据集为空，请检查 JSON 文件")
    random.shuffle(images)
    train_size = int(len(images) * train_ratio)
    if train_size == 0:
        train_size = 1
    if len(images) - train_size == 0:
        train_size = len(images) - 1
    train_data = images[:train_size]
    val_data = images[train_size:]
    logger.info(f"数据划分完成：训练集大小 {len(train_data)}, 验证集大小 {len(val_data)}")
    return train_data, val_data


# 训练函数（确保保存权重）
def train_model(model, criterion, optimizer, scheduler, num_epochs, model_name):
    best_accuracy = 0.0
    early_stopping_patience = 10
    no_improvement_epochs = 0
    logger.info(f"开始训练基准模型：{model_name}")
    epoch_progress = tqdm(range(num_epochs), desc=f"Overall Progress ({model_name})", leave=True)

    for epoch in epoch_progress:
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1} - Train', leave=False)
        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
        train_progress.close()
        epoch_loss = running_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1} - Val', leave=False)
        with torch.no_grad():
            for images, labels in val_progress:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_progress.close()
        accuracy = 100 * correct / total
        logger.info(f'Epoch {epoch + 1} - 训练损失：{epoch_loss:.4f}，验证准确率：{accuracy:.2f}%')

        # 保存最佳权重（核心：确保权重文件生成）
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            weight_path = f'{model_name}_best_weights_20251118.pth'  # 权重文件名明确
            torch.save(model.state_dict(), weight_path)
            logger.info(f'→ 最佳权重更新：{weight_path}（准确率：{best_accuracy:.2f}%）')
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        # 早停与学习率更新
        epoch_progress.set_postfix({"Best Acc": f"{best_accuracy:.2f}%"})
        if no_improvement_epochs >= early_stopping_patience:
            logger.info(f'早停触发：{model_name} 训练结束')
            epoch_progress.close()
            break
        scheduler.step()

    epoch_progress.close()
    logger.info(f"{model_name} 训练完成，最佳准确率：{best_accuracy:.2f}%")
    return best_accuracy


# -------------------------- 5个基准模型定义--------------------------
class VGG16Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.vgg16.classifier[6] = nn.Linear(self.vgg16.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)


class ResNet34Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet34(x)


class ResNet101Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.resnet101.fc = nn.Linear(self.resnet101.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet101(x)


class DenseNet169Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.densenet169 = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        self.densenet169.classifier = nn.Linear(self.densenet169.classifier.in_features, num_classes)

    def forward(self, x):
        return self.densenet169(x)


class EfficientNetBaseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)


# -------------------------- 数据加载与训练启动 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备：{device}")

# 加载数据
try:
    train_data, val_data = load_and_split_data('annotation.json')
except ValueError as e:
    logger.error(f"数据加载失败：{e}")
    exit()

# 类别权重计算
class_counts = [0, 0]
for img_info in train_data:
    class_counts[img_info['label']] += 1
total = sum(class_counts)
class_weights = [total / count for count in class_counts]
class_weights = torch.FloatTensor(class_weights).to(device)

# 数据加载器
train_dataset = FaceDataset(train_data, transform=transform)
val_dataset = FaceDataset(val_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 批量训练所有基准模型
if __name__ == "__main__":
    baseline_models = [
        ("VGG16", VGG16Baseline),
        ("ResNet34", ResNet34Baseline),
        ("ResNet101", ResNet101Baseline),
        ("DenseNet169", DenseNet169Baseline),
        ("EfficientNet", EfficientNetBaseline)
    ]

    # 训练配置
    num_epochs = 50
    lr = 1e-4
    weight_decay = 5e-4
    step_size = 10
    gamma = 0.1

    all_best_acc = {}
    for model_name, ModelClass in baseline_models:
        logger.info("\n" + "=" * 80)
        logger.info(f"训练模型：{model_name}")
        logger.info("=" * 80)

        # 初始化模型、优化器、损失函数
        model = ModelClass(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        # 训练
        best_acc = train_model(model, criterion, optimizer, scheduler, num_epochs, model_name)
        all_best_acc[model_name] = best_acc

    # 结果汇总
    logger.info("\n" + "=" * 80)
    logger.info("所有基准模型训练完成（权重已保存）")
    logger.info("=" * 80)
    for model_name, acc in sorted(all_best_acc.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{model_name:<20} 最佳准确率：{acc:.2f}%，权重文件：{model_name}_best_weights_20251118.pth")