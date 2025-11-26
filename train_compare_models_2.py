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
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 固定随机种子 --------------------------
os.environ['PYTHONHASHSEED'] = '2025'
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------- 日志配置 --------------------------
def setup_logger():
    log_dir = "compare_train_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# -------------------------- 数据集类 --------------------------
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

# -------------------------- 2个对比模型 --------------------------
# 1. Xception-based MesoNet
class MesoXceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MesoXceptionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, padding=0) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        out = self.relu(out)
        return self.maxpool(out)

class XceptionMesoNet(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionMesoNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.blocks = nn.Sequential(
            MesoXceptionBlock(8, 16),
            MesoXceptionBlock(16, 32),
            MesoXceptionBlock(32, 64),
            MesoXceptionBlock(64, 128)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x

# 2. Face-X-ray
class FaceXray(nn.Module):
    def __init__(self, num_classes=2):
        super(FaceXray, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# -------------------------- 数据加载与预处理 --------------------------
def load_data(json_file, train_ratio=0.8):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if not isinstance(data.get('images', []), list):
        raise ValueError("data['images'] should be a list")
    images = data['images']
    if len(images) == 0:
        raise ValueError("数据集为空")
    random.shuffle(images)
    train_size = int(len(images) * train_ratio)
    train_data = images[:train_size]
    val_data = images[train_size:]
    logger.info(f"数据划分完成：训练集{len(train_data)}个样本，验证集{len(val_data)}个样本")
    return train_data, val_data

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------- 训练函数 --------------------------
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_accuracy = 0.0
    early_stopping_patience = 10
    no_improvement_epochs = 0
    logger.info(f"\n===== 开始训练 {model_name} =====")
    epoch_progress = tqdm(range(num_epochs), desc=f"[{model_name}] 整体训练进度", leave=True)

    for epoch in epoch_progress:
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch+1} 训练', leave=False)
        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_progress.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        train_progress.close()

        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f'[{model_name}] Epoch {epoch+1}/{num_epochs} - 训练损失：{epoch_loss:.4f}')

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        val_progress = tqdm(val_loader, desc=f'[{model_name}] Epoch {epoch+1} 验证', leave=False)
        with torch.no_grad():
            for images, labels in val_progress:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_progress.set_postfix({"验证样本数": total})
        val_progress.close()

        accuracy = 100 * correct / total
        logger.info(f'[{model_name}] Epoch {epoch+1}/{num_epochs} - 验证准确率：{accuracy:.2f}%')
        epoch_progress.set_postfix({"最佳准确率": f"{best_accuracy:.2f}%", "当前准确率": f"{accuracy:.2f}%"})

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = f"{model_name}_best_model_2025.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f'[{model_name}] 保存最佳模型：{model_path}（准确率：{accuracy:.2f}%）')
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            logger.info(f'[{model_name}] 准确率未提升，连续{no_improvement_epochs}个epoch')

        # 早停机制
        if no_improvement_epochs >= early_stopping_patience:
            logger.info(f'[{model_name}] 早停触发！{early_stopping_patience}个epoch无提升')
            epoch_progress.close()
            break

        # 学习率调度
        scheduler.step()
        logger.info(f'[{model_name}] Epoch {epoch+1} - 学习率更新为：{scheduler.get_last_lr()[0]:.6f}')

    epoch_progress.close()
    logger.info(f"===== {model_name} 训练结束，最佳验证准确率：{best_accuracy:.2f}% =====")
    return best_accuracy, model_path

# -------------------------- 主训练流程 --------------------------
if __name__ == "__main__":
    # 配置参数
    train_json_path = ""
    batch_size = 32
    num_epochs = 50
    lr = 1e-4
    weight_decay = 5e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.FloatTensor([1.0, 1.0]).to(device)  # 权重1:1

    logger.info("=" * 80)
    logger.info("===== 简单Deepfake模型训练 =====")
    logger.info("=" * 80)
    logger.info(f"设备：{device}")
    logger.info(f"训练参数：批次大小={batch_size}，epoch={num_epochs}，学习率={lr}")
    logger.info(f"类别权重：1:1（真实人脸=1.0，AI人脸=1.0）")
    logger.info(f"训练模型：Xception-MesoNet、Face-X-ray")

    # 加载训练/验证数据
    try:
        train_data, val_data = load_data(train_json_path)
        train_dataset = FaceDataset(train_data, transform=transform)
        val_dataset = FaceDataset(val_data, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        logger.error(f"加载训练数据失败：{e}")
        exit()

    # 定义要训练的2个模型
    models_to_train = [
        ("Xception-MesoNet", XceptionMesoNet(num_classes=2)),
        ("Face-X-ray", FaceXray(num_classes=2))
    ]

    # 批量训练
    for model_name, model in models_to_train:
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        logger.info("\n" + "=" * 60)
        logger.info(f"{model_name} 训练配置")
        logger.info("=" * 60)
        train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device
        )

    logger.info("\n所有模型训练完成！")