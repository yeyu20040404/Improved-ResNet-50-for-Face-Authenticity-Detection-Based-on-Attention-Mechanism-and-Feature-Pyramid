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

# -------------------------- 日志配置--------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('train_ablation_study.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------------- 固定所有随机种子--------------------------
os.environ['PYTHONHASHSEED'] = '2025'
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------- 数据集类--------------------------
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

# -------------------------- 消融实验模型定义--------------------------
# 1. ResNet50 Baseline（完全复用官方权重，仅修改最后全连接层）
class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50Baseline, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 替换最后全连接层（保持输入维度一致，输出维度=2）
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)

# 2. ResNet50+FPN
class ResNet50FPN(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50FPN, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.fpn = FPN([256, 512, 1024, 2048], 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * 4, num_classes)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        c2 = self.resnet50.layer1(x)
        c3 = self.resnet50.layer2(c2)
        c4 = self.resnet50.layer3(c3)
        c5 = self.resnet50.layer4(c4)
        fpn_out = self.fpn([c2, c3, c4, c5])
        out = []
        for feature in fpn_out:
            out.append(self.avgpool(feature).flatten(1))
        out = torch.cat(out, dim=1)
        out = self.fc(out)
        return out

# 3. ResNet50+CAM（CAM注意力机制，移除FPN）
class CAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNet50CAM(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50CAM, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cam1 = CAM(256)
        self.cam2 = CAM(512)
        self.cam3 = CAM(1024)
        self.cam4 = CAM(2048)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        c2 = self.resnet50.layer1(x)
        c2 = self.cam1(c2)
        c3 = self.resnet50.layer2(c2)
        c3 = self.cam2(c3)
        c4 = self.resnet50.layer3(c3)
        c4 = self.cam3(c4)
        c5 = self.resnet50.layer4(c4)
        c5 = self.cam4(c5)
        x = self.resnet50.avgpool(c5)
        x = torch.flatten(x, 1)
        x = self.resnet50.fc(x)
        return x

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.reverse()
        for i in range(len(laterals) - 1):
            laterals[i + 1] += self.upsample(laterals[i])
        laterals.reverse()
        out = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return out

# -------------------------- 数据预处理+加载--------------------------
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# -------------------------- 训练函数--------------------------
def train_model(model, model_name, criterion, optimizer, scheduler, num_epochs, train_loader, val_loader, device):
    best_accuracy = 0.0
    early_stopping_patience = 10
    no_improvement_epochs = 0
    logger.info(f"\n===== 开始训练 {model_name} =====")
    epoch_progress = tqdm(range(num_epochs), desc=f"[{model_name}] Overall Training", leave=True)

    for epoch in epoch_progress:
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch+1}/{num_epochs} - Training', leave=False)
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

        # 验证
        model.eval()
        correct = 0
        total = 0
        val_progress = tqdm(val_loader, desc=f'[{model_name}] Epoch {epoch+1}/{num_epochs} - Validation', leave=False)
        with torch.no_grad():
            for images, labels in val_progress:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_progress.set_postfix({"Validated Samples": total})
        val_progress.close()

        accuracy = 100 * correct / total
        logger.info(f'[{model_name}] Epoch {epoch+1}/{num_epochs} - 验证准确率：{accuracy:.2f}%')
        epoch_progress.set_postfix({"Best Acc": f"{best_accuracy:.2f}%", "Current Acc": f"{accuracy:.2f}%"})

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = f'{model_name}_20251119_ablation.pth'
            torch.save(model.state_dict(), model_path)
            logger.info(f'[{model_name}] Epoch {epoch+1}/{num_epochs} - 最佳模型更新：{model_path}（准确率 {accuracy:.2f}%）')
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            logger.info(f'[{model_name}] Epoch {epoch+1}/{num_epochs} - 准确率未提升，已连续 {no_improvement_epochs} 个epoch')

        # 早停
        if no_improvement_epochs >= early_stopping_patience:
            logger.info(f'[{model_name}] 早停触发：{early_stopping_patience} 个epoch准确率未提升')
            epoch_progress.close()
            break

        scheduler.step()
        logger.info(f'[{model_name}] Epoch {epoch+1}/{num_epochs} - 学习率更新：{scheduler.get_last_lr()[0]:.6f}')

    epoch_progress.close()
    logger.info(f"===== {model_name} 训练结束，最佳验证准确率：{best_accuracy:.2f}% =====")
    return best_accuracy

# -------------------------- 主流程--------------------------
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备：{device}")

    # 加载数据
    try:
        train_data, val_data = load_and_split_data('annotation.json')
    except ValueError as e:
        logger.error(f"数据加载失败：{e}")
        exit()
    if len(train_data) == 0 or len(val_data) == 0:
        logger.error("训练集或验证集为空")
        raise ValueError("训练集或验证集为空")

    # 固定类别权重
    class_weights = torch.FloatTensor([1.0, 1.0]).to(device)
    logger.info(f"固定类别权重：类别0（真实人脸）1.0, 类别1（AI人脸）1.0")

    # 数据加载器
    train_dataset = FaceDataset(train_data, transform=transform)
    val_dataset = FaceDataset(val_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    logger.info(f"数据加载器创建完成：训练集批次32，验证集批次32")

    # 消融实验配置（3个模型，参数完全一致）
    ablation_models = [
        ("ResNet50_Baseline", ResNet50Baseline(num_classes=2)),
        ("ResNet50_FPN", ResNet50FPN(num_classes=2)),
        ("ResNet50_CAM", ResNet50CAM(num_classes=2))
    ]

    # 记录所有模型的最佳结果
    ablation_results = {}

    # 循环训练每个消融模型
    for model_name, model in ablation_models:
        model = model.to(device)
        # 损失函数、优化器、调度器与你的完全一致
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # 记录训练配置
        logger.info("=" * 60)
        logger.info(f"{model_name} 训练配置（与主模型一致）")
        logger.info(f"优化器：AdamW（lr=1e-4，weight_decay=5e-4）")
        logger.info(f"调度器：StepLR（step_size=10，gamma=0.1）")
        logger.info(f"损失函数：CrossEntropyLoss（带类别权重）")
        logger.info(f"训练epoch：50，早停耐心值：10")
        logger.info("=" * 60)

        # 启动训练
        best_acc = train_model(
            model=model,
            model_name=model_name,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=50,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )

        # 保存结果
        ablation_results[model_name] = best_acc

    # 打印消融实验汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("===== 消融实验汇总结果 =====")
    for model_name, best_acc in ablation_results.items():
        logger.info(f"{model_name}：最佳验证准确率 = {best_acc:.2f}%")
    logger.info("=" * 60)