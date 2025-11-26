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
# 配置日志：同时输出到终端和文件，包含时间戳、日志级别
logging.basicConfig(
    level=logging.INFO,  # 日志级别：只记录INFO及以上信息
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式：时间-级别-内容
    datefmt='%Y-%m-%d %H:%M:%S',  # 时间戳格式（精确到秒，便于复现追溯）
    handlers=[
        logging.FileHandler('train_improved_resnet50.log', encoding='utf-8'),  # 写入日志文件（永久留存）
        logging.StreamHandler(sys.stdout)  # 同时在终端显示（实时查看）
    ]
)
logger = logging.getLogger(__name__)  # 创建日志实例

# -------------------------- 固定所有随机种子--------------------------
os.environ['PYTHONHASHSEED'] = '2025'
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 定义数据集类
class FaceDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

        # 验证数据
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


# 定义注意力模块
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return identity * out


# 定义特征金字塔网络（FPN）
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


# 定义改进的ResNet50模型
class ImprovedResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ImprovedResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.attention1 = AttentionModule(256)
        self.attention2 = AttentionModule(512)
        self.attention3 = AttentionModule(1024)
        self.attention4 = AttentionModule(2048)
        self.fpn = FPN([256, 512, 1024, 2048], 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * 4, num_classes)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        c2 = self.resnet50.layer1(x)
        c2 = self.attention1(c2)
        c3 = self.resnet50.layer2(c2)
        c3 = self.attention2(c3)
        c4 = self.resnet50.layer3(c3)
        c4 = self.attention3(c4)
        c5 = self.resnet50.layer4(c4)
        c5 = self.attention4(c5)
        fpn_out = self.fpn([c2, c3, c4, c5])
        out = []
        for feature in fpn_out:
            out.append(self.avgpool(feature).flatten(1))
        out = torch.cat(out, dim=1)
        out = self.fc(out)
        return out


# 数据预处理（保持原参数不变）
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 读取数据并划分训练集和验证集
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


# 检查是否有可用的 GPU（保持原逻辑不变）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备：{device}")

# 加载数据（保持原逻辑不变）
try:
    train_data, val_data = load_and_split_data('annotation.json')
except ValueError as e:
    logger.error(f"数据加载失败：{e}")
    exit()

# 确保数据集不为空
if len(train_data) == 0 or len(val_data) == 0:
    logger.error("训练集或验证集为空，请检查数据集划分")
    raise ValueError("训练集或验证集为空，请检查数据集划分")

# 计算每个类别的样本数量和权重
class_counts = [0, 0]
for img_info in train_data:
    class_counts[img_info['label']] += 1

total = sum(class_counts)
class_weights = [total / count for count in class_counts]
class_weights = torch.FloatTensor(class_weights).to(device)
logger.info(f"类别分布：类别0样本数 {class_counts[0]}, 类别1样本数 {class_counts[1]}")
logger.info(f"类别权重：类别0权重 {class_weights[0]:.4f}, 类别1权重 {class_weights[1]:.4f}")

# 创建自定义数据集和数据加载器
train_dataset = FaceDataset(train_data, transform=transform)
val_dataset = FaceDataset(val_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
logger.info(f"数据加载器创建完成：训练集批次大小 {32}, 验证集批次大小 {32}")


# 训练模型的函数
def train_model(model, criterion, optimizer, scheduler, num_epochs, model_name):
    best_accuracy = 0.0
    early_stopping_patience = 10
    no_improvement_epochs = 0

    logger.info(f"开始训练 {model_name}，总epoch数：{num_epochs}，早停耐心值：{early_stopping_patience}")

    # 外层epoch进度条（终端显示，记录当前epoch进度）
    epoch_progress = tqdm(range(num_epochs), desc=f"Overall Training Progress ({model_name})", leave=True)

    for epoch in epoch_progress:
        model.train()
        running_loss = 0.0
        # 内层batch进度条（终端显示，记录当前epoch的batch进度+实时损失）
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training', leave=False)

        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            # 实时更新内层进度条的损失显示（终端可见）
            train_progress.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        # 关闭当前epoch的batch进度条
        train_progress.close()

        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f'Epoch {epoch + 1}/{num_epochs} - 训练损失：{epoch_loss:.4f}')

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation', leave=False)
        with torch.no_grad():
            for images, labels in val_progress:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # 验证进度条显示当前验证样本数
                val_progress.set_postfix({"Validated Samples": total})

        # 关闭验证进度条
        val_progress.close()

        accuracy = 100 * correct / total
        logger.info(f'Epoch {epoch + 1}/{num_epochs} - 验证准确率：{accuracy:.2f}%')

        # 更新外层epoch进度条的显示（当前最佳准确率）
        epoch_progress.set_postfix({"Best Acc": f"{best_accuracy:.2f}%", "Current Acc": f"{accuracy:.2f}%"})

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = f'{model_name}_20251118_9.pth'
            torch.save(model.state_dict(), model_path)
            logger.info(f'Epoch {epoch + 1}/{num_epochs} - 最佳模型更新：准确率 {accuracy:.2f}%，保存路径：{model_path}')
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            logger.info(f'Epoch {epoch + 1}/{num_epochs} - 准确率未提升，已连续 {no_improvement_epochs} 个epoch')

        # 早停机制
        if no_improvement_epochs >= early_stopping_patience:
            logger.info(
                f'早停触发：{model_name} 在第 {epoch + 1} 个epoch停止训练（{early_stopping_patience} 个epoch准确率未提升）')
            # 关闭外层进度条
            epoch_progress.close()
            break

        # 更新学习率
        scheduler.step()
        logger.info(f'Epoch {epoch + 1}/{num_epochs} - 学习率更新：{scheduler.get_last_lr()[0]:.6f}')

    # 关闭外层进度条
    epoch_progress.close()
    logger.info(f"{model_name} 训练结束，最佳验证准确率：{best_accuracy:.2f}%")
    return best_accuracy


improved_model = ImprovedResNet50(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(improved_model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 记录训练配置（保存超参数）
logger.info("=" * 50)
logger.info("训练配置汇总（论文复现用）")
logger.info(f"模型：ImprovedResNet50（含注意力模块+FPN）")
logger.info(f"优化器：AdamW，初始学习率：1e-4，权重衰减：5e-4")
logger.info(f"学习率调度器：StepLR，步长：10，衰减系数：0.1")
logger.info(f"损失函数：CrossEntropyLoss（带类别权重）")
logger.info(f"训练epoch数：50，早停耐心值：10")
logger.info(f"批次大小：32，输入图片尺寸：224x224")
logger.info("=" * 50)

# 启动训练
improved_best_accuracy = train_model(improved_model, criterion, optimizer, scheduler, 50, 'improved_model')

# 最终结果日志（突出显示）
logger.info("=" * 50)
logger.info(f"最终训练结果：Improved model 最佳验证准确率 = {improved_best_accuracy:.2f}%")
logger.info("=" * 50)