import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
os.environ['PYTHONHASHSEED'] = '2025'
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ===================== 日志配置 =====================
def setup_logger():
    """设置日志保存功能"""
    log_dir = "test_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logger()


# ===================== 数据集类 =====================
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


# ===================== 模型相关模块 =====================
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


# ===================== 工具函数 =====================
def load_test_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if not isinstance(data.get('images', []), list):
        raise ValueError("data['images'] should be a list")
    images = data['images']
    if len(images) == 0:
        raise ValueError("测试数据集为空，请检查 JSON 文件")
    return images


def calculate_eer(fpr, tpr):
    fnr = 1 - tpr
    min_dist_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[min_dist_idx] + fnr[min_dist_idx]) / 2 * 100
    return eer


# ===================== 主测试流程 =====================
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    try:
        test_data = load_test_data(r'')
        logger.info(f"成功加载测试数据，共 {len(test_data)} 个样本")
    except ValueError as e:
        logger.error(f"加载测试数据失败: {e}")
        exit()

    test_dataset = FaceDataset(test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    logger.info(f"测试集样本总数: {len(test_dataset)}")

    model = ImprovedResNet50(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(''))
        logger.info("成功加载预训练模型权重")
    except FileNotFoundError:
        logger.error("未找到预训练模型权重文件，请先进行训练")
        exit()

    model.eval()
    true_labels = []
    pred_labels = []
    pred_probs = []

    logger.info("开始模型测试...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            pred_probs.extend(probs)

            _, predicted = torch.max(outputs.data, 1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                logger.info(f"已完成 {batch_idx + 1}/{len(test_loader)} 批次测试")

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)

    # 计算核心指标
    overall_accuracy = 100 * np.sum(true_labels == pred_labels) / len(true_labels)
    precision_macro = precision_score(true_labels, pred_labels, average='macro') * 100
    recall_macro = recall_score(true_labels, pred_labels, average='macro') * 100
    f1_macro = f1_score(true_labels, pred_labels, average='macro') * 100

    auc_score = 0.0
    try:
        auc_score = roc_auc_score(true_labels, pred_probs) * 100
    except ValueError as e:
        logger.warning(f"计算AUC失败: {e}")

    eer_score = 0.0
    try:
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        eer_score = calculate_eer(fpr, tpr)
    except Exception as e:
        logger.warning(f"计算EER失败: {e}")

    # 打印最终结果
    logger.info("\n" + "=" * 80)
    logger.info("===== 模型测试核心指标结果 =====")
    logger.info("=" * 80)
    logger.info(f"模型名称: ImprovedResNet50")
    logger.info(f"整体准确率: {overall_accuracy:.2f}%")
    logger.info(f"宏观精确率 (Precision): {precision_macro:.2f}%")
    logger.info(f"宏观召回率 (Recall): {recall_macro:.2f}%")
    logger.info(f"宏观F1分数 (F1-Score): {f1_macro:.2f}%")
    logger.info(f"AUC: {auc_score:.2f}%")
    logger.info(f"EER: {eer_score:.2f}%")
    logger.info("=" * 80)

    # 保存核心结果到JSON文件
    results_dir = "test_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    core_results = {
        "模型名称": "ImprovedResNet50",
        "测试时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "测试样本数": len(test_dataset),
        "整体准确率(%)": round(overall_accuracy, 2),
        "宏观精确率(%)": round(precision_macro, 2),
        "宏观召回率(%)": round(recall_macro, 2),
        "宏观F1分数(%)": round(f1_macro, 2),
        "AUC(%)": round(auc_score, 2),
        "EER(%)": round(eer_score, 2)
    }

    results_filename = os.path.join(results_dir, f"test_core_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(core_results, f, ensure_ascii=False, indent=4)

    logger.info(f"核心测试结果已保存到: {results_filename}")