import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pandas as pd
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
    log_dir = "compare_test_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

# -------------------------- 所有模型定义--------------------------
# 1. 对比模型1：Xception-MesoNet
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

# 2. 对比模型2：Face-X-ray
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

# 3. 最终模型（ResNet50+注意力+FPN）
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

class YourFinalModel(nn.Module):
    def __init__(self, num_classes=2):
        super(YourFinalModel, self).__init__()
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

# -------------------------- 数据预处理与加载 --------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_test_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if not isinstance(data.get('images', []), list):
        raise ValueError("data['images'] should be a list")
    images = data['images']
    if len(images) == 0:
        raise ValueError("测试数据集为空")
    return images

# -------------------------- 测试函数--------------------------
def test_model(model, model_name, test_loader, device, threshold=0.5):
    logger.info(f"\n===== 开始测试 {model_name} =====")
    model.eval()
    true_labels = []
    pred_labels = []
    pred_probs = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        test_progress = tqdm(test_loader, desc=f'[{model_name}] 测试进度', leave=False)
        for images, labels in test_progress:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            ai_probs = probs[:, 1].cpu().numpy()
            preds = (ai_probs >= threshold).astype(int)

            # 统计结果
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds)
            pred_probs.extend(ai_probs)
            total_samples += labels.size(0)
            total_correct += (torch.tensor(preds).to(device) == labels).sum().item()
            test_progress.set_postfix({"测试样本数": total_samples})
        test_progress.close()

    # 转换为numpy数组
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)

    # 计算核心指标
    overall_acc = 100 * total_correct / total_samples
    precision = precision_score(true_labels, pred_labels, average='macro') * 100
    recall = recall_score(true_labels, pred_labels, average='macro') * 100
    f1 = f1_score(true_labels, pred_labels, average='macro') * 100

    # AUC和EER
    try:
        auc = roc_auc_score(true_labels, pred_probs) * 100
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        eer = (fpr[np.argmin(np.abs(tpr - (1 - fpr)))] + (1 - tpr[np.argmin(np.abs(tpr - (1 - fpr)))])) / 2 * 100
    except:
        auc = 0.0
        eer = 0.0
        logger.warning(f"[{model_name}] AUC/EER计算失败")

    # 打印核心结果
    logger.info(f"[{model_name}] 核心测试结果：")
    logger.info(f"  整体准确率：{overall_acc:.2f}%")
    logger.info(f"  Precision：{precision:.2f}%")
    logger.info(f"  Recall：{recall:.2f}%")
    logger.info(f"  F1：{f1:.2f}%")
    logger.info(f"  AUC：{auc:.2f}%")
    logger.info(f"  EER：{eer:.2f}%")

    return {
        "模型名称": model_name,
        "整体准确率(%)": round(overall_acc, 2),
        "Precision(%)": round(precision, 2),
        "Recall(%)": round(recall, 2),
        "F1(%)": round(f1, 2),
        "AUC(%)": round(auc, 2),
        "EER(%)": round(eer, 2)
    }

# -------------------------- 主测试对比流程 --------------------------
if __name__ == "__main__":
    # 配置参数
    test_json_path = r''
    threshold = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    your_model_path = ""  # 替换为实际路径

    logger.info("=" * 80)
    logger.info("===== 模型测试对比实验 =====")
    logger.info("=" * 80)
    logger.info(f"设备：{device}")
    logger.info(f"测试阈值：{threshold}")
    logger.info(f"对比模型：最终模型、Xception-MesoNet、Face-X-ray")

    # 加载测试数据
    try:
        test_data = load_test_data(test_json_path)
        test_dataset = FaceDataset(test_data, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        logger.info(f"测试集加载完成：{len(test_dataset)}个样本")
    except Exception as e:
        logger.error(f"加载测试数据失败：{e}")
        exit()

    # 定义要测试的模型
    models_to_test = [
        ("最终模型(ResNet50+注意力+FPN)", YourFinalModel(num_classes=2), your_model_path),
        ("Xception-MesoNet", XceptionMesoNet(num_classes=2), "Xception-MesoNet_best_model.pth"),
        ("Face-X-ray", FaceXray(num_classes=2), "Face-X-ray_best_model.pth")
    ]

    # 批量测试所有模型
    all_test_results = []
    for model_name, model, model_path in models_to_test:
        model = model.to(device)
        try:
            model.load_state_dict(torch.load(model_path))
            logger.info(f"\n加载模型权重成功：{model_name} -> {model_path}")
        except FileNotFoundError:
            logger.error(f"未找到模型权重：{model_path}")
            continue

        # 测试并记录结果
        test_result = test_model(model, model_name, test_loader, device, threshold)
        all_test_results.append(test_result)

    # 生成对比表格并保存
    if all_test_results:
        results_df = pd.DataFrame(all_test_results)
        col_order = [
            "模型名称", "整体准确率(%)", "Precision(%)",
            "Recall(%)", "F1(%)", "AUC(%)", "EER(%)"
        ]
        results_df = results_df[col_order].sort_values(by="F1(%)", ascending=False).reset_index(drop=True)

        # 保存表格
        results_dir = "deepfake_compare_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        excel_path = os.path.join(results_dir, f"模型对比核心指标表格_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        results_df.to_excel(excel_path, index=False, engine='openpyxl')

        # 打印最终对比结果
        logger.info("\n" + "=" * 120)
        logger.info("===== 模型对比核心指标汇总 =====")
        logger.info(results_df.to_string(index=False))
        logger.info("=" * 120)
        logger.info(f"对比表格已保存：{excel_path}")
    else:
        logger.error("没有成功测试任何模型！")

    logger.info("\n测试对比完成！")