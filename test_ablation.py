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
import pandas as pd

warnings.filterwarnings('ignore')

# -------------------------- 固定随机种子--------------------------
os.environ['PYTHONHASHSEED'] = '2025'
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------- 日志配置 --------------------------
def setup_logger():
    log_dir = "ablation_test_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"ablation_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

# -------------------------- 所有模型定义--------------------------
class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50Baseline, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)

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

# -------------------------- 工具函数 --------------------------
def load_test_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if not isinstance(data.get('images', []), list):
        raise ValueError("data['images'] should be a list")
    images = data['images']
    if len(images) == 0:
        raise ValueError("测试数据集为空")
    return images

def calculate_eer(fpr, tpr):
    fnr = 1 - tpr
    min_dist_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[min_dist_idx] + fnr[min_dist_idx]) / 2 * 100
    return eer

# -------------------------- 单个模型测试函数--------------------------
def test_single_model(model, model_name, test_loader, device, threshold=0.5):
    logger.info(f"\n===== 开始测试 {model_name}（AI人脸阈值={threshold}）=====")
    model.eval()
    true_labels = []
    pred_labels = []
    pred_probs = []

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # 计算AI人脸（1类）概率
            probs = torch.softmax(outputs, dim=1)
            ai_probs = probs[:, 1].cpu().numpy()
            pred_probs.extend(ai_probs)

            # 按阈值判定预测标签
            predicted = (ai_probs >= threshold).astype(int)
            predicted = torch.tensor(predicted).to(device)

            # 仅统计整体准确率
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # 保存标签用于后续指标计算
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                logger.info(f"[{model_name}] 已完成 {batch_idx+1}/{len(test_loader)} 批次")

    # 转换为numpy数组
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)

    # 计算核心指标
    # 1. 整体准确率
    overall_acc = 100 * total_correct / total_samples

    # 2. 宏观Precision/Recall/F1（综合两类表现）
    precision_macro = precision_score(true_labels, pred_labels, average='macro') * 100
    recall_macro = recall_score(true_labels, pred_labels, average='macro') * 100
    f1_macro = f1_score(true_labels, pred_labels, average='macro') * 100

    # 3. AUC/EER
    try:
        auc = roc_auc_score(true_labels, pred_probs) * 100
    except ValueError:
        auc = 0.0
        logger.warning(f"[{model_name}] 计算AUC失败")
    try:
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        eer = calculate_eer(fpr, tpr)
    except Exception:
        eer = 0.0
        logger.warning(f"[{model_name}] 计算EER失败")

    # 打印单模型核心结果
    logger.info(f"\n[{model_name}] 核心测试结果：")
    logger.info(f"  整体准确率：{overall_acc:.2f}%")
    logger.info(f"  宏观Precision：{precision_macro:.2f}%")
    logger.info(f"  宏观Recall：{recall_macro:.2f}%")
    logger.info(f"  宏观F1：{f1_macro:.2f}%")
    logger.info(f"  AUC：{auc:.2f}%")
    logger.info(f"  EER：{eer:.2f}%")

    # 返回核心指标结果
    return {
        "模型名称": model_name,
        "整体准确率(%)": round(overall_acc, 2),
        "宏观Precision(%)": round(precision_macro, 2),
        "宏观Recall(%)": round(recall_macro, 2),
        "宏观F1(%)": round(f1_macro, 2),
        "AUC(%)": round(auc, 2),
        "EER(%)": round(eer, 2),
        "测试样本数": total_samples
    }

# -------------------------- 主测试流程 --------------------------
if __name__ == "__main__":
    # 配置
    test_json_path = r''  # 你的测试集JSON路径
    threshold = 0.5  # AI人脸判定阈值
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备：{device}")
    logger.info(f"AI人脸判定阈值：{threshold}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载测试数据
    try:
        test_data = load_test_data(test_json_path)
        logger.info(f"成功加载测试数据，共 {len(test_data)} 个样本")
    except ValueError as e:
        logger.error(f"加载测试数据失败：{e}")
        exit()

    # 创建测试数据加载器
    test_dataset = FaceDataset(test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    logger.info(f"测试数据加载器创建完成，批次大小：32")

    # 定义要测试的所有模型（消融实验+最终模型）
    models_to_test = [
        (
            "ResNet50_Baseline",
            ResNet50Baseline(num_classes=2),
            "ResNet50_Baseline_ablation.pth"
        ),
        (
            "ResNet50_FPN",
            ResNet50FPN(num_classes=2),
            "ResNet50_FPN_ablation.pth"
        ),
        (
            "ResNet50_CAM",
            ResNet50CAM(num_classes=2),
            "ResNet50_CAM_ablation.pth"
        ),
        (
            "最终模型(ResNet50+注意力+FPN)",
            ImprovedResNet50(num_classes=2),
            "improved_model_1.pth"
        )
    ]

    # 批量测试所有模型
    all_results = []
    for model_name, model, weight_path in models_to_test:
        # 加载模型权重
        model = model.to(device)
        try:
            model.load_state_dict(torch.load(weight_path))
            logger.info(f"\n成功加载 {model_name} 的权重：{weight_path}")
        except FileNotFoundError:
            logger.error(f"未找到 {model_name} 的权重文件：{weight_path}")
            continue

        # 测试模型
        result = test_single_model(model, model_name, test_loader, device, threshold)
        all_results.append(result)

    # -------------------------- 生成汇总结果 --------------------------
    if all_results:
        # 1. 生成DataFrame对比表格
        df_results = pd.DataFrame(all_results)
        # 调整列顺序
        col_order = [
            "模型名称", "整体准确率(%)", "宏观Precision(%)",
            "宏观Recall(%)", "宏观F1(%)", "AUC(%)", "EER(%)"
        ]
        df_results = df_results[col_order]

        # 2. 保存为Excel表格
        results_dir = "ablation_test_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        excel_path = os.path.join(results_dir, f"消融实验核心指标对比表格_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df_results.to_excel(excel_path, index=False, engine='openpyxl')
        logger.info(f"\n消融实验核心指标对比表格已保存：{excel_path}")

        # 3. 保存为JSON文件
        json_path = os.path.join(results_dir, f"消融实验核心结果汇总_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        logger.info(f"消融实验核心结果JSON已保存：{json_path}")

        # 4. 打印汇总表格（终端可视化）
        logger.info("\n" + "=" * 120)
        logger.info("===== 消融实验核心指标批量测试汇总结果 =====")
        logger.info(df_results.to_string(index=False))
        logger.info("=" * 120)
    else:
        logger.error("没有成功测试任何模型，请检查权重文件路径")

    logger.info("\n测试完成！")