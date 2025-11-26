import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import logging
from datetime import datetime
import warnings
import pandas as pd

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
    log_dir = "test_logs_baselines"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"test_baseline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

# ===================== 基准模型定义 =====================
class VGG16Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vgg16 = models.vgg16(weights=None)
        self.vgg16.classifier[6] = nn.Linear(self.vgg16.classifier[6].in_features, num_classes)
    def forward(self, x):
        return self.vgg16(x)

class ResNet34Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet34 = models.resnet34(weights=None)
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet34(x)

class ResNet101Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet101 = models.resnet101(weights=None)
        self.resnet101.fc = nn.Linear(self.resnet101.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet101(x)

class DenseNet169Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.densenet169 = models.densenet169(weights=None)
        self.densenet169.classifier = nn.Linear(self.densenet169.classifier.in_features, num_classes)
    def forward(self, x):
        return self.densenet169(x)

class EfficientNetBaseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
    def forward(self, x):
        return self.efficientnet(x)

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

# ===================== 测试单个基准模型函数 =====================
def test_single_baseline(model_name, model, weight_path, test_loader, device):
    logger.info(f"\n===== 开始测试基准模型：{model_name} =====")
    # 加载模型权重
    model = model.to(device)
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        logger.info(f"成功加载权重：{weight_path}")
    except FileNotFoundError:
        logger.error(f"未找到权重文件：{weight_path}")
        return None

    # 测试模型
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

            # 预测标签
            _, predicted = torch.max(outputs.data, 1)

            # 统计整体准确率
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # 保存标签用于指标计算
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                logger.info(f"[{model_name}] 已完成 {batch_idx + 1}/{len(test_loader)} 批次")

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
    except ValueError as e:
        auc = 0.0
        logger.warning(f"[{model_name}] 计算AUC失败：{e}")
    try:
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        eer = calculate_eer(fpr, tpr)
    except Exception as e:
        eer = 0.0
        logger.warning(f"[{model_name}] 计算EER失败：{e}")

    # 打印核心结果
    logger.info(f"\n[{model_name}] 核心测试结果：")
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

# ===================== 主测试流程=====================
if __name__ == "__main__":
    # 配置参数
    test_json_path = r''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载测试数据
    try:
        test_data = load_test_data(test_json_path)
        test_dataset = FaceDataset(test_data, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        logger.info(f"成功加载测试数据，共 {len(test_dataset)} 个样本")
    except ValueError as e:
        logger.error(f"加载测试数据失败: {e}")
        exit()

    # 定义所有要测试的基准模型（模型名→模型类→权重路径）
    baseline_models = [
        ("VGG16", VGG16Baseline(num_classes=2), "VGG16_best_weights.pth"),
        ("ResNet34", ResNet34Baseline(num_classes=2), "ResNet34_best_weights.pth"),
        ("ResNet101", ResNet101Baseline(num_classes=2), "ResNet101_best_weights.pth"),
        ("DenseNet169", DenseNet169Baseline(num_classes=2), "DenseNet169_best_weights.pth"),
        ("EfficientNet", EfficientNetBaseline(num_classes=2), "EfficientNet_best_weights.pth")
    ]

    # 批量测试所有基准模型
    all_results = []
    for model_name, model, weight_path in baseline_models:
        result = test_single_baseline(model_name, model, weight_path, test_loader, device)
        if result is not None:
            all_results.append(result)

    # 生成汇总结果
    if all_results:
        # 转换为DataFrame并调整列顺序
        results_df = pd.DataFrame(all_results)
        col_order = [
            "模型名称", "整体准确率(%)", "Precision(%)",
            "Recall(%)", "F1(%)", "AUC(%)", "EER(%)"
        ]
        results_df = results_df[col_order].sort_values(by="F1(%)", ascending=False).reset_index(drop=True)

        # 保存结果到Excel
        results_dir = "test_results_baselines"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        excel_path = os.path.join(results_dir, f"基准模型核心指标对比_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        results_df.to_excel(excel_path, index=False, engine='openpyxl')

        # 保存结果到JSON
        json_path = os.path.join(results_dir, f"基准模型核心结果汇总_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        # 打印汇总表格
        logger.info("\n" + "=" * 120)
        logger.info("===== 所有基准模型核心指标汇总 =====")
        logger.info(results_df.to_string(index=False))
        logger.info("=" * 120)
        logger.info(f"对比表格已保存：{excel_path}")
        logger.info(f"结果JSON已保存：{json_path}")
    else:
        logger.error("没有成功测试任何基准模型，请检查权重文件路径！")

    logger.info("\n基准模型测试完成！")