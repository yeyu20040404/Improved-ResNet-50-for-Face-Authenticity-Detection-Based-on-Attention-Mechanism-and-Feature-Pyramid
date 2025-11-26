import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
from scipy.stats import chi2
import logging
import cv2
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from torch.autograd import Variable

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    log_dir = "mcnemar_test_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"mcnemar_test_final_model_cam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
        img_path = img_info['file_path']
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


# -------------------------- 模型定义（修改最终模型，支持提取FPN特征）--------------------------
# 1. 基准模型：ResNet50 Baseline
class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50Baseline, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)


# 2. 注意力模块
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


# 3. FPN模块
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


# 4. 最终模型（修改：支持返回FPN最后一个特征图，不影响原有推理）
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

    def forward(self, x, return_fpn_feature=False):  # 新增可选参数，默认不返回特征
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

        fpn_out = self.fpn([c2, c3, c4, c5])  # FPN输出：4个特征图（通道数256）
        last_fpn_feature = fpn_out[-1]  # 取FPN最后一个特征图

        out = []
        for feature in fpn_out:
            out.append(self.avgpool(feature).flatten(1))
        out = torch.cat(out, dim=1)
        logits = self.fc(out)

        if return_fpn_feature:
            return logits, last_fpn_feature
        return logits


# -------------------------- CAM热图生成工具函数--------------------------
def get_cam_feature(model, img_tensor, device):
    """获取FPN最后一个特征图和全连接层权重"""
    model.eval()
    with torch.no_grad():
        output, last_fpn_feature = model(img_tensor.to(device), return_fpn_feature=True)
    fc_weights = model.fc.weight
    return last_fpn_feature, fc_weights


def generate_cam(last_fpn_feature, fc_weights, target_class):
    """生成CAM热图（适配FPN结构：通道数256）"""
    # last_fpn_feature: (1, 256, H, W) → FPN最后一个特征图
    # fc_weights: (2, 1024) → 提取对应target_class的权重，取最后256维（对应FPN最后一个特征图）
    batch_size, channels, h, w = last_fpn_feature.shape

    # 提取目标类别的权重，取最后256维（匹配FPN最后一个特征图的通道数）
    target_weights = fc_weights[target_class, -256:]  # (256,)

    # 计算CAM：权重 × 特征图 → 求和降维（1, 256, H, W）→ (1, H, W)
    cam = torch.sum(target_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) * last_fpn_feature, dim=1)
    #cam = cam.squeeze(0).cpu().numpy()  # (H, W)
    cam = cam.squeeze(0).detach().cpu().numpy()  # 分离梯度后转NumPy

    # 归一化（0-1），避免数值溢出
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def visualize_cam(img_path, cam, save_path, label, pred_label):
    """可视化：原始图片 + CAM热图叠加（修复尺寸不匹配问题）"""
    # 加载原始图片（224×224）
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_np = np.array(img)  # (224, 224, 3)

    # 关键修复：将 7×7 的 CAM 缩放到 224×224（与原始图片尺寸一致）
    cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)  # (224, 224)

    # 生成热图（基于缩放后的 CAM）
    norm = Normalize(vmin=0, vmax=1)
    heatmap = cm.jet(norm(cam_resized))[:, :, :3]  # (224, 224, 3) → 现在尺寸匹配

    # 叠加热图（透明度0.5）
    overlay = (img_np / 255.0) * 0.5 + heatmap * 0.5  # 两者都是 (224,224,3)，可正常叠加
    overlay = np.clip(overlay, 0, 1)

    # 绘制图像
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_np)
    ax[0].set_title(
        f'原始图片\n真实标签：{"真实人脸(0)" if label == 0 else "AI人脸(1)"}\n预测标签：{"真实人脸(0)" if pred_label == 0 else "AI人脸(1)"}',
        fontsize=10)
    ax[0].axis('off')

    ax[1].imshow(overlay)
    ax[1].set_title('CAM注意力热图\n（红色区域为模型聚焦区域）', fontsize=10)
    ax[1].axis('off')

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def sample_and_visualize_cam(model, test_loader, device, save_dir, sample_num=10):
    """随机选择10张真实人脸+10张AI人脸，生成CAM热图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 收集样本：10真+10假
    real_samples = []  # 真实人脸（0类）
    fake_samples = []  # AI人脸（1类）

    model.eval()
    with torch.no_grad():
        for img_tensor, label, img_path in test_loader:
            # 获取预测标签
            outputs = model(img_tensor.to(device))
            probs = torch.softmax(outputs, dim=1)
            pred_labels = (probs[:, 1] >= 0.5).cpu().numpy().astype(int)

            # 收集样本
            for i in range(len(label)):
                if label[i] == 0 and len(real_samples) < sample_num:
                    real_samples.append((img_tensor[i:i + 1], label[i].item(), img_path[i], pred_labels[i]))
                elif label[i] == 1 and len(fake_samples) < sample_num:
                    fake_samples.append((img_tensor[i:i + 1], label[i].item(), img_path[i], pred_labels[i]))

                # 收集够20个样本则停止
                if len(real_samples) == sample_num and len(fake_samples) == sample_num:
                    break
            if len(real_samples) == sample_num and len(fake_samples) == sample_num:
                break

    logger.info(f"\n成功收集样本：{len(real_samples)}张真实人脸 + {len(fake_samples)}张AI人脸")

    # 生成并保存CAM热图
    for idx, (img_tensor, label, img_path, pred_label) in enumerate(real_samples + fake_samples):
        # 获取CAM特征和热图（调用修改后的函数）
        last_fpn_feature, fc_weights = get_cam_feature(model, img_tensor, device)
        cam = generate_cam(last_fpn_feature, fc_weights, target_class=label)

        # 保存路径
        img_type = "真实人脸" if label == 0 else "AI人脸"
        save_path = os.path.join(save_dir, f"{img_type}_样本{idx + 1}_标签{label}_预测{pred_label}.png")

        # 可视化
        visualize_cam(img_path, cam, save_path, label, pred_label)
        logger.info(f"CAM热图已保存：{save_path}")

    # 生成汇总图（2行10列）
    fig, axes = plt.subplots(2, 10, figsize=(30, 6))
    axes = axes.flatten()

    for idx, (img_tensor, label, img_path, pred_label) in enumerate(real_samples + fake_samples):
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        axes[idx].imshow(img)
        axes[idx].set_title(f'{"真实" if label == 0 else "AI"}_样本{idx + 1}\n真{label}_预{pred_label}', fontsize=8)
        axes[idx].axis('off')

    plt.suptitle('CAM可视化汇总（上：真实人脸，下：AI人脸）', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "CAM汇总图.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"CAM汇总图已保存：{os.path.join(save_dir, 'CAM汇总图.png')}")


# --------------------------McNemar检验--------------------------
def load_test_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if not isinstance(data.get('images', []), list):
        raise ValueError("data['images'] should be a list")
    images = data['images']
    if len(images) == 0:
        raise ValueError("测试数据集为空")
    true_labels = [int(img_info['label']) for img_info in images]
    return images, true_labels


def get_model_predictions(model, test_loader, device, threshold=0.4):
    model.eval()
    pred_labels = []
    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            ai_probs = probs[:, 1].cpu().numpy()
            preds = (ai_probs >= threshold).astype(int)
            pred_labels.extend(preds)
    return np.array(pred_labels)


def mcnemar_test(true_labels, pred_baseline, pred_final, alpha=0.05):
    assert len(true_labels) == len(pred_baseline) == len(pred_final),"标签长度不一致"
    a = np.sum((pred_baseline == true_labels) & (pred_final == true_labels))
    b = np.sum((pred_baseline == true_labels) & (pred_final != true_labels))
    c = np.sum((pred_baseline != true_labels) & (pred_final == true_labels))
    d = np.sum((pred_baseline != true_labels) & (pred_final != true_labels))
    contingency_table = (a, b, c, d)

    if b + c >= 25:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2_stat = (abs(b - c)) ** 2 / (b + c)

    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    significant = p_value < alpha
    return chi2_stat, p_value, significant, contingency_table


# -------------------------- 主流程（McNemar检验 + CAM可视化）--------------------------
if __name__ == "__main__":
    # 配置参数
    test_json_path = r''
    threshold = 0.5
    alpha = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam_save_dir = "CAM_visualization"  # CAM热图保存目录

    logger.info("=" * 80)
    logger.info("===== McNemar检验 + CAM热图可视化 =====")
    logger.info("=" * 80)
    logger.info(f"使用设备：{device}")
    logger.info(f"AI人脸阈值：{threshold}")
    logger.info(f"显著性水平：α={alpha}")
    logger.info(f"CAM可视化：随机选择10张真实人脸 + 10张AI人脸")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载测试数据和真实标签
    try:
        test_images, true_labels = load_test_data(test_json_path)
        true_labels = np.array(true_labels)
        logger.info(f"成功加载测试数据：{len(test_images)} 个样本")
    except ValueError as e:
        logger.error(f"加载测试数据失败：{e}")
        exit()

    # 创建测试数据加载器
    test_dataset = FaceDataset(test_images, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    logger.info(f"测试数据加载器创建完成（批次大小：32）")

    # -------------------------- 1. McNemar检验 --------------------------
    # 加载基准模型
    baseline_model = ResNet50Baseline(num_classes=2).to(device)
    baseline_weight_path = "ResNet50_Baseline_20251119_ablation.pth"
    try:
        baseline_model.load_state_dict(torch.load(baseline_weight_path))
        pred_baseline = get_model_predictions(baseline_model, test_loader, device, threshold)
        acc_baseline = np.sum(pred_baseline == true_labels) / len(true_labels) * 100
        logger.info(f"\n基准模型（ResNet50 Baseline）：准确率 {acc_baseline:.2f}%")
    except FileNotFoundError:
        logger.error(f"未找到基准模型权重：{baseline_weight_path}")
        exit()

    # 加载最终模型（用于检验和CAM可视化）
    final_model = ImprovedResNet50(num_classes=2).to(device)
    final_weight_path = "improved_model_random.pth"
    try:
        final_model.load_state_dict(torch.load(final_weight_path))
        pred_final = get_model_predictions(final_model, test_loader, device, threshold)
        acc_final = np.sum(pred_final == true_labels) / len(true_labels) * 100
        logger.info(f"最终模型（ResNet50+注意力+FPN）：准确率 {acc_final:.2f}%")
    except FileNotFoundError:
        logger.error(f"未找到最终模型权重：{final_weight_path}")
        exit()

    # 执行McNemar检验
    chi2_stat, p_value, significant, contingency_table = mcnemar_test(
        true_labels, pred_baseline, pred_final, alpha=alpha
    )
    a, b, c, d = contingency_table
    logger.info(f"\n【McNemar检验结果】")
    logger.info(f"列联表：a={a}, b={b}, c={c}, d={d}")
    logger.info(f"卡方统计量：{chi2_stat:.4f}，p值：{p_value:.4f}")
    logger.info(f"结论：{'显著差异' if significant else '无显著差异'}（α={alpha}）")

    # -------------------------- 2. CAM热图可视化 --------------------------
    logger.info(f"\n开始生成CAM热图...")
    sample_and_visualize_cam(
        model=final_model,
        test_loader=test_loader,
        device=device,
        save_dir=cam_save_dir,
        sample_num=10
    )
    logger.info(f"CAM热图生成完成！保存路径：{cam_save_dir}")

    # 保存检验结果
    result_dict = {
        "检验时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "对比模型": "ResNet50 Baseline vs 最终模型（ResNet50+注意力+FPN）",
        "CAM可视化": f"10张真实人脸+10张AI人脸，保存于{cam_save_dir}",
        "基准模型准确率(%)": round(acc_baseline, 2),
        "最终模型准确率(%)": round(acc_final, 2),
        "卡方统计量": round(chi2_stat, 4),
        "p值": round(p_value, 4),
        "是否显著差异": bool(significant)
    }
    results_dir = "mcnemar_test_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    json_path = os.path.join(results_dir, f"mcnemar_cam_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    logger.info(f"\n检验+CAM结果已保存到：{json_path}")

    logger.info("\n" + "=" * 80)
    logger.info("McNemar检验 + CAM可视化完成！")
    logger.info("=" * 80)