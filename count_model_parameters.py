import torch
import torch.nn as nn
from torchvision import models, transforms
import time
from thop import profile
import numpy as np
import logging
from datetime import datetime
import os
import pandas as pd


# -------------------------- 日志配置--------------------------
def setup_logger():
    log_dir = "model_perf_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"model_perf_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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


# -------------------------- 模型定义--------------------------
# 1. 基准模型：ResNet50 Baseline
class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50Baseline, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)


# 2. FPN模块
# 替换原有的FPN类为以下版本（自适应上采样，解决尺寸匹配问题）
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, inputs):
        # inputs：[c2, c3, c4, c5]（从浅到深的特征图）
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # 从最深层开始，向上融合（关键：用interpolate指定目标尺寸，替代固定2倍上采样）
        laterals.reverse()  # 变成 [c5_lat, c4_lat, c3_lat, c2_lat]
        for i in range(len(laterals) - 1):
            # 目标尺寸：下一层特征图的尺寸（h, w），动态匹配而非固定2倍
            target_size = laterals[i + 1].shape[2:]  # (h, w)
            # 自适应上采样：直接将当前层缩放到目标尺寸
            upsampled = nn.functional.interpolate(
                laterals[i],
                size=target_size,  # 精准匹配目标尺寸（解决135≠136的问题）
                mode='nearest'
            )
            # 特征融合（尺寸已匹配，无冲突）
            laterals[i + 1] += upsampled

        laterals.reverse()  # 恢复为 [c2_lat, c3_lat, c4_lat, c5_lat]
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return outs


# 3. ResNet50+FPN
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


# 4. CAM模块
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


# 5. ResNet50+CAM
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


# 6. 最终模型（ResNet50+注意力+FPN）
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


# -------------------------- 测量工具函数--------------------------
def measure_model_performance(model, model_name, device, input_sizes):
    model.eval()
    results = {"模型名称": model_name}

    # 1. 计算参数量和FLOPs
    dummy_input = torch.randn(1, 3, input_sizes[0][0], input_sizes[0][1]).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    results["参数量（M）"] = round(params / 1e6, 2)
    results["FLOPs（G）"] = round(flops / 1e9, 2)

    # 2. 测量推理延时
    for h, w in input_sizes:
        input_tensor = torch.randn(1, 3, h, w).to(device)

        # GPU预热
        with torch.no_grad():
            for _ in range(50):
                _ = model(input_tensor)

        # 多次测量取平均
        total_time = 0.0
        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(input_tensor)
                torch.cuda.synchronize()
                end = time.time()
                total_time += (end - start)

        avg_latency = round((total_time / num_runs) * 1000, 2)
        results[f"推理延时（{h}×{w}，ms/帧）"] = avg_latency

    return results


# -------------------------- 主测量流程--------------------------
if __name__ == "__main__":
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_sizes = [(224, 224), (1080, 1920)]  # 224×224 + 1080p
    logger.info(f"使用设备：{device}")
    logger.info(f"测量尺寸：{[f'{h}×{w}' for h, w in input_sizes]}")

    # 模型-权重映射
    models_config = [
        (
            "ResNet50 Baseline",
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
            "improved_model.pth"
        )
    ]

    # 加载模型并测量
    all_results = []
    for model_name, model, weight_path in models_config:
        logger.info("\n" + "=" * 80)
        logger.info(f"开始测量：{model_name}")
        logger.info(f"权重路径：{weight_path}")

        # 加载模型权重
        model = model.to(device)
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
            logger.info(f"成功加载 {model_name} 的权重")
        except FileNotFoundError:
            logger.error(f"未找到权重文件：{weight_path}，跳过该模型")
            continue

        # 测量性能
        perf_result = measure_model_performance(model, model_name, device, input_sizes)
        all_results.append(perf_result)

        # 打印单模型结果
        for key, value in perf_result.items():
            logger.info(f"{key}: {value}")

    # -------------------------- 生成汇总结果--------------------------
    if all_results:
        # 1. 生成DataFrame表格
        df_results = pd.DataFrame(all_results)
        # 调整列顺序
        col_order = [
            "模型名称", "参数量（M）", "FLOPs（G）",
            "推理延时（224×224，ms/帧）", "推理延时（1080×1920，ms/帧）"
        ]
        df_results = df_results[col_order]

        # 2. 保存为Excel
        results_dir = "model_perf_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        excel_path = os.path.join(results_dir, f"模型性能对比表格_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df_results.to_excel(excel_path, index=False, engine='openpyxl')
        logger.info(f"\n模型性能对比表格已保存：{excel_path}")

        # 3. 打印终端汇总
        logger.info("\n" + "=" * 100)
        logger.info("===== 模型性能测量汇总 =====")
        logger.info(df_results.to_string(index=False))
        logger.info("=" * 100)
    else:
        logger.error("没有成功测量任何模型，请检查权重路径")

    logger.info("\n测量完成！")