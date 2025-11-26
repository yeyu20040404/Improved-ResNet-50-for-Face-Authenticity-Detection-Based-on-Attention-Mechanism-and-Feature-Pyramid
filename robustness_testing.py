import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import random

# -------------------------- 1. 定义鲁棒性扰动函数--------------------------
class RobustnessTransforms:
    """实现3类扰动：JPEG压缩、高斯模糊、Instagram风格滤波"""
    @staticmethod
    def jpeg_compression(img, quality=50):
        """JPEG压缩：quality=50/70（越低压缩越严重）"""
        img_np = np.array(img)  # PIL→numpy
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img_np[:, :, ::-1], encode_param)  # RGB→BGR
        decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR→RGB
        return Image.fromarray(decimg)

    @staticmethod
    def gaussian_blur(img, sigma=1):
        """高斯模糊：sigma=1/2（越大越模糊）"""
        img_np = np.array(img)
        ksize = int(6 * sigma + 1)  # 核大小=6σ+1（确保模糊效果合理）
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(img_np, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return Image.fromarray(blurred)

    @staticmethod
    def instagram_filter(img, filter_type='warm'):
        """Instagram风格颜色滤波：4种常见风格"""
        img_np = np.array(img).astype(np.float32) / 255.0
        if filter_type == 'warm':  # 暖色调（增加红、黄通道）
            img_np[:, :, 0] *= 1.1  # R
            img_np[:, :, 1] *= 1.05  # G
            img_np[:, :, 2] *= 0.9  # B
        elif filter_type == 'cool':  # 冷色调（增加蓝、绿通道）
            img_np[:, :, 0] *= 0.9  # R
            img_np[:, :, 1] *= 1.05  # G
            img_np[:, :, 2] *= 1.1  # B
        elif filter_type == 'vintage':  # 复古风（降低饱和度+偏黄）
            gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
            img_np = img_np * 0.3 + gray[:, :, np.newaxis] * 0.7
            img_np[:, :, 1] *= 1.05  # 偏黄
        elif filter_type == 'high_saturation':  # 高饱和（增加对比度+饱和度）
            img_np = (img_np - 0.5) * 1.2 + 0.5  # 增加对比度
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            img_np[:, :, 1] *= 1.5  # 增加饱和度
            img_np = cv2.cvtColor(img_np, cv2.COLOR_HSV2RGB)
        # 裁剪到0-1范围
        img_np = np.clip(img_np, 0.0, 1.0)
        return Image.fromarray((img_np * 255).astype(np.uint8))

# -------------------------- 2. 定义测试集Dataset--------------------------
class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: 测试集目录，格式要求：
            data_dir/
                real/  # 真实人脸文件夹
                    img1.jpg, img2.jpg...
                fake/  # AI人脸文件夹
                    img1.jpg, img2.jpg...
        transform: 数据预处理+扰动
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []  # 0=真实人脸，1=AI人脸

        # 加载真实人脸（label=0）
        real_dir = os.path.join(data_dir, 'real')
        for img_name in os.listdir(real_dir):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                self.img_paths.append(os.path.join(real_dir, img_name))
                self.labels.append(0)

        # 加载AI人脸（label=1）
        fake_dir = os.path.join(data_dir, 'fake')
        for img_name in os.listdir(fake_dir):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                self.img_paths.append(os.path.join(fake_dir, img_name))
                self.labels.append(1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')  # 转为RGB

        if self.transform is not None:
            img = self.transform(img)

        return img, label

# -------------------------- 3. 模型加载（本地ResNet50，不在线下载）--------------------------
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

    def forward(self, inputs):
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.reverse()
        for i in range(len(laterals) - 1):
            target_size = laterals[i+1].shape[2:]
            upsampled = nn.functional.interpolate(laterals[i], size=target_size, mode='nearest')
            laterals[i + 1] += upsampled
        laterals.reverse()
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return outs

class ImprovedResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ImprovedResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
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
        logits = self.fc(out)
        return logits

# -------------------------- 4. 测试函数（计算准确率和下降幅度）--------------------------
def test_model_robustness(model, test_loader, device):
    """测试模型在扰动后的准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total * 100
    return accuracy

# -------------------------- 5. 主流程（配置+运行鲁棒性测试）--------------------------
if __name__ == "__main__":
    # 配置参数（根据实际情况修改）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data_dir = r""  # 替换成你的测试集路径（如：E:/face_test_set/）
    model_weight_path = ""  # 替换成你的模型权重路径
    batch_size = 32
    input_size = 224  # 训练时的输入尺寸（必须和训练一致）

    # 1. 加载模型
    model = ImprovedResNet50(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print("模型加载完成！")

    # 2. 定义基础预处理
    base_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 关键修改：统一所有场景的transform，确保都包含base_transform
    robustness_scenarios = [
        ("原始（无扰动）", base_transform),
        ("JPEG压缩（QF=70）", transforms.Compose([RobustnessTransforms.jpeg_compression, base_transform])),
        ("JPEG压缩（QF=50）", transforms.Compose([lambda x: RobustnessTransforms.jpeg_compression(x, quality=50), base_transform])),
        ("高斯模糊（σ=1）", transforms.Compose([lambda x: RobustnessTransforms.gaussian_blur(x, sigma=1), base_transform])),
        ("高斯模糊（σ=2）", transforms.Compose([lambda x: RobustnessTransforms.gaussian_blur(x, sigma=2), base_transform])),
        ("Instagram暖色调", transforms.Compose([lambda x: RobustnessTransforms.instagram_filter(x, 'warm'), base_transform])),
        ("Instagram冷色调", transforms.Compose([lambda x: RobustnessTransforms.instagram_filter(x, 'cool'), base_transform])),
        ("Instagram复古风", transforms.Compose([lambda x: RobustnessTransforms.instagram_filter(x, 'vintage'), base_transform])),
        ("Instagram高饱和", transforms.Compose([lambda x: RobustnessTransforms.instagram_filter(x, 'high_saturation'), base_transform])),
    ]

    # 4. 运行所有场景测试
    results = []
    for scenario_name, transform in robustness_scenarios:
        print(f"\n{'='*50}")
        print(f"测试场景：{scenario_name}")
        dataset = FaceDataset(test_data_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        accuracy = test_model_robustness(model, loader, device)
        results.append((scenario_name, accuracy))
        print(f"场景 {scenario_name} 准确率：{accuracy:.2f}%")

    # 5. 计算准确率下降幅度并生成论文表格
    print(f"\n{'='*80}")
    print("鲁棒性测试汇总")
    print("{'='*80}")
    original_acc = results[0][1]
    print(f"测试场景\t\t准确率(%)\t准确率下降幅度(%)")
    print(f"{'-'*50}")
    for scenario_name, acc in results:
        drop = original_acc - acc
        print(f"{scenario_name:<16}\t{acc:.2f}\t\t{drop:.2f}")

    print(f"\n鲁棒性测试完成！")