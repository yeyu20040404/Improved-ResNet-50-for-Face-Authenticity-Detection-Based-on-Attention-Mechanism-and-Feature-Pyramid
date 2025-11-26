import os
import json

# 指定真实人脸和AI人脸的文件夹路径
real_faces_dir = r''
ai_faces_dir = r''

# 初始化标注数据结构
annotation = {"images": []}

# 遍历真实人脸文件夹
for filename in os.listdir(real_faces_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        file_path = os.path.join(real_faces_dir, filename)
        label = 0  # 真实人脸的标签
        annotation["images"].append({"file_path": file_path, "label": label})

# 遍历AI人脸文件夹
for filename in os.listdir(ai_faces_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        file_path = os.path.join(ai_faces_dir, filename)
        label = 1  # AI人脸的标签
        annotation["images"].append({"file_path": file_path, "label": label})

# 将标注数据转换为JSON格式的字符串
json_str = json.dumps(annotation, indent=4)

# 将JSON字符串保存到文件
with open('annotation_test.json', 'w') as f:
    f.write(json_str)

print("Annotation file 'annotation_test.json' has been created.")