import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. 先加载模型（后续会详细讲，这里先简单写）
from your_model_file import ConstrainedFormationNet  # 导入你的模型类（替换为你的实际文件路径）

# 实例化模型（参数要和训练时一致，比如num_graphs、num_robots等）
model = ConstrainedFormationNet(num_graphs=3, num_robots=3)
# 加载训练好的权重
model.load_state_dict(torch.load("/home/lpp/formation_test/data/final_model.pth"))
# 设置为评估模式（关闭BatchNorm/Dropout，不影响推理结果）
model.eval()
# 指定设备（和训练时一致，优先cuda）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2. 准备验证集的一个样本（这里用你验证集中的任意一个样本，替换为你的实际数据）
# 注意：输入features的格式要和训练时完全一致（形状、数据类型、设备）
val_features = ...  # 你的验证集features，形状：[1, num_features]（单个样本，批量大小为1）
val_expert_positions = ...  # 该样本的真实专家位置，形状：[1, 3, 2]（3个机器人，x/y坐标）
val_features = val_features.to(device)

# 3. 执行推理，获取预测结果（关闭梯度计算，节省资源，不影响结果）
with torch.no_grad():
    pred_positions, pred_scores = model(val_features, "imitation")

# 4. 提取最佳编队的预测位置（和你训练时的逻辑一致）
best_graph_idx = torch.argmax(pred_scores, dim=1)  # 找到最佳编队索引
best_pred_position = pred_positions[0, best_graph_idx[0], :, :]  # 提取单个样本的最佳编队位置
best_pred_position = best_pred_position.cpu().numpy()  # 转到cpu，转为numpy数组，方便绘图
val_expert_position = val_expert_positions[0].cpu().numpy()  # 真实位置转为numpy数组

# 设置绘图样式
plt.rcParams["figure.figsize"] = (10, 5)

# 子图1：真实专家编队
plt.subplot(1, 2, 1)
plt.scatter(val_expert_position[:, 0], val_expert_position[:, 1], c='red', label='Expert Robot', s=100)
# 标注机器人索引
for i, (x, y) in enumerate(val_expert_position):
    plt.annotate(f"Robot {i}", (x, y), xytext=(5, 5), textcoords='offset points')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("True Expert Formation")
plt.grid(True, alpha=0.3)
plt.legend()

# 子图2：模型预测编队
plt.subplot(1, 2, 2)
plt.scatter(best_pred_position[:, 0], best_pred_position[:, 1], c='blue', label='Pred Robot', s=100)
# 标注机器人索引
for i, (x, y) in enumerate(best_pred_position):
    plt.annotate(f"Robot {i}", (x, y), xytext=(5, 5), textcoords='offset points')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Model Predicted Formation")
plt.grid(True, alpha=0.3)
plt.legend()

# 保存图片
plt.tight_layout()
plt.savefig("/home/lpp/formation_test/data/formation_comparison.png")
plt.show()
