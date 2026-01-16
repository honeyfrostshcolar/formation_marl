import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import formation_core 

class FormationTrainer:
    """
    编队学习训练器 - 实现混合学习策略
    """
    
    def __init__(self, model, optimizer, device, cpp_evaluator, grid_map, control_graphs, safetyThreshold, maxCommDistance):
        self.model = model.to(device) # 把模型放到GPU/CPU（就是ConstrainedFormationNet）
        self.optimizer = optimizer # 优化器（用于更新模型参数，是谁？？？）
        self.device = device # 训练设备（cuda/GPU 或 cpu）
        self.cpp_evaluator = cpp_evaluator # C++评估器（判断编队好坏的核心工具， RL中“奖励信号”的来源）
        self.grid_map = grid_map
        self.safetyThreshold = safetyThreshold
        self.maxCommDistance = maxCommDistance
        self.control_graphs = control_graphs
        
        # 训练历史（记录每个epoch的损失和验证分数，可用于可视化）
        self.train_losses = []
        self.val_scores = []
        
    def train_epoch(self, dataloader, loss_fn, training_phase="mixed", 
                   imitation_weight=0.7, curriculum_stage=0):
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器（之前CurriculumDataLoader生成的批量数据）
            loss_fn: 损失函数（之前的 HybridLoss）
            training_phase: 训练阶段
            imitation_weight: 模仿学习权重
            curriculum_stage: 课程学习阶段
        """
        self.model.train() # 把模型设置为训练模式（启用 dropout 和 batch normalization）
        total_loss = 0
        loss_components = {'imitation_loss': 0, 'rl_loss': 0, 'diversity_loss': 0}
        
        # 更新损失函数权重
        loss_fn.imitation_weight = imitation_weight
        loss_fn.rl_weight = 1.0 - imitation_weight
        
        progress_bar = tqdm(dataloader, desc=f"Training ({training_phase})") # 进度条（显示训练进度）
        
        for batch_idx, batch in enumerate(progress_bar):

            # sample_dict = {
            #     'features': features,
            #     'leader_pose': leader_pose,
            #     'environment_type': environment_type,
            #     'has_expert': has_expert,
            #     'controlGraph': ctrlGraph,
            #     'expert_positions': expert_positions
            # }

            # 准备数据
            features = batch['features'].to(self.device)

            if torch.isnan(features).any() or torch.isinf(features).any():
                print(f"❌ Batch {batch_idx}: 输入数据包含NaN或Inf")
            
            leader_pose = batch['leader_pose']
            environment_type = batch['environment_type']
            
            has_expert = batch.get('has_expert', None)
            expert_positions = batch.get('expert_positions', None)
            expert_graph = batch.get('controlGraph', None)

            # print(features)
            # print(leader_pose)
            # print(environment_type)
            # print(expert_graph)
            
            
            # 前向传播（前向传播仅负责 “基于当前权重做预测”）
            # 网络调整阶段
            # 预测的位置： [batch_size, num_graphs, num_robots, 2] 包括了所有控制图对跟随者的预测位置（领航者位置是相对位置[0,0]）
            # 预测的编队分数： [batch_size, num_graphs] 包括了每个控制图的评分
            pred_positions, pred_scores = self.model(features, training_phase) #初始化的时候ConstrainedFormationNet就是model
            
            # 计算优势函数（强化学习），这个优势函数计算的是Environment& env下 FormationConfig& formation的优势
            advantages = self._compute_advantages(pred_positions, leader_pose, expert_graph,
                                                environment_type, curriculum_stage
                                                )
            
            # if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            #    print("❌ 优势函数包含NaN/Inf，重置为零")
            
            # print(333333333333)
            # 准备专家数据
            if expert_positions is not None:
                expert_positions = expert_positions.to(self.device)
                # expert_graph_idx = expert_graph_idx.to(self.device) #这个现在我已经删了
                has_expert_mask = has_expert.to(self.device)
                has_expert_mask = has_expert_mask.bool()
                # print("has_expert_mask:", has_expert_mask)
                # print("expert_positions:", expert_positions)
                # sys.exit("stop here")
            else:
                # 如果没有专家数据，创建空的mask
                has_expert_mask = torch.zeros(features.size(0), dtype=torch.bool, device=self.device)
                expert_positions = torch.zeros_like(pred_positions[:, 0])
                # expert_graph_idx = torch.zeros(features.size(0), dtype=torch.long, device=self.device)
            
            # 计算损失(强化学习就在里面)
            # loss是加权损失，losses_dict是各个部分的损失
            loss, losses_dict = loss_fn(
                pred_positions, pred_scores, expert_positions,
                expert_graph, advantages, has_expert_mask
            )
            
            # print("loss" ,loss)
            # print("losses_dict", losses_dict)
            # 反向传播
            self.optimizer.zero_grad() # 清空上一轮的梯度（为当前轮的梯度计算做准备）
            loss.backward()  # 计算当前损失的梯度（反向传播）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # 梯度裁剪（防止梯度爆炸）
            self.optimizer.step() # 更新模型参数（根据梯度和学习率）
            
            for name, param in self.model.named_parameters():
                has_nan = torch.isnan(param).any()
                has_inf = torch.isinf(param).any()
                if has_nan or has_inf:
                    print(f"❌ 参数 {name} 被污染：nan={has_nan.item()}, inf={has_inf.item()}")
                    sys.exit("stop here")
                    break  # 找到第一个污染的参数，直接退出
          

            # for key, val in losses_dict.items():
            #     print(f"key: {key}, 类型: {type(val)}, 值: {val}")

            # 记录损失
            total_loss += loss.item() # item() 方法将单元素张量转换为 Python 数值 
            for key in loss_components:
                loss_components[key] += losses_dict[key].item()
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'imitation': f'{loss_components["imitation_loss"]/(batch_idx+1):.4f}',
                'rl': f'{loss_components["rl_loss"]/(batch_idx+1):.4f}'
            })
        
        # 计算平均损失
        avg_total_loss = total_loss / len(dataloader)
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        
        self.train_losses.append(avg_total_loss)
        
        return avg_total_loss, loss_components
    
    def _compute_advantages(self, pred_positions, leader_poses, expert_graphs, environment_types, 
                            curriculum_stage):
        """
        计算优势函数 - 基于评估函数得分
        """
        batch_size, num_graphs = pred_positions.shape[:2] # 批量大小（batch_size）和每个样本的编队数量（num_graphs）
        advantages = torch.zeros(batch_size, num_graphs, device=self.device)
        
        # 为每个样本和每个编队计算得分
        for i in range(batch_size):
            # print(leader_poses)
            # leader_pose = leader_poses[i]
            env_type = environment_types[i]
            expert_graph = expert_graphs[i]
            # print(expert_graph)
            for j in range(num_graphs):
                # 获取预测的位置配置
                positions = pred_positions[i, j].detach().cpu().numpy() # 提取第i个样本、第j个编队的预测位置（相对位置）
                # print(positions)
                
                # 转换为绝对坐标
                # absolute_positions = self._relative_to_absolute(positions, leader_pose)

                # 
                formation_config = formation_core.FormationConfig()
                formation_config.control_graph = expert_graph.tolist()

                positions_list = []
                for pos in positions:
                    pos_cpp = formation_core.Point2D()
                    pos_cpp.x = pos[0]
                    pos_cpp.y = pos[1]
                    positions_list.append(pos_cpp)

                formation_config.positions = positions_list

                leader_pose_cpp = formation_core.RobotState()
                leader_pose_cpp.position.x = leader_poses["position"]["x"][i]
                leader_pose_cpp.position.y = leader_poses["position"]["y"][i]
                leader_pose_cpp.orientation = leader_poses["orientation"][i]

                # 转换为C++环境
                env = formation_core.Environment(self.grid_map, leader_pose_cpp, self.safetyThreshold, self.maxCommDistance) # 最后两个参数是安全阈值（碰撞风险）和最大通信距离

                # print(formation_config.control_graph)
                # 使用C++评估函数计算得分：evaluateFormation(const Environment& env, const FormationConfig& formation) 
                score = self.cpp_evaluator.evaluate_formation(env, formation_config)
                advantages[i, j] = score

        # print("advantages before norm:", advantages)
        # sys.exit("stop here")
        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def _relative_to_absolute(self, relative_positions, leader_pose):
        """将相对位置转换为绝对位置"""
        absolute_positions = []
        cos_theta = np.cos(leader_pose.orientation)
        sin_theta = np.sin(leader_pose.orientation)
        
        for rel_pos in relative_positions:
            # 坐标变换
            abs_x = leader_pose.position.x + rel_pos[0] * cos_theta - rel_pos[1] * sin_theta
            abs_y = leader_pose.position.y + rel_pos[0] * sin_theta + rel_pos[1] * cos_theta
            absolute_positions.append([abs_x, abs_y])
        
        return absolute_positions
    
    def validate(self, dataloader, curriculum_stage=0):
        """验证模型性能"""
        self.model.eval()  # 把模型设置为评估模式（禁用 dropout 和 batch normalization）
        total_score = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                leader_poses = batch['leader_pose']
                environment_type = batch['environment_type']
                
                # 前向传播
                pred_positions, pred_scores = self.model(features, "imitation")
                # print(pred_scores)
                
                # 选择最佳编队
                best_graph_idx = torch.argmax(pred_scores, dim=1) # 找到每个样本中得分最高的编队索引（dim=1 表示在每个样本的(num_graphs)个得分中选择最大的）
                # print(best_graph_idx)
                best_positions = pred_positions[torch.arange(pred_positions.size(0)), best_graph_idx]
                # print(best_positions)
                
                # 计算得分
                for i in range(features.size(0)):

                    # absolute_positions = self._relative_to_absolute(
                    #     best_positions[i].cpu().numpy(), leader_pose[i]
                    # )

                    # positions = pred_positions[i, j].detach().cpu().numpy() # 提取第i个样本、第j个编队的预测位置（相对位置）
                  
                    formation_config = formation_core.FormationConfig()
                    formation_config.control_graph = self.control_graphs[best_graph_idx[i]].tolist()
                    # print("formation_config.control_graph:", formation_config.control_graph)

                    positions_list = []
                    for pos in best_positions[i]:
                        pos_cpp = formation_core.Point2D()
                        pos_cpp.x = pos[0]
                        pos_cpp.y = pos[1]
                        positions_list.append(pos_cpp)

                    formation_config.positions = positions_list

                    leader_pose_cpp = formation_core.RobotState()
                    leader_pose_cpp.position.x = leader_poses["position"]["x"][i]
                    leader_pose_cpp.position.y = leader_poses["position"]["y"][i]
                    leader_pose_cpp.orientation = leader_poses["orientation"][i]

                    # 转换为C++环境
                    env = formation_core.Environment(self.grid_map, leader_pose_cpp, self.safetyThreshold, self.maxCommDistance) # 最后两个参数是安全阈值（碰撞风险）和最大通信距离

                    # print(formation_config.control_graph)
                    # 使用C++评估函数计算得分：evaluateFormation(const Environment& env, const FormationConfig& formation) 
                    score = self.cpp_evaluator.evaluate_formation(env, formation_config)
                    
                    total_score += score
                    num_samples += 1
        
        avg_score = total_score / num_samples if num_samples > 0 else 0
        self.val_scores.append(avg_score)
        
        return avg_score
    
    def save_checkpoint(self, path, epoch, curriculum_stage):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(), # 模型的可训练参数（权重、偏置等，通过 self.model.state_dict() 获取）
            'optimizer_state_dict': self.optimizer.state_dict(), # 优化器的实时状态（Adam 的动量、学习率缓存、参数更新轨迹等）
            'train_losses': self.train_losses, # 训练损失列表
            'val_scores': self.val_scores, # 验证得分列表
            'curriculum_stage': curriculum_stage # 课程学习阶段
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_scores = checkpoint['val_scores']
        return checkpoint['epoch'], checkpoint['curriculum_stage']
    
    def plot_training_progress(self, save_path=None):
        """绘制训练进度图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 绘制损失曲线
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制验证得分曲线
        if self.val_scores:
            ax2.plot(self.val_scores, label='Validation Score', color='orange')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Score')
            ax2.set_title('Validation Performance')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()