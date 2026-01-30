

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import argparse
import cv2

from datetime import datetime

# 添加模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.formation_net import ConstrainedFormationNet
from training.trainer import PPOTrainer
from envs.formation_env import FormationEnvironment
from envs.reward_fn import FormationReward
from training.curriculum import CurriculumDataLoader
import formation_core

def parse_args():
    parser = argparse.ArgumentParser(description='PPO Formation Training')
    parser.add_argument('--num-robots', type=int, default=3, help='Number of robots')
    parser.add_argument('--feature-dim', type=int, default=21, help='Feature dimension')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num-episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size') # 批次大小
    parser.add_argument('--data-dir', type=str, default='formation_data_stage1_easy_train.csv', help='Training data directory') # 训练数据目录
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'ppo_train_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 初始化C++模块
    try:
        cpp_enumerator = formation_core.FormationEnumerator(args.num_robots)
        all_formations = cpp_enumerator.get_all_formations()
        adj_np_list = []
        for cg in all_formations:
            adj_np = cg.get_adjacency_matrix()
            adj_np_list.append(adj_np)
        
        candidate_graphs = [torch.from_numpy(adj).float() for adj in adj_np_list]
        num_graphs = len(candidate_graphs)
        print(f'Successfully loaded {num_graphs} control graphs from C++')
    except ImportError:
        print('Warning: C++ core module not found, using dummy graphs')
        # 创建虚拟图
        candidate_graphs = []
        num_graphs = 5
    
    # 初始化模型
    model = ConstrainedFormationNet(
        feature_dim=args.feature_dim,
        num_graphs=num_graphs,
        max_robots=args.num_robots
    ).to(device)
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 初始化奖励函数
    reward_fn = FormationReward(
        safety_threshold=0.5,
        max_comm_distance=5.0,
        formation_weight=0.4,
        safety_weight=0.3,
        comm_weight=0.3
    )

    # 学习数据加载器
    curriculum_loader = CurriculumDataLoader(args.data_dir, args.batch_size)
    # 获取数据加载器
    train_loader = curriculum_loader.get_stage_dataloader() # 训练数据加载器（打乱，加载、处理用于数据的训练）

    # 初始化环境
    # 读取PGM图像（OpenCV自动识别PGM格式）
    # pgm_path = args.map  # 替换为你的PGM文件实际路径
    # obstacle_threshold = 128  # 灰度阈值，低于该值视为障碍物
    # img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        
    # if img is None:
    #     print(f"错误：无法读取PGM文件 {pgm_path}（可能路径错误或格式不支持）")
            
    # # 灰度值转栅格值
    # grid_map = np.where(img < obstacle_threshold, 1, 0).tolist()

    # 注意：这里使用虚拟地图，需要替换为真实的地图数据
    grid_map = [[0 for _ in range(200)] for _ in range(200)]
    env = FormationEnvironment(
        grid_map=grid_map,
        num_robots=args.num_robots,
        candidate_graphs=candidate_graphs,
        safety_threshold=0.5,
        max_comm_distance=5.0
        train_loader=train_loader
    )
    
    # 初始化PPO训练器
    trainer = PPOTrainer(
        model=model,
        optimizer=optimizer,
        reward_fn=reward_fn,
        device=device,
        config={
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'ppo_epochs': 4,
            'mini_batch_size': 32
        }
    )

    # 恢复训练
    start_episode = 0
    if args.resume and os.path.exists(args.resume):
        print(f'Resuming from checkpoint: {args.resume}')
        start_episode = trainer.load_checkpoint(args.resume)
    
    # 训练循环
    print(f'\n{"="*50}')
    print(f'Starting PPO training for {args.num_episodes} episodes')
    print(f'{"="*50}\n')
    
    for episode in range(start_episode, args.num_episodes):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # 环境交互循环
        for step in range(args.max_steps):
            
            action, log_prob, value, outputs = trainer.act(state, candidate_graphs)
        
            # 2. 执行动作（与环境交互）
            next_state, reward, done, reward_info = env.step(action)
        
            # 3. 存储经验
            trainer.store_experience(state, candidate_graphs, action, log_prob, value, reward, done)
        
            # 更新状态
            state = next_state

            episode_reward += reward
            episode_length += 1
            
        
        # 完成一个episode，进行训练
        trainer.finish_episode(episode_reward, episode_length)
    
        # 获取训练统计
        losses_dict = trainer.update() if hasattr(trainer, 'update') else None
        loss = losses_dict['total_loss'] if losses_dict is not None else None
        
        # 记录历史
        trainer.history['episode_rewards'].append(episode_reward)
        trainer.history['episode_lengths'].append(episode_length)
        if loss is not None:
            trainer.history['losses'].append(loss)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(trainer.history['episode_rewards'][-10:])
            trainer.history['avg_rewards'].append(avg_reward)
            
            print(f'Episode {episode + 1}/{args.num_episodes}:')
            print(f'  Reward: {episode_reward:.2f} | Length: {episode_length}')
            print(f'  Avg Reward (last 10): {avg_reward:.2f}')
            if loss is not None:
                print(f'  Loss: {loss:.4f}')
            print()
        
        # 定期保存
        if (episode + 1) % 50 == 0:
            checkpoint_path = os.path.join(
                output_dir, f'checkpoint_episode_{episode+1}.pth'
            )
            trainer.save_checkpoint(checkpoint_path, episode)
            print(f'Checkpoint saved: {checkpoint_path}')
            
            # 可视化训练进度
            if hasattr(trainer, 'plot_training_progress'):
                plot_path = os.path.join(output_dir, f'training_progress_ep{episode+1}.png')
                trainer.plot_training_progress(plot_path)
    
    # 训练完成
    print(f'\n{"="*50}')
    print(f'Training completed!')
    print(f'{"="*50}\n')
    
    # 保存最终模型
    final_path = os.path.join(output_dir, 'final_model.pth')
    trainer.save_checkpoint(final_path, args.num_episodes)
    print(f'Final model saved: {final_path}')
    
    # 输出训练统计
    print(f'\nTraining Statistics:')
    print(f'  Total episodes: {args.num_episodes}')
    print(f'  Average reward: {np.mean(trainer.history["episode_rewards"]):.2f}')
    print(f'  Max reward: {np.max(trainer.history["episode_rewards"]):.2f}')
    print(f'  Min reward: {np.min(trainer.history["episode_rewards"]):.2f}')

if __name__ == '__main__':
    main()

import torch
import numpy as np

class FormationReward:
    """编队任务奖励函数"""
    def __init__(self, safety_threshold=0.5, max_comm_distance=5.0,
                 formation_weight=0.4, safety_weight=0.3, comm_weight=0.3):
        self.safety_threshold = safety_threshold # 最小安全距离
        self.max_comm_distance = max_comm_distance # 最大通信距离
        self.formation_weight = formation_weight # 编队质量权重
        self.safety_weight = safety_weight # 安全权重
        self.comm_weight = comm_weight # 通信质量权重
    
    def compute(self, state, action, next_positions):
        """计算总奖励"""
        reward_components = {}
        
        # 1. 编队质量奖励
        formation_reward = self._formation_quality_reward(
            action['graph'], next_positions
        )
        reward_components['formation'] = formation_reward
        
        # 2. 安全奖励
        safety_penalty = self._safety_penalty(next_positions)
        reward_components['safety'] = -safety_penalty
        
        # 3. 通信质量奖励
        comm_reward = self._communication_reward(
            action['graph'], next_positions
        )
        reward_components['communication'] = comm_reward
        
        # 4. 任务特定奖励
        task_reward = self._task_reward(state, next_positions)
        reward_components['task'] = task_reward
        
        # 加权总奖励
        total_reward = (
            self.formation_weight * formation_reward +
            self.safety_weight * (-safety_penalty) +
            self.comm_weight * comm_reward +
            task_reward
        )
        
        return total_reward, reward_components
    
    def _formation_quality_reward(self, graph, positions):
        """编队质量奖励"""
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        
        num_robots = positions.shape[0]
        reward = 0.0
        
        # 检查图连接
        for i in range(num_robots):
            for j in range(num_robots):
                if graph[i, j] > 0:
                    distance = torch.norm(positions[i] - positions[j])
                    # 理想距离设为1.0，实际可根据任务调整
                    ideal_distance = 1.0
                    error = torch.abs(distance - ideal_distance)
                    reward += torch.exp(-error)
        
        return reward.item() if isinstance(reward, torch.Tensor) else reward
    
    def _safety_penalty(self, positions):
        """安全惩罚"""
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        
        num_robots = positions.shape[0]
        penalty = 0.0
        
        # 机器人间碰撞检测
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                distance = torch.norm(positions[i] - positions[j])
                if distance < self.safety_threshold:
                    penalty += (self.safety_threshold - distance) * 10.0
        
        return penalty.item() if isinstance(penalty, torch.Tensor) else penalty
    
    def _communication_reward(self, graph, positions):
        """通信质量奖励"""
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        
        num_robots = positions.shape[0]
        reward = 0.0
        connections = 0
        
        for i in range(num_robots):
            for j in range(num_robots):
                if graph[i, j] > 0:
                    distance = torch.norm(positions[i] - positions[j])
                    if distance <= self.max_comm_distance:
                        # 在通信范围内，距离越近通信质量越好
                        reward += 1.0 - (distance / self.max_comm_distance)
                    else:
                        # 超出通信范围，惩罚
                        reward -= 1.0
                    connections += 1
        
        if connections > 0:
            reward /= connections
        
        return reward.item() if isinstance(reward, torch.Tensor) else reward
    
    def _task_reward(self, state, positions):
        """任务特定奖励，根据你的具体任务定义"""
        # 示例：如果接近目标区域，给予奖励
        target_area = np.array([5.0, 5.0])  # 假设目标区域
        if isinstance(positions, torch.Tensor):
            center = positions.mean(dim=0).numpy()
        else:
            center = positions.mean(axis=0)
        
        distance_to_target = np.linalg.norm(center - target_area)
        return np.exp(-distance_to_target)


import numpy as np
import torch

from envs.reward_fn import FormationReward

class FormationEnvironment:
    """编队控制环境"""
    def __init__(self, grid_map, num_robots, candidate_graphs, train_loader,
                 safety_threshold=0.5, max_comm_distance=5.0):
        self.grid_map = grid_map
        self.num_robots = num_robots
        self.candidate_graphs = candidate_graphs
        self.safety_threshold = safety_threshold
        self.max_comm_distance = max_comm_distance
        self.train_loader = train_loader
        self.batch_size = train_loader.batch_size 
        self.data_iter = iter(train_loader)  # 批量数据迭代器
        self.current_batch_states = None  # 关键修复：初始化批量状态变量
        self.step_count = 0
        self.max_steps = 100

        self.device = next(iter(candidate_graphs)).device if candidate_graphs else 'cpu'

        self.reward_fn = FormationReward(
            safety_threshold=0.5,
            max_comm_distance=5.0,
            formation_weight=0.4,  # 可自定义
            safety_weight=0.3,        # 可自定义
            comm_weight=0.3             # 可自定义
        )
        
    def reset(self):
        """批量重置：一次性返回一个批次的初始状态"""
        self.step_count = 0
        # 加载下一个批次数据
 
        self.current_batch = next(self.data_iter) # 一次拿到的是一整批（batch）数据
        
        batch_states = []
        # 批量处理当前批次的每个样本
        for i in range(self.batch_size):
            # 从批量数据中提取单个样本的信息
            features = self.current_batch['features'][i]  # 21维环境特征（批量维度：[batch, 21]）
            leader_x = self.current_batch['leader_pose']['position']['x'][i].item()  # 领航x（批量维度：[batch]）
            leader_y = self.current_batch['leader_pose']['position']['y'][i].item()  # 领航y
            leader_orientation = self.current_batch['leader_pose']['orientation'][i].item()  # 领航朝向
            robot_count = self.current_batch['robot_nums'][i].item()  # 机器人数量
            
            # 批量生成初始位置：领航者用文件数据，跟随者占位（后续模型批量生成）
            positions = [np.array([leader_x, leader_y, leader_orientation])]
            for _ in range(robot_count - 1):
                positions.append(np.array([0.0, 0.0, 0.0]))  # 占位
            
            # 单个样本的状态，最后组合成批次
            batch_states.append({
                'features': features,
                'positions': np.array(positions), 
                'robot_count': robot_count,
                'step': self.step_count,
                'done': False
            })
        
        self.current_batch_states = batch_states
        
        return batch_states  # 返回：[batch_size, state_dict]
    
    def step(self, batch_actions):
        """批量执行动作：接收批量动作，返回批量下一个状态、奖励、终止标志"""
        self.step_count += 1
        batch_next_states = []
        batch_rewards = []
        batch_dones = []
        batch_infos = []

        try:
            next_batch = next(self.data_iter)
        except StopIteration:
            # 所有batch取完，重新初始化迭代器（进入下一个epoch）
            self.data_iter = iter(self.train_loader)
            next_batch = next(self.data_iter)
        
        # 批量处理每个样本的动作
        for i in range(self.batch_size):
            action = batch_actions[i]  # 单个样本的动作：{graph_idx, position}
            current_state = self.current_batch_states[i]  # 上一步的批量状态
            
            # 1. 计算跟随者绝对位置
            current_pos = current_state['positions'] 
            updated_pos = current_pos.copy()
            leader_abs_pos = current_state['positions'][0][:2]
            updated_pos[1:] = leader_abs_pos + action['position'].cpu().numpy()  # 相对→绝对转换
            
            # 2. 检查终止条件（批量统一判断：达到最大步数则终止）
            done = self.step_count >= self.max_steps
            
            # 3. 批量计算奖励（复用原Reward函数，适配单样本后汇总）
            graph = self.candidate_graphs[action['graph_idx']]
            reward, reward_info = self.reward_fn.compute(
                state=current_state,
                action={'graph': graph, **action}, # 创建立一个新的action字典，包含graph
                next_positions=updated_pos
            )
            
            # 生成下一个状态
            features = next_batch['features'][i] 
            leader_x = next_batch['leader_pose']['position']['x'][i].item() 
            leader_y = next_batch['leader_pose']['position']['y'][i].item()  
            leader_orientation = next_batch['leader_pose']['orientation'][i].item()  
            robot_count = next_batch['robot_nums'][i].item()  
            
            # 批量生成初始位置：领航者用文件数据，跟随者占位（后续模型批量生成）
            positions = [np.array([leader_x, leader_y, leader_orientation])]
            for _ in range(robot_count - 1):
                positions.append(np.array([0.0, 0.0, 0.0]))  # 占位

            next_state = {
                'features': features,
                'positions': np.array(positions), 
                'robot_count': robot_count,
                'step': self.step_count,
                'done': False
            }

            batch_next_states.append(next_state)
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_infos.append(reward_info)
        
        # 更新当前批次状态
        self.current_batch_states = batch_next_states
        # 转换为张量
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(self.device)
        batch_dones = torch.tensor(batch_dones, dtype=torch.bool).to(self.device)
        
        return batch_next_states, batch_rewards, batch_dones, batch_infos
    
    def _random_init_positions(self):
        """随机初始化位置"""
        positions = []
        leader_pos = np.array([0.0, 0.0])  # 领导者在中心
        positions.append(leader_pos)
        
        # 随机生成跟随者位置
        for _ in range(self.num_robots - 1):
            angle = np.random.uniform(-np.pi, np.pi)
            distance = np.random.uniform(1.0, 2.0)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            positions.append(np.array([x, y]))
        
        return np.array(positions)
    
    def _extract_features(self, positions):
        """从位置提取特征"""
        # 实现特征提取逻辑
        features = np.random.randn(21)  # 临时，需要根据实际实现
        return features
    
    def _check_done(self):
        """检查是否终止"""
        # 达到最大步数或完成任务
        if self.step_count >= self.max_steps:
            return True
        return False
    
    def render(self, mode='human'):
        """可视化环境"""
        # 实现可视化逻辑
        pass


import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy

class PPOTrainer:
    def __init__(self, model, optimizer, device='cuda', config=None):
        self.model = model
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        self.optimizer = optimizer
        self.device = device
        
        # 配置参数
        self.config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'ppo_epochs': 4,
            'mini_batch_size': 32,
            'max_grad_norm': 0.5
        }
        if config:
            self.config.update(config)
        
        # 经验缓冲区（使用列表存储每个episode的经验）
        self.buffer = deque(maxlen=1000)  # 存储整个episode
        
        # 训练历史
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'value_losses': [],
            'policy_losses': [],
            'entropy_losses': []
        }
    
    def act(self, state, candidate_graphs):
        """
        使用旧策略选择动作
        
        Args:
            state: 包含features和robot_ids的字典
                    'features': features,
                    'positions': positions,
                    'robot_count': self.num_robots,
                    'step': self.step_count,
                    'done': False

            candidate_graphs: 候选图列表
            
        Returns:
            action: 动作字典 {graph_idx, position}
            log_prob: 动作的对数概率
            value: 状态价值
            outputs: 模型输出（用于后续存储）
        """
        with torch.no_grad():
            # 确保输入在正确的设备上
            features = state['features'].unsqueeze(0).to(self.device) if len(state['features'].shape) == 1 else state['features'].to(self.device)
            robot_count = state['robot_count']
            robot_ids = torch.arange(0, robot_count, dtype=torch.long, device=self.device)
            
            # 前向传播
            outputs = self.old_model(features, robot_ids, candidate_graphs, robot_count)
            # 'graph_scores': graph_scores,      
            # 'position_dists': position_dists,  
            # 'value': value,                   
            # 'graph_features': all_graph_features 
            
            # 采样图选择
            graph_scores = outputs['graph_scores']
            graph_probs = F.softmax(graph_scores, dim=-1)
            graph_dist = torch.distributions.Categorical(graph_probs)
            graph_idx = graph_dist.sample().item()  # 采样图索引（轮盘赌方式得到）
            
            # 采样位置
            position_dist = outputs['position_dists'][graph_idx]
            position = position_dist.sample().squeeze(0)  # [robot_count-1, 2]
            
            # 计算对数概率
            graph_log_prob = graph_dist.log_prob(torch.tensor([graph_idx], device=self.device))
            position_log_prob = position_dist.log_prob(position.unsqueeze(0)).sum()
            total_log_prob = graph_log_prob + position_log_prob
            
            # 状态价值
            value = outputs['value'].squeeze()
            
            action = {
                'graph_idx': graph_idx,
                'position': position.cpu() if self.device != 'cpu' else position
            }
            
            return action, total_log_prob.item(), value.item(), outputs
    
    def store_experience(self, state, candidate_graphs, action, log_prob, value, reward, done):
        """
        存储单步经验
        
        Args:
            state: 批量当前状态 [batch_size, state_dict]
            candidate_graphs: 候选图列表（批量共用）
            action: 批量动作 [batch_size, {graph_idx, position}]
            log_prob: 批量对数概率 [batch_size]（张量）
            value: 批量状态价值 [batch_size]（张量）
            reward: 批量奖励 [batch_size]（张量）
            done: 批量终止标志 [batch_size]（张量）
        """
        experience = {
            'state': state.copy(),
            'candidate_graphs': candidate_graphs,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'done': done
        }
        self.buffer.append(experience)
    
    def compute_returns_and_advantages(self, last_value=0):
        """
        批量计算回报和优势函数：用PyTorch张量向量化运算，替代逐个样本循环
        Returns:
            experiences: 带回报和优势的单个样本经验列表 [总样本数]
        """
        if not self.buffer:
            return
        
        # 1. 批量提取经验，转换为PyTorch张量
        experiences = list(self.buffer)
        total_samples = len(experiences)
        
        # 提取核心数据，组成 [total_samples] 维度的张量
        rewards = torch.tensor([exp['reward'] for exp in experiences], dtype=torch.float32, device=self.device)
        values = torch.tensor([exp['value'] for exp in experiences], dtype=torch.float32, device=self.device)
        dones = torch.tensor([exp['done'] for exp in experiences], dtype=torch.float32, device=self.device)  # 0/1 张量
        
        # 2. 批量计算 next_values 和 next_nonterminals（无需循环，用切片实现）
        next_values = torch.zeros_like(values)  # [total_samples]
        next_nonterminals = 1.0 - dones  # [total_samples]（非终止标志：0=终止，1=继续）
        
        # 除了最后一个样本，其余样本的next_value = 下一个样本的value
        next_values[:-1] = values[1:]
        # 最后一个样本的next_value = 传入的last_value（终止状态为0）
        next_values[-1] = last_value
        
        # 3. 批量计算TD误差（向量化运算，一次性完成所有样本）
        gamma = self.config['gamma']
        gae_lambda = self.config['gae_lambda']
        deltas = rewards + gamma * next_values * next_nonterminals - values  # [total_samples]
        
        # 4. 批量反向计算GAE（核心：用累积乘积替代循环）
        # 先计算每个时间步的衰减系数：(gamma * gae_lambda) ^ t
        # 生成 [total_samples] 维度的衰减系数张量
        steps = torch.arange(total_samples, device=self.device)
        decay_coeffs = (gamma * gae_lambda) ** steps  # [total_samples]（从 (gammaλ)^0 到 (gammaλ)^(T-1)）
        
        # 反向累积GAE：gae[t] = delta[t] + gammaλ * next_nonterminal[t] * gae[t+1]
        # 用flip+cumprod+flip实现反向累积，避免循环
        deltas_flipped = deltas.flip(dims=[0])  # 反转deltas：[T-1, T-2, ..., 0]
        next_nonterminals_flipped = next_nonterminals.flip(dims=[0])  # 反转非终止标志
        
        # 计算累积乘积：每个位置的累积系数 = product(gammaλ * next_nonterminal[k] for k >= t)
        cum_prod = torch.cumprod(gamma * gae_lambda * next_nonterminals_flipped, dim=0)
        # 第一个位置的累积系数为1（因为没有后续步骤）
        cum_prod = torch.cat([torch.ones(1, device=self.device), cum_prod[:-1]])
        
        # 批量计算GAE：每个样本的gae = delta[t] * 累积系数[t]
        gae_flipped = deltas_flipped * cum_prod
        gae = gae_flipped.flip(dims=[0])  # 反转回原顺序：[0, 1, ..., T-1]
        
        # 5. 批量计算回报和标准化优势
        returns = gae + values  # [total_samples]（回报 = GAE + 状态价值）
        
        # 优势函数标准化（张量批量运算，比numpy更快）
        advantages = (gae - gae.mean()) / (gae.std() + 1e-8)  # [total_samples]
        
        # 6. 批量更新经验中的回报和优势（循环仅用于赋值，无计算）
        for i, exp in enumerate(experiences):
            exp['return'] = returns[i].item()
            exp['advantage'] = advantages[i].item()
        
        return experiences

    
    def update(self):
        """
        执行PPO更新
        
        Returns:
            losses_dict: 各项损失的字典
        """
        if len(self.buffer) < self.config['mini_batch_size']:
            return None
        
        # 计算回报和优势
        experiences = self.compute_returns_and_advantages()
        
        # 转换为张量
        states = []
        candidate_graphs_list = []
        actions = []
        old_log_probs = []
        advantages = []
        returns = []
        
        for exp in experiences:
            states.append(exp['state'])
            candidate_graphs_list.append(exp['candidate_graphs'])
            actions.append({
                'graph_idx': torch.tensor([exp['action']['graph_idx']], device=self.device),
                'position': exp['action']['position'].unsqueeze(0).to(self.device)
            })
            old_log_probs.append(exp['log_prob'])
            advantages.append(exp['advantage'])
            returns.append(exp['return'])
        
        # 转换为张量
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        
        # 多轮PPO更新
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for epoch in range(self.config['ppo_epochs']):
            # 随机打乱索引
            indices = torch.randperm(len(experiences))
            
            # 小批量处理
            for start_idx in range(0, len(indices), self.config['mini_batch_size']):
                batch_indices = indices[start_idx:start_idx+self.config['mini_batch_size']]
                
                # 收集小批量数据
                batch_loss = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_entropy_loss = 0
                
                for idx in batch_indices:
                    state = states[idx.item()]
                    candidate_graphs = candidate_graphs_list[idx.item()]
                    action = actions[idx.item()]
                    
                    # 重新计算旧策略输出
                    with torch.no_grad():
                        features = state['features'].unsqueeze(0).to(self.device)
                        robot_ids = state['robot_ids'].to(self.device)
                        robot_count = state['robot_count']
                        
                        old_outputs = self.old_model(features, robot_ids, candidate_graphs, robot_count)
                    
                    # 计算新策略输出
                    new_outputs = self.model(features, robot_ids, candidate_graphs, robot_count)
                    
                    # 计算图选择损失
                    old_graph_logits = old_outputs['graph_scores']
                    old_graph_probs = F.softmax(old_graph_logits, dim=-1)
                    old_graph_log_prob = torch.log(old_graph_probs[0, action['graph_idx']])
                    
                    new_graph_probs = F.softmax(new_outputs['graph_scores'], dim=-1)
                    new_graph_log_prob = torch.log(new_graph_probs[0, action['graph_idx']])
                    
                    # 计算位置损失
                    graph_idx = action['graph_idx'].item()
                    old_position_dist = old_outputs['position_dists'][graph_idx]
                    old_position_log_prob = old_position_dist.log_prob(action['position']).sum()
                    
                    new_position_dist = new_outputs['position_dists'][graph_idx]
                    new_position_log_prob = new_position_dist.log_prob(action['position']).sum()
                    
                    # 总的对数概率
                    old_total_log_prob = old_graph_log_prob + old_position_log_prob
                    new_total_log_prob = new_graph_log_prob + new_position_log_prob
                    
                    # 计算比率
                    ratio = torch.exp(new_total_log_prob - old_total_log_prob)
                    
                    # PPO clip损失
                    adv = advantages[idx]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 
                                       1 + self.config['clip_epsilon']) * adv
                    policy_loss = -torch.min(surr1, surr2)
                    
                    # 价值损失
                    value_pred = new_outputs['value'].squeeze()
                    return_target = returns[idx]
                    value_loss = F.mse_loss(value_pred, return_target)
                    
                    # 熵正则化
                    graph_entropy = -(new_graph_probs * torch.log(new_graph_probs + 1e-8)).sum()
                    
                    position_entropy = 0
                    for dist in new_outputs['position_dists']:
                        position_entropy += dist.entropy().mean()
                    position_entropy = position_entropy / len(new_outputs['position_dists'])
                    
                    entropy_loss = -(graph_entropy + position_entropy) * self.config['entropy_coef']
                    
                    # 总损失
                    loss = (policy_loss + 
                           self.config['value_coef'] * value_loss + 
                           entropy_loss)
                    
                    batch_loss += loss
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_entropy_loss += entropy_loss.item()
                
                # 平均损失
                batch_size = len(batch_indices)
                avg_loss = batch_loss / batch_size
                
                # 反向传播
                self.optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                
                # 累计损失
                total_loss += avg_loss.item()
                total_policy_loss += batch_policy_loss / batch_size
                total_value_loss += batch_value_loss / batch_size
                total_entropy_loss += batch_entropy_loss / batch_size
        
        # 计算平均损失
        num_updates = self.config['ppo_epochs'] * (len(indices) // self.config['mini_batch_size'] + 1)
        
        losses_dict = {
            'total_loss': total_loss / num_updates if num_updates > 0 else 0,
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0,
            'value_loss': total_value_loss / num_updates if num_updates > 0 else 0,
            'entropy_loss': total_entropy_loss / num_updates if num_updates > 0 else 0
        }
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 更新旧模型
        self.old_model.load_state_dict(self.model.state_dict())
        
        # 记录损失
        self.history['losses'].append(losses_dict['total_loss'])
        self.history['value_losses'].append(losses_dict['value_loss'])
        self.history['policy_losses'].append(losses_dict['policy_loss'])
        self.history['entropy_losses'].append(losses_dict['entropy_loss'])
        
        return losses_dict
    
    def finish_episode(self, episode_reward, episode_length):
        """
        完成一个episode，记录结果
        
        Args:
            episode_reward: 累计奖励
            episode_length: episode长度
        """
        self.history['episode_rewards'].append(episode_reward)
        self.history['episode_lengths'].append(episode_length)
        
        # 如果缓冲区有经验，执行更新
        if len(self.buffer) > 0:
            self.update()
    
    def save_checkpoint(self, path, episode, config):
        """
        保存检查点
        
        Args:
            path: 保存路径
            episode: 当前episode数
            config: 训练配置
        """
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'old_model_state_dict': self.old_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': config,
            'history': self.history
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """
        加载检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            episode: 加载的episode数
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.old_model.load_state_dict(checkpoint['old_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        # 更新配置
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['episode']
    
    def get_training_stats(self):
        """获取训练统计信息"""
        if not self.history['episode_rewards']:
            return {}
        
        last_10_rewards = self.history['episode_rewards'][-10:] if len(self.history['episode_rewards']) >= 10 else self.history['episode_rewards']
        
        return {
            'current_episode': len(self.history['episode_rewards']),
            'avg_reward_last_10': np.mean(last_10_rewards) if last_10_rewards else 0,
            'max_reward': np.max(self.history['episode_rewards']) if self.history['episode_rewards'] else 0,
            'min_reward': np.min(self.history['episode_rewards']) if self.history['episode_rewards'] else 0,
            'avg_episode_length': np.mean(self.history['episode_lengths']) if self.history['episode_lengths'] else 0
        }


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class FormationDataset(Dataset):
    """编队学习数据集"""
    
    def __init__(self, csv_file, feature_columns, transform=None):
        """
        Args:
            csv_file: CSV文件路径
            feature_columns: 特征列名列表
            transform: 数据变换
        """
        self.data = pd.read_csv(csv_file)
        self.feature_columns = feature_columns #[]
        self.transform = transform
        
        # 确保数据包含必要的列
        required_columns = ['env_type', 'robot_x', 'robot_y', 'robot_theta', 'has_expert_label'] #robot_x、robot_y、robot_theta是领航的初始位置和方向
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def __len__(self):
        return len(self.data)
    
    # 索引获取样本
    def __getitem__(self, idx):
        sample = self.data.iloc[idx] # 获取第idx行的样本
        
        # 环境特征
        features = torch.tensor(sample[self.feature_columns].values.astype(np.float32)) #[feature_dim]
        
        # 机器人位姿
        leader_pose = {
            'position': {'x': sample['robot_x'], 'y': sample['robot_y']},
            'orientation': sample['robot_theta']
        }

        #总机器人数量
        robot_nums = sample['ctrlnums']
        
        # 环境类型
        environment_type = sample['env_type']
        # print(environment_type)
        
        sample_dict = {
            'features': features,
            'leader_pose': leader_pose,
            'environment_type': environment_type,
        }
        
        # 不打算设计
        if self.transform:
            sample_dict = self.transform(sample_dict)
        
        return sample_dict

class CurriculumDataLoader:
    """
    课程学习数据加载器
    按照环境复杂度逐步增加训练难度
    """
    
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir # 数据目录(假设data_dir是具体的CSV文件路径)
        self.batch_size = batch_size

        # 特征列定义（根据实际CSV文件调整），通过feature_columns里的特征来从CSV文件中提取数据
        self.feature_columns = [
            'corridor_width', 'front_clearance', 'left_clearance', 'right_clearance', 
            'obstacle_density'
        ] + [f'sector_{i}_min' for i in range(8)] + [f'sector_{i}_avg' for i in range(8)]
    
    def get_stage_dataloader(self):
        """获取指定数据加载器"""
        
        try:
            dataset = FormationDataset(self.data_dir, self.feature_columns)
        except FileNotFoundError:

            raise FileNotFoundError(f"错误：未找到指定文件 '{self.data_dir}'，程序终止运行！")
        
        # 创建数据加载器
        # 循环切分数据成批次；
        # 手动打乱数据顺序；
        # 单进程加载数据（速度慢）；
        # 手动处理数据加载的异常（比如样本数不是 batch_size 整数倍）；
        # 这些都是重复且易出错的工作，DataLoader 全帮你封装好了。
        # 训练时直接 for batch in dataloader 就能循环取批次数据

        # 训练时打乱数据，验证时不打乱
        dataloader = DataLoader(  
            dataset, 
            batch_size=self.batch_size, # 32
            shuffle=True,
            num_workers=4
        )
        
        return dataloader
    
    def _filter_dataset_by_stage(self, dataset, stage):
        """根据阶段过滤数据集"""
        filtered_indices = []
        
        for idx in range(len(dataset)):
            sample = dataset.data.iloc[idx]
            if sample['env_type'] in stage['env_types']:
                # 根据专家比例随机选择
                if np.random.random() < stage['expert_ratio'] or sample['has_expert_label']:
                    filtered_indices.append(idx)
        
        # 创建子集
        from torch.utils.data import Subset
        return Subset(dataset, filtered_indices) # 返回根据阶段过滤后的数据集子集
    
    def get_total_epochs(self):
        """获取总训练轮数"""
        return sum(stage['epochs'] for stage in self.curriculum_stages)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from torch_geometric.nn import GATConv

class ConstrainedFormationNet(nn.Module):
    """
    带约束的编队选择神经网络
    基于分层结构和约束位置生成
    输入固定维度的环境特征向量，输出每个控制图的安全位置配置和编队选择分数
    """
    
    def __init__(self, feature_dim, num_graphs, num_robots, 
                 min_distance=0.5, max_distance=3.0, angle_range=(-90, 90)):
        super(ConstrainedFormationNet, self).__init__()
        
        self.num_graphs = num_graphs
        self.num_robots = num_robots
        self.num_followers = num_robots - 1  # 减去领航者
        
        # 约束参数
        self.min_distance = min_distance # 跟随者与领航者的最小距离
        self.max_distance = max_distance # 跟随者与领航者的最大距离
        self.angle_min, self.angle_max = angle_range # 跟随者与领航者的角度范围

        self.input_norm = nn.LayerNorm(feature_dim)
        
        ###### 初始化神经网络模块
        # 共享特征编码器
        self.encoder = nn.Sequential(
            self.input_norm,  # 先归一化输入
            nn.Linear(feature_dim, 128),  #全连接层
            nn.ReLU(),  
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 2. GNN层（处理可变节点数）
        self.gnn_layers = nn.ModuleList([
            GATConv(4 + 32 + 16, 64, heads=2, concat=False),  # 输入: 4(机器人编码) + 32(环境) + 16(图结构) = 52
            GATConv(64, 64, heads=2, concat=False),
            GATConv(64, 64, heads=2, concat=False)
        ])
        
        # 3. 位置生成器（连续Actor）
        self.position_generator = nn.Sequential(
            nn.Linear(64, 32),  # GNN输出的节点特征
            nn.ReLU(),
            nn.Linear(32, 2)    # 每个机器人的2D位置
        )
        
        # 4. 图结构编码器（将控制图编码为特征向量）
        # 假设最大机器人数量为max_robots = 10
        self.graph_encoder = nn.Sequential(
            nn.Linear(10 * 10, 128),  # 邻接矩阵展平
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # 16维图特征
        )
        
        # 5. 编队选择器（离散Actor）
        # 不再输出固定数量的分数，而是通过图匹配计算分数
        self.graph_scorer = nn.Sequential(
            nn.Linear(32 + 16, 64),  # 环境特征 + 图特征
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出单个图的分数
        )
        
        # 6. 机器人ID嵌入
        self.robot_embedding = nn.Embedding(10, 4) # 机器人ID（最多为10个）嵌入为4维向量

        self.position_log_std = nn.Parameter(torch.zeros(1, 2))

        # 评估当前状态的价值（Critic） 
        self.critic = nn.Sequential(
            nn.Linear(32 + 16, 64),  # 环境特征 + 平均图特征
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 状态价值
        )
        
        # 位置微调模块（用于后期强化学习微调）
        self.position_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32 + 2 * self.num_followers, 32),
                nn.ReLU(),
                nn.Linear(32, 2 * self.num_followers)
            ) for _ in range(num_graphs)
        ])
        
    def forward(self, features, robot_ids, candidate_graphs, robot_count):
        """
        前向传播
        
        Args:
            env_features: 环境特征 [batch_size, feature_dim]
            training_phase: 训练阶段 ("imitation", "mixed", "rl_finetune")
            robot_count: 整数，机器人数量
            candidate_graphs: 列表，每个元素是一个邻接矩阵 [robot_count, robot_count]
                            不同机器人数量对应的图数量不同
        """
        batch_size = features.size(0) # 获取批量大小
        num_graphs = len(candidate_graphs) # 获取控制图数量

        #print("batch_size:", batch_size)
        
        # 特征编码
        env_features = self.encoder(features) # [batch_size, 32]

        graph_features_list = [] 
        for graph in candidate_graphs:
            # 将图填充到最大尺寸
            pad_size = (0, 10 - robot_count,
                       0, 10 - robot_count)
            graph_padded = F.pad(graph, pad_size, mode='constant', value=0)
            
            # 编码图结构
            graph_flat = graph_padded.flatten().unsqueeze(0)  # [1, max_robots²]
            graph_feat = self.graph_encoder(graph_flat)  # [1, 16]
            graph_features_list.append(graph_feat) # [num_graphs, 16]

        # 堆叠所有图特征 [num_graphs, 16]
        all_graph_features = torch.cat(graph_features_list, dim=0)
        # 扩展环境特征到每个图
        env_expanded = env_features.unsqueeze(1)  # [batch, 1, 32]
        env_expanded = env_expanded.expand(-1, num_graphs, -1)  # [batch, num_graphs, 32]
        
        # 扩展图特征到每个batch
        graph_expanded = all_graph_features.unsqueeze(0)  # [1, num_graphs, 16]
        graph_expanded = graph_expanded.expand(batch_size, -1, -1)  # [batch, num_graphs, 16]
        
        # 拼接特征并计算分数（combined包括了环境特征和图特征）
        combined = torch.cat([env_expanded, graph_expanded], dim=-1)  # [batch, num_graphs, 48]

        # 重塑为 [batch * num_graphs, 48] 以便批量处理
        combined_flat = combined.view(-1, 48)
        scores_flat = self.graph_scorer(combined_flat)  # [batch * num_graphs, 1]
        
        # 恢复形状
        graph_scores = scores_flat.view(batch_size, num_graphs)  # [batch, num_graphs]

        # 为每个候选图生成位置分布
        position_dists = []
        
        # 预计算机器人ID嵌入
        # follower_robot_ids = robot_ids[1:]
        robot_emb = self.robot_embedding(robot_ids)  # [batch * robot_count, 4] 
        
        # 预计算扩展的环境特征（每个机器人一份）
        env_per_robot = env_features.repeat_interleave(robot_count, dim=0)  # [batch * robot_count, 32]

        # 对每个候选图独立处理
        for i, graph in enumerate(candidate_graphs):
            # 获取当前图的特征
            graph_feat = all_graph_features[i]  # [16]
            graph_feat_per_robot = graph_feat.unsqueeze(0).expand(batch_size * robot_count, -1)  # [batch * (robot_count), 16]
            
            # 构建GNN输入
            gnn_input = torch.cat([
                robot_emb,           # [batch * (robot_count), 4]
                env_per_robot,       # [batch * (robot_count), 32]
                graph_feat_per_robot # [batch * (robot_count), 16]
            ], dim=1)  # [batch * (robot_count), 52]
            
            # GNN处理
            edge_index = self._create_edges(batch_size, robot_count, graph)
            x = gnn_input
            
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, edge_index)
                x = F.relu(x)

            # 获取跟随者节点的特征（假设领导者是第一个机器人）
            # 重塑为 [batch, robot_count, 64]
            x_reshaped = x.view(batch_size, robot_count, -1)
            
            # 只处理跟随者（索引1到robot_count-1）
            follower_features = x_reshaped[:, 1:, :]  # [batch, robot_count-1, 64]

            # 先处理图消息，在处理位置生成会不会好一点
            
            # 生成位置均值（获取相对位置）
            position_mean = self.position_generator(follower_features)  # [batch, robot_count-1, 2]

            # 创建位置分布
            position_log_std = self.position_log_std.expand_as(position_mean)
            position_dist = torch.distributions.Normal(   # 正态分布
                position_mean, 
                torch.exp(position_log_std)
            ) 
            
            position_dists.append(position_dist)

        # 计算状态价值（Critic）
        # 使用平均图特征作为全局图上下文
        mean_graph_feat = torch.mean(all_graph_features, dim=0, keepdim=True)  # [1, 16]
        mean_graph_feat = mean_graph_feat.expand(batch_size, -1)  # [batch, 16]
        
        value_input = torch.cat([env_features, mean_graph_feat], dim=-1)  # [batch, 48]
        value = self.critic(value_input)  # [batch, 1]
        
        return {
            'graph_scores': graph_scores,      # [batch, num_graphs] - 用于离散动作选择
            'position_dists': position_dists,  # 列表，长度num_graphs，每个元素是位置分布
            'value': value,                    # [batch, 1] - 状态价值
            'graph_features': all_graph_features  # [num_graphs, 16] - 用于后续计算
        }
    
    def _adj_matrix_to_edge_index(self, adj_matrix):
        """将邻接矩阵转换为edge_index格式

           把 0/1 矩阵变成 ‘source→target’ 两行坐标
        """
        edge_list = []
        n = adj_matrix.size(0)
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:  # 有连接
                    edge_list.append([i, j])
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    def _create_edges(self, batch_size, robot_count, adj_matrix):
        """根据邻接矩阵创建边索引，支持批量处理"""
        edge_list = []
        
        # 获取邻接矩阵中的边（非零元素）
        if isinstance(adj_matrix, torch.Tensor):
            rows, cols = torch.where(adj_matrix > 0)
        else:
            # 如果adj_matrix是numpy数组
            rows, cols = torch.where(torch.tensor(adj_matrix) > 0)
        
        # 为每个batch样本创建边
        for b in range(batch_size):
            offset = b * robot_count
            for i, j in zip(rows, cols):
                edge_list.append([offset + i.item(), offset + j.item()])
        
        if len(edge_list) == 0:
            # 如果没有边，返回空张量
            return torch.empty((2, 0), dtype=torch.long, device=adj_matrix.device if hasattr(adj_matrix, 'device') else 'cpu')
        
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    def _batch_edge_index(self, edge_index, num_nodes, batch_size):
        """为批次处理扩展边索引
        
           给每张图的边编号加上偏移，拼成一批大图，实现 GNN 的批量并行
        """
        edge_indices = []
        for b in range(batch_size):
            offset = b * num_nodes
            edges = edge_index + offset
            edge_indices.append(edges)
        return torch.cat(edge_indices, dim=1)

    def _polar_to_constrained_cartesian(self, polar_params):
        """
        将极坐标参数转换为带约束的直角坐标
        
        Args:
            polar_params: [batch_size, num_followers, 2]
                         [:, :, 0] 距离参数 (0-1)
                         [:, :, 1] 角度参数 (0-1)
        """
        # 应用sigmoid确保参数在0-1范围内
        distance_param = torch.sigmoid(polar_params[..., 0])
        angle_param = torch.sigmoid(polar_params[..., 1])
        
        # 转换为实际距离和角度
        distances = distance_param * (self.max_distance - self.min_distance) + self.min_distance
        angles = angle_param * (self.angle_max - self.angle_min) + self.angle_min
        
        # 转换为弧度
        angles_rad = angles * torch.pi / 180.0
        
        # 转换为直角坐标 (在领航者后方)
        x = -distances * torch.cos(angles_rad)  # 负号确保在后方
        y = distances * torch.sin(angles_rad)
        
        return torch.stack([x, y], dim=-1)
    
    def _cartesian_to_polar(self, cartesian_pos):
        """
        将直角坐标转换为极坐标参数
        并归一化到0-1范围内
        最后逆变换用于位置微调模块
        """
        x, y = cartesian_pos[..., 0], cartesian_pos[..., 1]
        
        # 计算距离和角度
        distances = torch.sqrt(x**2 + y**2)
        angles = torch.atan2(y, -x) * 180.0 / torch.pi  # 注意负号
        
        # 归一化到0-1范围
        distance_param = (distances - self.min_distance) / (self.max_distance - self.min_distance)
        angle_param = (angles - self.angle_min) / (self.angle_max - self.angle_min)
        
        # 应用sigmoid逆函数（近似）
        distance_param = torch.logit(distance_param.clamp(1e-6, 1-1e-6))
        angle_param = torch.logit(angle_param.clamp(1e-6, 1-1e-6))
        
        return torch.stack([distance_param, angle_param], dim=-1)

class SimplifiedPPOLoss(nn.Module):
    def __init__(self, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def forward(self, new_outputs, old_outputs, actions, advantages, returns):
        """
        计算PPO损失
        
        Args:
            new_outputs: 新策略的输出
            old_outputs: 旧策略的输出（结构相同）
            actions: 包含图索引和位置的动作
                graph_idx: [batch] 选择的图索引
                position: [batch, robot_count-1, 2] 选择的位置
            advantages: [batch] 优势函数（动作比当前状态的平均水平好多少）
            returns: [batch] 回报
        """
        batch_size = actions['graph_idx'].size(0)
        
        # 1. 离散动作损失（图选择）
        # 获取新旧策略的图选择概率
        new_graph_logits = new_outputs['graph_scores']
        old_graph_logits = old_outputs['graph_scores']
        
        # 转换为概率分布
        new_graph_probs = F.softmax(new_graph_logits, dim=-1)
        old_graph_probs = F.softmax(old_graph_logits, dim=-1)
        
        # 获取选择动作的概率（从概率表中精准找出对应图的概率值）
        new_graph_log_probs = torch.log(new_graph_probs.gather(1, actions['graph_idx'].unsqueeze(1))).squeeze(1) #新策略下，“选中该图” 的对数概率
        old_graph_log_probs = torch.log(old_graph_probs.gather(1, actions['graph_idx'].unsqueeze(1))).squeeze(1) #旧策略下，“选中该图” 的对数概率
        
        # 计算比率和PPO clip损失
        ratio_graph = torch.exp(new_graph_log_probs - old_graph_log_probs)
        graph_surr1 = ratio_graph * advantages
        graph_surr2 = torch.clamp(ratio_graph, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        graph_loss = -torch.min(graph_surr1, graph_surr2).mean()
        
        # 2. 连续动作损失（位置生成）
        position_loss = 0
        ratio_position_list = []
        
        # 对每个样本单独处理
        for i in range(batch_size):
            graph_idx = actions['graph_idx'][i].item()
            
            # 获取该图对应的位置分布
            new_position_dist = new_outputs['position_dists'][graph_idx]
            old_position_dist = old_outputs['position_dists'][graph_idx]
            
            # 获取该样本的位置动作
            sample_position = actions['position'][i:i+1]  # [1, robot_count-1, 2]
            
            # 计算新旧策略下该位置的对数概率
            new_position_log_prob = new_position_dist.log_prob(sample_position).sum(dim=[1, 2])
            old_position_log_prob = old_position_dist.log_prob(sample_position).sum(dim=[1, 2])
            
            # 计算比率
            ratio_position = torch.exp(new_position_log_prob - old_position_log_prob)
            ratio_position_list.append(ratio_position)
            
            # PPO clip损失
            position_surr1 = ratio_position * advantages[i]
            position_surr2 = torch.clamp(ratio_position, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages[i]
            position_loss = position_loss - torch.min(position_surr1, position_surr2)
        
        position_loss = position_loss / batch_size
        
        # 3. 价值损失
        value = new_outputs['value'].squeeze(-1)
        value_loss = F.mse_loss(value, returns)
        
        # 4. 熵正则化
        # 图选择的熵（鼓励探索）
        graph_entropy = -(new_graph_probs * torch.log(new_graph_probs + 1e-8)).sum(dim=-1).mean()
        
        # 位置生成的熵（所有位置分布的平均熵）
        position_entropy = 0
        for dist in new_outputs['position_dists']:
            position_entropy += dist.entropy().mean()
        position_entropy = position_entropy / len(new_outputs['position_dists'])
        
        entropy_bonus = graph_entropy + position_entropy
        
        # 5. 总损失
        total_loss = (
            graph_loss + 
            position_loss + 
            self.value_coef * value_loss - 
            self.entropy_coef * entropy_bonus
        )
        
        # 记录各项损失
        losses_dict = {
            'total_loss': total_loss.item(),
            'graph_loss': graph_loss.item(),
            'position_loss': position_loss.item(),
            'value_loss': value_loss.item(),
            'graph_entropy': graph_entropy.item(),
            'position_entropy': position_entropy.item(),
            'graph_ratio_mean': ratio_graph.mean().item(),
            'position_ratio_mean': torch.mean(torch.stack(ratio_position_list)).item() if ratio_position_list else 0
        }
        
        return total_loss, losses_dict

    
    def _compute_diversity_loss(self, positions):
        """计算位置配置的多样性损失"""
        batch_size, num_graphs, num_robots, _ = positions.shape
        
        if num_graphs < 2:
            return torch.tensor(0.0, device=positions.device)
        
        total_distance = 0.0
        count = 0
        
        # 计算所有控制图对之间的平均距离
        for i in range(num_graphs):
            for j in range(i + 1, num_graphs):
                pos_i = positions[:, i]  # [batch_size, num_robots, 2]
                pos_j = positions[:, j]
                
                # 计算平均位置差异
                diff = pos_i - pos_j
                distance = torch.mean(torch.norm(diff, dim=2))
                total_distance += distance
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=positions.device)
        
        avg_distance = total_distance / count
        
        # 我们希望平均距离不要太接近0（即不要所有控制图都生成相似的位置）
        # 因为模型的优化逻辑是 “最小化总损失”，所以损失函数必须满足：avg_distance 与损失值呈 “负相关”
        diversity_loss = -torch.log(avg_distance + 1e-8)
        
        return diversity_loss


那是不是我这些都需要改？