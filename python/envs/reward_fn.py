# envs/reward_fn.py
import numpy as np
import torch
import torch.nn.functional as F

class FormationReward:
    """编队任务奖励函数"""
    
    def __init__(self, safety_threshold=0.5, max_comm_distance=5.0, 
                 formation_weight=0.4, safety_weight=0.3, comm_weight=0.3):
        self.safety_threshold = safety_threshold  # 最小安全距离
        self.max_comm_distance = max_comm_distance  # 最大通信距离
        self.formation_weight = formation_weight  # 编队质量权重
        self.safety_weight = safety_weight  # 安全权重
        self.comm_weight = comm_weight  # 通信质量权重
    
    def compute(self, robot_id, positions, graph):
        """计算总奖励"""
        reward_components = {}
        
        # 1. 编队质量奖励
        formation_reward = self._formation_quality_reward(graph, positions)
        reward_components['formation'] = formation_reward
        
        # 2. 安全奖励
        safety_penalty = self._safety_penalty(positions)
        reward_components['safety'] = -safety_penalty
        
        # 3. 通信质量奖励
        comm_reward = self._communication_reward(graph, positions)
        reward_components['communication'] = comm_reward
        
        # 4. 任务特定奖励
        task_reward = self._task_reward(positions)
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
                    ideal_distance = 1.0  # 理想距离
                    error = torch.abs(distance - ideal_distance)
                    reward += torch.exp(-error)
        
        return reward.item()
    
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
        
        return penalty.item()
    
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
        
        return reward.item()
    
    def _task_reward(self, positions):
        """任务特定奖励，根据具体任务定义"""
        # 示例：如果接近目标区域，给予奖励
        target_area = np.array([5.0, 5.0])  # 目标区域
        center = positions.mean(axis=0)
        distance_to_target = np.linalg.norm(center - target_area)
        return np.exp(-distance_to_target)