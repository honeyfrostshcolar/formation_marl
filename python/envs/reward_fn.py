# envs/reward_fn.py
import numpy as np
import torch
import torch.nn.functional as F

class FormationReward:
    """编队任务奖励函数"""
    
    def __init__(self, safety_threshold=0.5, max_comm_distance=10.0, 
                 formation_weight=0.4, safety_weight=0.3, comm_weight=0.3):
        self.safety_threshold = safety_threshold  # 最小安全距离
        self.max_comm_distance = max_comm_distance  # 最大通信距离
        self.formation_weight = formation_weight  # 编队质量权重
        self.safety_weight = safety_weight  # 安全权重
        self.comm_weight = comm_weight  # 通信质量权重
    
    def compute(self, positions, graph):
        # 计算各分量（可能返回张量或 numpy 标量）
        formation_reward = self._formation_quality_reward(graph, positions) 
        safety_penalty = self._safety_penalty(positions)
        comm_reward = self._communication_reward(graph, positions)
        task_reward = self._task_reward(positions)

        # 加权总奖励（此时可能仍是张量）
        # total_reward = (
        #     self.formation_weight * formation_reward +
        #     self.safety_weight * (-safety_penalty) +
        #     self.comm_weight * comm_reward 
        #     # task_reward
        # )

        # print(f"formation_reward: {formation_reward:.3f}, safety_penalty: {safety_penalty:.3f}, comm_reward: {comm_reward:.3f}")
        total_reward = (
            formation_reward 
        )

        # 转换为 Python 标量
        total_reward = float(total_reward)

        # 构建奖励组件字典，确保所有值都是 Python float
        # reward_components = {
        #     'formation': float(formation_reward),
        #     'safety': float(-safety_penalty),
        #     'communication': float(comm_reward)
        #     # 'task': float(task_reward)
        # }
        reward_components = {
            'formation': float(formation_reward),
        }

        # print(reward_components)

        return total_reward, reward_components
    
    def _formation_quality_reward(self, graph, positions):
        """编队质量奖励"""

        # print("positions:", positions)

        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        num_robots = positions.shape[0]
        reward = 0.0
        
        # 检查图连接
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                if graph[i, j] > 0:
                    rel = positions[j, :2] - positions[i, :2]
                    distance = torch.norm(positions[i, :2] - positions[j, :2])
                    # print("distance:", distance)
                    # reward += torch.exp(-error)
                    # reward += -torch.abs(distance - ideal_distance)
                    if distance < 0.5:
                        penalty = -2.0 * (0.5 - distance)   # 越近越惩罚
                    elif distance > 1.5:
                        penalty = -2.0 * (distance - 1.5)   # 越远越惩罚
                    else:
                        penalty = 0.0

                    if rel[1] < 0:
                        direction_bonus = 1.0 - 0.5 * abs(rel[1] + 1.0)   # 在后方时，越接近-1越好
                    else:
                        direction_bonus = -1.0   # 在前方则惩罚

                    reward += penalty + direction_bonus

        return reward
    
    def _safety_penalty(self, positions):
        """安全惩罚"""
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        num_robots = positions.shape[0]
        penalty = 0.0
        
        # 机器人间碰撞检测
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                distance = torch.norm(positions[i, :2] - positions[j, :2])
                # print("distance:", distance)
                if distance < self.safety_threshold:
                    penalty += (self.safety_threshold - distance) * 2.0 # safety_threshold = 0.5

        # print("penalty:", penalty)
        
        return penalty
    
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
                    distance = torch.norm(positions[i, :2] - positions[j, :2])
                    if distance <= self.max_comm_distance:  # self.max_comm_distance = 5.0
                        # 在通信范围内，距离越近通信质量越好
                        reward += 1.0 - (distance / self.max_comm_distance)
                    else:
                        # 超出通信范围，惩罚
                        reward -= 0.0
                    connections += 1
        
        if connections > 0:
            reward /= connections
        
        return reward
    
    def _task_reward(self, positions):
        """任务特定奖励，根据具体任务定义"""
        # 示例：如果接近目标区域，给予奖励
        target_area = np.array([5.0, 5.0])  # 目标区域
        center = positions.mean(axis=0)[:2]
        distance_to_target = np.linalg.norm(center - target_area)
        return np.exp(-distance_to_target)