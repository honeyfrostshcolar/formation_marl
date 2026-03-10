import numpy as np
import torch

class FormationReward:
    """复杂消防救援场景下的多机器人编队奖励函数"""
    def __init__(self, safety_threshold=0.5, max_comm_distance=5.0):
        self.safety_threshold = safety_threshold
        self.max_comm_distance = max_comm_distance
        
        # 奖励/惩罚权重系数 (可根据训练情况微调)
        self.w_form = 1.0       # 编队保持权重
        self.w_spread = 0.5     # 横向展开权重
        self.w_danger = 2.0     # 危险区(膨胀层)惩罚权重
        self.w_jitter = 3.0     # 阵型抖动(能量损耗)惩罚权重
        self.w_switch = 1.0     # 拓扑图切换惩罚权重

    def compute(self, positions, prev_positions, graph, is_graph_changed, leader_yaw, in_danger_zone):
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        if isinstance(prev_positions, np.ndarray):
            prev_positions = torch.from_numpy(prev_positions).float()
            
        # 1. 编队质量与通信奖励
        form_reward, spread_reward = self._formation_and_spread_reward(graph, positions, leader_yaw)
        
        # 2. 危险区惩罚 (Proactive Repulsion)
        danger_penalty = self._danger_zone_penalty(in_danger_zone)
        
        # 3. 能量损耗与抖动惩罚 (Jitter Penalty)
        jitter_penalty = self._jitter_penalty(positions, prev_positions)
        
        # 4. 拓扑图稳定性惩罚 (Topology Stability)
        switch_penalty = self.w_switch if is_graph_changed else 0.0
        
        # 总奖励计算
        total_reward = (
            self.w_form * form_reward +
            self.w_spread * spread_reward -
            self.w_danger * danger_penalty -
            self.w_jitter * jitter_penalty -
            switch_penalty
        )
        
        reward_components = {
            'formation': float(form_reward),
            'spread': float(spread_reward),
            'danger_penalty': float(-danger_penalty),
            'jitter_penalty': float(-jitter_penalty),
            'switch_penalty': float(-switch_penalty)
        }
        
        return float(total_reward), reward_components

    def _formation_and_spread_reward(self, graph, positions, leader_yaw):
        num_robots = positions.shape[0]
        form_reward = 0.0
        
        # 车头方向 (X轴) 和 横向方向 (Y轴)
        forward_vec = torch.tensor([np.cos(leader_yaw), np.sin(leader_yaw)], dtype=torch.float32)
        right_vec = torch.tensor([np.sin(leader_yaw), -np.cos(leader_yaw)], dtype=torch.float32)
        
        local_y_list = [] # 用于计算横向展开度
        
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                if graph[i, j] > 0: # 只有在拓扑图中有连线的机器人之间才计算保持奖励
                    distance = torch.norm(positions[i, :2] - positions[j, :2])
                    
                    # 距离惩罚与奖励 (最佳通信与安全距离为 1.0米)
                    if distance < self.safety_threshold:
                        form_reward -= (self.safety_threshold - distance) * 2.0
                    elif distance > self.max_comm_distance:
                        form_reward -= 2.0 # 断联重罚
                    else:
                        form_reward += 1.0 - abs(distance - 1.0)
                    
                    # 方向判定：要求跟随者(j)必须在领航者(i=0)的后方
                    if i == 0:
                        rel = positions[j, :2] - positions[i, :2]
                        local_x = torch.dot(rel, forward_vec)
                        local_y = torch.dot(rel, right_vec)
                        local_y_list.append(local_y)
                        
                        if local_x < 0:  # 局部 X 为负，在车尾后方
                            form_reward += 1.0 - 0.2 * abs(local_x)
                        else:
                            form_reward -= 1.0 # 跑到车前，严厉惩罚！

        # 横向展开奖励 (计算跟随者在 Y 轴上的标准差，越大说明散得越开，视野越好)
        spread_reward = 0.0
        if len(local_y_list) > 1:
            local_y_tensor = torch.stack(local_y_list)
            spread_reward = torch.std(local_y_tensor).item()
            # 限制最大展开奖励，防止为了拿分跑得太散
            spread_reward = min(spread_reward, 2.0)
            
        return form_reward, spread_reward

    def _danger_zone_penalty(self, in_danger_zone):
        """进入膨胀缓冲区的跟随者将被惩罚 (主动避障)"""
        penalty = 0.0
        # 我们只惩罚跟随者 (索引 1 开始)，领航者由导航算法保证安全
        for is_danger in in_danger_zone[1:]:
            if is_danger:
                penalty += 1.0 
        return penalty

    def _jitter_penalty(self, positions, prev_positions):
        """计算阵型抖动（跟随者相对领航者位置的剧烈变化）"""
        # 计算当前的相对位置
        curr_rel_pos = positions[1:, :2] - positions[0, :2]
        # 计算上一帧的相对位置
        prev_rel_pos = prev_positions[1:, :2] - prev_positions[0, :2]
        
        # 计算相对位置的变化量 (物理移动距离)
        movement = torch.norm(curr_rel_pos - prev_rel_pos, dim=1)
        # 取平均移动量作为抖动惩罚
        return movement.mean().item()