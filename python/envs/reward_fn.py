import numpy as np

class MADDPGFormationReward:
    """
    专为 MADDPG + G2ANet 打造的物理结果导向奖励函数 (融合版)
    融合了动态拓扑距离约束与硬性物理防撞/防抖底线。
    """
    def __init__(self, safety_threshold=0.6, target_distance=1.0, max_distance=4.0):
        # 物理边界定义
        self.safety_threshold = safety_threshold # 互碰红线 (米)
        self.target_distance = target_distance   # 目标连线距离 (米)
        self.max_distance = max_distance         # 掉队红线 (米)
        
        # ✅ 核心价值观权重分配 (全部补齐了！)
        self.w_link = 2.0       # 连线奖励：连了谁，就跟谁保持 target_distance
        self.w_iso = 3.0        # 孤立惩罚：如果谁也不连，当场重罚
        self.w_sparsity = 0.2   # 稀疏惩罚：每多连一条线扣一点分，逼迫图精简
        
        self.w_separation = 5.0 # 分离度：无差别防互撞的惩罚权重 (底线，给很高)
        self.w_danger = 5.0     # 危险区：撞墙/进入障碍物膨胀层的重罚权重 (保命)
        self.w_jitter = 1.0     # 平滑度：过度抖动/能量损耗的惩罚权重
        self.w_direction = 0.5  # 方向感：保持在老大后方的得分权重

    def compute(self, leader_pos, follower_positions, prev_follower_positions, leader_yaw, in_danger_zone, dynamic_connections):
        num_followers = len(follower_positions)
        team_reward = 0.0

        # 为了方便计算，把老大(0号)和小弟(1~N)的位置拼成一个统一的列表
        all_positions = [leader_pos] + list(follower_positions)
        
        # ✅ 更新了日志字典，方便 TensorBoard 追踪新指标
        reward_details = {
            'link_score': 0.0,
            'iso_penalty': 0.0,
            'separation': 0.0,
            'direction': 0.0,
            'danger': 0.0,
            'jitter': 0.0
        }

        forward_vec = np.array([np.cos(leader_yaw), np.sin(leader_yaw)])

        # ==========================================
        # 1. 动态拓扑相对距离约束 & 方向约束
        # ==========================================
        for i in range(num_followers):
            global_i = i + 1 
            targets = dynamic_connections.get(global_i, []) 
            rel_pos = follower_positions[i] - leader_pos 

            # A. 孤立惩罚
            if len(targets) == 0:
                team_reward -= self.w_iso
                reward_details['iso_penalty'] -= self.w_iso
                continue

            # B. 动态连线距离评估
            for target_id in targets:
                team_reward -= self.w_sparsity 
                dist_to_target = np.linalg.norm(all_positions[global_i] - all_positions[target_id])
                
                link_score = 1.0 - abs(dist_to_target - self.target_distance)
                team_reward += self.w_link * link_score
                reward_details['link_score'] += (self.w_link * link_score)

            # C. 方向奖励：鼓励小弟待在老大后方
            local_x = np.dot(rel_pos, forward_vec)
            if local_x < 0:
                direction_score = 0.5  
            else:
                direction_score = -1.0 
                
            team_reward += self.w_direction * direction_score
            reward_details['direction'] += (self.w_direction * direction_score)

        # ==========================================
        # 2. 分离度 (Separation) - 铁血防互撞
        # ==========================================
        for i in range(num_followers):
            for j in range(i + 1, num_followers):
                dist_ij = np.linalg.norm(follower_positions[i] - follower_positions[j])
                if dist_ij < self.safety_threshold:
                    separation_score = - (self.safety_threshold - dist_ij) * 5.0
                    team_reward += self.w_separation * separation_score
                    reward_details['separation'] += (self.w_separation * separation_score)

        # ==========================================
        # 3. 危险区惩罚 (Danger Avoidance)
        # ==========================================
        for is_danger in in_danger_zone[1:]: 
            if is_danger:
                danger_score = -1.0
                team_reward += self.w_danger * danger_score
                reward_details['danger'] += (self.w_danger * danger_score)

        # ==========================================
        # 4. 抖动惩罚 (Jitter Penalty)
        # ==========================================
        movement = np.linalg.norm(follower_positions - prev_follower_positions, axis=1)
        jitter_score = -np.mean(movement)
        
        team_reward += self.w_jitter * jitter_score
        reward_details['jitter'] += (self.w_jitter * jitter_score)

        normalized_team_reward = team_reward / num_followers

        return float(normalized_team_reward), reward_details