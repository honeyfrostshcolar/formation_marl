import numpy as np

class MADDPGFormationReward:
    """
    专为 MADDPG + G2ANet 打造的物理结果导向奖励函数
    彻底抛弃复杂的拓扑图约束，完全基于物理指标进行团队打分 (Team Reward)
    """
    def __init__(self, safety_threshold=0.6, target_distance=1.5, max_distance=4.0):
        # 物理边界定义
        self.safety_threshold = safety_threshold # 互碰红线 (米)
        self.target_distance = target_distance   # 跟随老大的完美距离 (米)
        self.max_distance = max_distance         # 掉队红线 (米)
        
        # 核心价值观权重分配 (针对消防编队任务)
        self.w_cohesion = 2.0   # 凝聚力：跟紧老大的得分权重
        self.w_separation = 5.0 # 分离度：无差别防互撞的惩罚权重 (底线，给很高)
        self.w_danger = 5.0     # 危险区：撞墙/进入障碍物膨胀层的重罚权重 (保命)
        self.w_jitter = 1.0     # 平滑度：过度抖动/能量损耗的惩罚权重
        self.w_direction = 0.5  # 方向感：保持在老大后方的得分权重

    def compute(self, leader_pos, follower_positions, prev_follower_positions, leader_yaw, in_danger_zone):
        """
        计算单步的全局团队奖励
        注意：传入的 in_danger_zone 包含了 [老大, 小弟1, 小弟2...]
        """
        num_followers = len(follower_positions)
        team_reward = 0.0
        
        # 记录每项得分的明细，方便你在 TensorBoard 或控制台打印排错
        reward_details = {
            'cohesion': 0.0,
            'separation': 0.0,
            'direction': 0.0,
            'danger': 0.0,
            'jitter': 0.0
        }

        # 预计算老大的车头朝向向量
        forward_vec = np.array([np.cos(leader_yaw), np.sin(leader_yaw)])

        # ==========================================
        # 1. 凝聚力 (Cohesion) & 方向感 (Direction)
        # ==========================================
        for i in range(num_followers):
            rel_pos = follower_positions[i] - leader_pos
            dist_to_leader = np.linalg.norm(rel_pos)
            
            # A. 距离奖励：离 target_distance(1.5) 越近，得分越高；超过 max_distance 重罚
            if dist_to_leader > self.max_distance:
                cohesion_score = -2.0 # 掉队重罚
            else:
                # 倒U型奖励：距离正好是 1.5 米时拿到满分 1.0
                cohesion_score = 1.0 - abs(dist_to_leader - self.target_distance)
            
            team_reward += self.w_cohesion * cohesion_score
            reward_details['cohesion'] += cohesion_score

            # B. 方向奖励：鼓励小弟乖乖待在老大后方 (即局部 X 坐标为负)
            local_x = np.dot(rel_pos, forward_vec)
            if local_x < 0:
                direction_score = 0.5  # 在后方，加分
            else:
                direction_score = -1.0 # 冲到老大前面挡路，扣分！
                
            team_reward += self.w_direction * direction_score
            reward_details['direction'] += direction_score

        # ==========================================
        # 2. 分离度 (Separation) - 铁血防互撞
        # ==========================================
        # 无视任何通信图，只要两个人物理距离太近，直接扣分！
        for i in range(num_followers):
            for j in range(i + 1, num_followers):
                dist_ij = np.linalg.norm(follower_positions[i] - follower_positions[j])
                
                # 如果距离小于安全阈值，按侵入深度进行线性重罚
                if dist_ij < self.safety_threshold:
                    separation_score = - (self.safety_threshold - dist_ij) * 5.0
                    team_reward += self.w_separation * separation_score
                    reward_details['separation'] += separation_score

        # ==========================================
        # 3. 危险区惩罚 (Danger Avoidance) - 防撞墙
        # ==========================================
        # in_danger_zone[0] 是老大，我们只算小弟的撞墙情况 (索引 1 开始)
        for is_danger in in_danger_zone[1:]: 
            if is_danger:
                danger_score = -1.0
                team_reward += self.w_danger * danger_score
                reward_details['danger'] += danger_score

        # ==========================================
        # 4. 抖动惩罚 (Jitter / Energy Penalty)
        # ==========================================
        # 计算每个小弟相比上一帧移动了多少距离
        movement = np.linalg.norm(follower_positions - prev_follower_positions, axis=1)
        # 移动量过大意味着动作突变抽搐，取平均值作为惩罚
        jitter_score = -np.mean(movement)
        
        team_reward += self.w_jitter * jitter_score
        reward_details['jitter'] += jitter_score

        # ==========================================
        # 5. 团队总分归一化
        # ==========================================
        # 为了防止小弟数量变多导致总分绝对值爆炸，除以人数算平均分
        normalized_team_reward = team_reward / num_followers

        return float(normalized_team_reward), reward_details