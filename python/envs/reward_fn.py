import numpy as np


class MADDPGFormationReward:
    """
    专为 MADDPG + G2ANet 打造的物理结果导向奖励函数 (融合版)
    融合了动态拓扑距离约束与硬性物理防撞/防抖底线。
    """
    def __init__(self, safety_threshold=0.5, target_distance=1.0, max_distance=4.0):
        # 物理边界定义
        self.safety_threshold = safety_threshold # 互碰红线 (米)
        self.target_distance = target_distance   # 目标连线距离 (米)
        self.max_distance = max_distance         # 掉队红线 (米)
        self.jitter_threshold = 0.20  # 恶意抖动阈值

        self.narrow_width = 3.0    # 小于此宽度，完全变成1字
        self.wide_width = 6.0      # 大于此宽度，完全变成V字
        self.lon_tolerance = 0.1   # 纵向(前后)允许的误差死区，给避障留出弹性空间
        self.danger_geom_scale = 0.1 # 在危险区时，几何奖励的降权系数 (保命优先)
        
        # ✅ 核心价值观权重分配 (全部补齐了！)
        self.w_link = 1.0      # 连线奖励：连了谁，就跟谁保持 target_distance
        self.w_iso = 5.0    # 孤立惩罚：如果谁也不连，当场重罚
        self.w_sparsity = 0.2   # 稀疏惩罚：每多连一条线扣一点分，逼迫图精简
        
        self.w_separation = 1.0  # 分离度：无差别防互撞的惩罚权重 (底线，给很高)
        self.w_danger = 2.0      # 危险区：撞墙/进入障碍物膨胀层的重罚权重 (保命)
        self.w_jitter = 1.0      # 平滑度：过度抖动/能量损耗的惩罚权重
        self.w_direction = 1.0   # 方向感：保持在老大后方的得分权重
        self.w_geometry = 2.0    # 几何形状：鼓励形成良好的队形奖励权重 (新加的)

    def compute_formation_alpha(self, corridor_width: float) -> float:
        """
        根据走廊宽度计算当前期望队形的混合系数：
        alpha = 0 -> line
        alpha = 1 -> V
        """
        # print(f"corridor_width: {corridor_width}")
        alpha = np.clip(
            (corridor_width - self.narrow_width) / (self.wide_width - self.narrow_width),
            0.0,
            1.0,
        )
        return float(alpha)

    
    def compute(self, leader_pos, follower_positions, prev_follower_positions, leader_yaw, follower_yaw, in_danger_zone, in_crash_zone, dynamic_connections, corridor_width):
        num_followers = len(follower_positions)
        team_reward = 0.0
        all_positions = [leader_pos] + list(follower_positions)
        forward_vec = np.array([np.cos(leader_yaw), np.sin(leader_yaw)])

        reward_details = {
            'link_score': 0.0,
            'jitter': 0.0,
            'distance': 0.0,
            'separation': 0.0,
            'direction': 0.0,
            'danger': 0.0,
            'geometry': 0.0,
            "formation_alpha": 0.0,
        }

        # ==========================================
        # 1. 拓扑连线距离评估 
        # ==========================================
        # 每个小弟只考察自己与领航者，以及与自己最近的一个小弟的距离
        left_vec = np.array([-forward_vec[1], forward_vec[0]]) #垂直与前进方向

        for i in range(num_followers):
            my_pos = follower_positions[i]

            rel_pos = my_pos - leader_pos
            local_x = np.dot(rel_pos, forward_vec)

            direction_penalty = max(local_x, 0.0)
            team_reward -= self.w_direction * direction_penalty
            reward_details['direction'] -= self.w_direction * direction_penalty

            my_yaw = follower_yaw[i]
            # 建立小弟自身的正前和正左方向向量
            my_forward_vec = np.array([np.cos(my_yaw), np.sin(my_yaw)])
            my_left_vec = np.array([-np.sin(my_yaw), np.cos(my_yaw)]) 

            delta_pos = follower_positions[i] - prev_follower_positions[i]
            
            # 将位移投影到小弟自己的车头方向上
            my_lon_move = abs(np.dot(delta_pos, my_forward_vec)) # 自己往前走了多少 (对应 ax)
            my_lat_move = abs(np.dot(delta_pos, my_left_vec))    # 自己横向漂了多少 (对应 ay)

            # 采用平方惩罚：对横向漂移施加毁灭性打击
            jitter_penalty = 100 * (my_lat_move ** 2)

            team_reward -= self.w_jitter * jitter_penalty
            reward_details['jitter'] -= self.w_jitter * jitter_penalty

            # global_i = i + 1  # 因为 leader_pos 在索引 0，所以小弟的全局索引是 i+1
            # targets_with_weights = dynamic_connections.get(global_i, []) # 格式类似： {1: [(2, 0.6), (3, 0.3), (4, 0.1)]}
            # # B. 孤立惩罚：如果这个小弟一个连接都没有，重罚它一笔
            # if len(targets_with_weights) == 0:
            #     team_reward -= self.w_iso
            #     reward_details['link_score'] -= self.w_iso

            # # C. 稀疏惩罚：每多连一条线扣一点分，逼迫图精简
            # num_links = len(targets_with_weights)
            # team_reward -= self.w_sparsity * num_links
            # reward_details['link_score'] -= self.w_sparsity * num_links

            # # D. 动态加权弹性距离评估
            # for target_id, weight in targets_with_weights: # 这里完美解包了刚才存入的元组
            #     dist_to_target = np.linalg.norm(all_positions[global_i] - all_positions[target_id])
                
            #     # 算绝对误差
            #     dist_error = abs(dist_to_target - self.target_distance)
                
            #     # 加权惩罚：权重越大，误差造成的扣分越疼
            #     link_score = np.exp(-dist_error) * weight
            #     team_reward += self.w_link * link_score
            #     reward_details['link_score'] += self.w_link * link_score


        # ==========================================
        # 2. 分离度 (Separation) - 带下界的平滑防互撞
        # ==========================================
        for i in range(num_followers + 1):
            for j in range(i + 1, num_followers + 1): 
                dist_ij = np.linalg.norm(all_positions[i] - all_positions[j])
                if dist_ij < self.safety_threshold:
                    
                    # ✅ 核心修改：设置一个物理极限的安全底线（比如 0.1 米，即 10 厘米）
                    # 就算它们完全重合 (dist_ij=0)，我们计算时也按 0.1 算，防止分母趋近于 0 导致数值爆炸
                    safe_dist = max(dist_ij, 0.1) 
                    
                    # 这样计算，惩罚最大也就是 -(0.5 / 0.1) = -5
                    # 乘上权重 10，最大惩罚是 -50。保留了梯度，又掐死了爆炸的可能
                    separation_score = - (self.safety_threshold / safe_dist)
                    
                    team_reward += self.w_separation * separation_score
                    reward_details['separation'] += (self.w_separation * separation_score)

        # ==========================================
        # 3. 危险区惩罚 
        # ==========================================
        for is_danger in in_danger_zone[1:]: 
            if is_danger:
                team_reward -= self.w_danger
                reward_details['danger'] -= self.w_danger

        # ==========================================
        # ✅ 修改 2：第4部分 彻底升级为【软切换 + 纵向容忍】架构
        # ==========================================
        geometry_score = 0.0
        v_angle = np.pi / 4 
        
        # 计算软切换系数 alpha (0 代表纯1字，1 代表纯V字)
        # narrow_width = 3.0   wide_width = 5.0
        # print(f"corridor_width: {corridor_width}")
        alpha = self.compute_formation_alpha(corridor_width)
        reward_details["formation_alpha"] = alpha
        # print("alpha", alpha)
        line_spacing = 0.9
        
        for i, pos in enumerate(follower_positions):
            global_i = i + 1 
            
            # 转局部坐标
            rel_pos = pos - leader_pos
            local_x = np.dot(rel_pos, forward_vec)
            local_y = np.dot(rel_pos, left_vec)
            
            # --- 计算目标1：一字长蛇阵 ---
            line_x = - global_i * line_spacing
            line_y = 0.0
            
            # --- 计算目标2：V字阵 ---
            row = (global_i + 1) // 2 
            side_multiplier = 1.0 if global_i % 2 != 0 else -1.0 
            v_x = - row * self.target_distance * np.cos(v_angle)
            v_y = side_multiplier * row * self.target_distance * np.sin(v_angle)

            # --- 软插值：融合两者的目标槽位 ---
            target_local_x = (1.0 - alpha) * line_x + alpha * v_x
            target_local_y = (1.0 - alpha) * line_y + alpha * v_y

            # --- 误差计算（横向与纵向解耦）---
            delta_x = abs(local_x - target_local_x)
            delta_y = abs(local_y - target_local_y)

            e = (delta_x) ** 2 + (delta_y) ** 2

            # 纵向给予一定容忍度（允许小车前后减速避障，只要不超标就不算偏离）
            # lon_error = max(0.0, delta_x - self.lon_tolerance) # lon_tolerance = 0.5
            # lat_error = delta_y # 横向零容忍，防撞墙

            # # 合成有效误差
            # effective_error = np.sqrt(lon_error**2 + lat_error**2)
            
            # --- 危险区降权判定 ---
            is_scared = in_danger_zone[global_i]
            # 如果处于危险区，编队分数大幅缩水，逼迫网络专注规避 danger 惩罚
            if is_scared and alpha > 0.5:
                geometry_scale = self.danger_geom_scale
            else:
                geometry_scale = 1.0

            # 最终几何得分
            geometry_score += e * geometry_scale

        team_reward -= self.w_geometry * geometry_score
        reward_details['geometry'] -= (self.w_geometry * geometry_score)

        # ==========================================
        # 5. 撞墙终结惩罚 (修复梯度悬崖)
        # ==========================================
        any_crash = any(in_crash_zone)
        if any_crash:
            # 不要用 -200，用 -20 足够让它明白这是致命错误，同时不炸毁梯度
            team_reward -= 100.0 
            reward_details['danger'] -= 100.0
        else:
            team_reward += 2 # 存活奖励
        
        # print(f"Reward Details: {reward_details}")

        normalized_team_reward = team_reward / num_followers
        return float(normalized_team_reward), reward_details