import numpy as np


class MADDPGFormationReward:
    """
    专为 MADDPG + G2ANet 打造的物理结果导向奖励函数 (融合版)
    融合了动态拓扑距离约束与硬性物理防撞/防抖底线。
    """
    def __init__(self, safety_threshold=0.5, target_distance=1.5, max_distance=4.0):
        # 物理边界定义
        self.safety_threshold = safety_threshold # 互碰红线 (米)
        self.target_distance = target_distance   # 目标连线距离 (米)
        self.max_distance = max_distance         # 掉队红线 (米)
        
        # ✅ 核心价值观权重分配 (全部补齐了！)
        self.w_link = 1.0      # 连线奖励：连了谁，就跟谁保持 target_distance
        self.w_iso = 5.0    # 孤立惩罚：如果谁也不连，当场重罚
        self.w_sparsity = 0.2   # 稀疏惩罚：每多连一条线扣一点分，逼迫图精简
        
        self.w_separation = 0.5  # 分离度：无差别防互撞的惩罚权重 (底线，给很高)
        self.w_danger = 2.0      # 危险区：撞墙/进入障碍物膨胀层的重罚权重 (保命)
        self.w_jitter = 1.0      # 平滑度：过度抖动/能量损耗的惩罚权重
        self.w_direction = 8.0   # 方向感：保持在老大后方的得分权重
        self.w_geometry = 1.0    # 几何形状：鼓励形成良好的队形奖励权重 (新加的)

    
    def compute(self, leader_pos, follower_positions, prev_follower_positions, leader_yaw, in_danger_zone, in_crash_zone, dynamic_connections, corridor_width):
        num_followers = len(follower_positions)
        team_reward = 0.0
        all_positions = [leader_pos] + list(follower_positions)
        forward_vec = np.array([np.cos(leader_yaw), np.sin(leader_yaw)])

        reward_details = {
            'link_score': 0.0,
            'distance': 0.0,
            'separation': 0.0,
            'direction': 0.0,
            'danger': 0.0,
            'geometry': 0.0
        }

        # ==========================================
        # 1. 拓扑连线距离评估 
        # ==========================================
        # 每个小弟只考察自己与领航者，以及与自己最近的一个小弟的距离
        for i in range(num_followers):
            my_pos = follower_positions[i]
            
            # A. 方向奖励：鼓励待在老大后方
            rel_pos = my_pos - leader_pos
            local_x = np.dot(rel_pos, forward_vec)

            direction_penalty = max(local_x, 0.0)   # 只罚跑前面
            team_reward -= self.w_direction * direction_penalty

            reward_details['direction'] -= (self.w_direction * direction_penalty)

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
        # 4. 几何队形奖励 (虚拟结构法 / 槽位对齐)
        # ==========================================
        geometry_score = 0.0
        
        # 定义 V 字形的展开角度，例如 45度 (pi/4) 或是 30度 (pi/6)
        v_angle = np.pi / 4 
        
        for i, pos in enumerate(follower_positions):
            global_i = i + 1 # 小弟的真实编号：1, 2, 3, 4...
            
            # 将当前小弟的全局坐标转为相对于老大的局部坐标
            rel_pos = pos - leader_pos
            local_x = np.dot(rel_pos, forward_vec)
            local_y = np.dot(rel_pos, np.array([-forward_vec[1], forward_vec[0]]))
            
            target_local_x = 0.0
            target_local_y = 0.0
            
            # 模式1：狭窄道 -> 一字长蛇阵 (1-shape)
            if corridor_width < 3.0: 
                # 都在 Y=0 的中心线上，X 轴向后排开
                target_local_x = - global_i * self.target_distance
                target_local_y = 0.0
                
            # 模式2：开阔地 -> 大雁 V 字形 (V-shape)
            elif corridor_width > 4.5: 
                # 规律：奇数号在左边 (1, 3, 5...)，偶数号在右边 (2, 4, 6...)
                # 行数计算：1,2号在第1排；3,4号在第2排
                row = (global_i + 1) // 2 
                
                # 方向：左侧 Y 为正，右侧 Y 为负
                side_multiplier = 1.0 if global_i % 2 != 0 else -1.0 
                
                # 利用三角函数算出目标槽位的精确坐标
                target_local_x = - row * self.target_distance * np.cos(v_angle)
                target_local_y = side_multiplier * row * self.target_distance * np.sin(v_angle)
            
            else:
                # 介于 3.0 和 4.5 之间的过渡区，可以暂时不给强硬的几何惩罚，让其自然过渡
                # 或者你也可以在这里写一个线性插值，让 V 字形慢慢向 1 字形收缩
                continue 

            # 计算当前真实位置与“理想虚拟槽位”的欧氏距离偏差
            slot_error = np.sqrt((local_x - target_local_x)**2 + (local_y - target_local_y)**2)
            
            # 偏差越大，扣分越狠 (乘以一个权重，比如 2.0)
            geometry_score -= slot_error  

        team_reward += self.w_geometry * geometry_score
        reward_details['geometry'] += (self.w_geometry * geometry_score)

        # ==========================================
        # 5. 撞墙终结惩罚 (修复梯度悬崖)
        # ==========================================
        any_crash = any(in_crash_zone)
        if any_crash:
            # 不要用 -200，用 -20 足够让它明白这是致命错误，同时不炸毁梯度
            team_reward -= 5.0 
            reward_details['danger'] -= 5.0
        else:
            team_reward += 0.2 # 存活奖励
        
        # print(f"Reward Details: {reward_details}")

        normalized_team_reward = team_reward / num_followers
        return float(normalized_team_reward), reward_details