import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import formation_core 
from envs.reward_fn import MADDPGFormationReward
import os
from PIL import Image
from collections import deque
from envs.reward_fn import MADDPGFormationReward

# ✅ 1. 导入 RLlib 的多智能体环境基类
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# ✅ 2. 继承 MultiAgentEnv 而不是 gym.Env
class Formation2DMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config, args=None):
        super().__init__()
        self.args = args
        self.num_robots = config.get("num_robots", 5) # 默认 5 个机器人
        self.num_followers = self.num_robots - 1
        
        # 即使现在我们只有 4 个跟随者，为了适应未来的泛化，我们可以预留最大容量
        self.max_robots = config.get("max_robots", 10)
        self.max_followers = self.max_robots - 1
        
        self.max_steps = config.get("max_steps", 200)
        self.radar_rays = config.get("radar_rays", 180)
        self.safety_threshold = config.get("safety_threshold", 0.5)

        # ✅ 3. 定义多智能体的 ID 列表 (例如: ['follower_0', 'follower_1', ...])
        self._agent_ids = [f"follower_{i}" for i in range(self.num_followers)]

        self.map_resolution = 0.1
        self.map_origin = np.array([0.0, 0.0])

        self.available_map_modes = ["open", "star_map", "hybrid", "z_map", "custom"]

        self.custom_map_path = config.get(
            "custom_map_path",
            "/home/nankai/formation_test/maps/underground_garage.pgm"
        )

        # 1. 极其干净的“地图注册表 (Map Registry)”
        # 以后想加什么地图，直接在这里加一行，指向你的生成函数！
        self.map_registry = {
            "open": self._generate_open_map,
            "z_map": self._generate_z_map,
            "star_map": self._generate_star_map,
            "hybrid": self._generate_hybrid_map,
            "custom": lambda: self._load_custom_map(self.custom_map_path), 
        }

        self.map_mode = config.get("map_mode", "open")

        self.sensing_radius = config.get("sensing_radius", 1.5)  # 最大感知半径 (米)
        self.max_visible_teammates = 3  # 网络最多只管最近的 3 个兄弟
        
        self.safety_threshold = 0.5
        self.target_distance = 1.0
        self.max_distance = 4.0
        self.reward_fn = MADDPGFormationReward(self.safety_threshold, self.target_distance, self.max_distance)
        
        self.leader_velocity = [0.0, 0.1]  
        self.leader_yaw = 0.0
        self.leader_pos = np.array([0.0, 0.0], dtype=np.float32)

        self.follower_pos = np.random.uniform(-1, 1, size=(self.num_followers, 2)).astype(np.float32)
        self.follower_yaw = np.full(self.num_followers, self.leader_yaw, dtype=np.float32)

        # ==========================================
        # ✅ 4. 彻底重构动作与状态空间 (针对单个智能体)
        # ==========================================
        # 动作空间：极其干净！就只有 X 和 Y 方向的相对位置指令
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # lidar_obs(41) + leader_rel(2) + teammates_obs(9) + role_code(2) + formation_alpha(1) + leader_heading(3)
        obs_dim = 41 + 2 + (self.max_visible_teammates * 3) + 2 + 1 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.render_mode = config.get("render", False)
        if self.render_mode:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()

        self.step_count = 0

        self._rebuild_map()

    def set_map_mode(self, map_mode: str, custom_map_path: str = None):
        if map_mode not in self.available_map_modes:
            raise ValueError(f"Unknown map_mode={map_mode}, allowed={self.available_map_modes}")
        self.map_mode = map_mode
        if custom_map_path is not None:
            self.custom_map_path = custom_map_path
        self._rebuild_map()


    def _rebuild_map(self):
        # ✅ 2. 没有任何 if-else，直接从字典里调函数生成地图
        self.map_grid = self.map_registry[self.map_mode]()
        self.planning_safe_margin = 1.0
        self.inflated_map_grid = self._inflate_map(self.map_grid, self.planning_safe_margin)


    def _load_custom_map(self, map_path: str):
        if os.path.exists(map_path):
            print(f"✅ 加载自定义地图成功: {map_path}")
            img = Image.open(map_path).convert("L")
            grid = np.array(img) < 250
            return grid
        else:
            print(f"❌ 自定义地图不存在: {map_path}，回退到默认 open 地图")
            return self._generate_open_map()


    def _generate_open_map(self):
        grid = np.zeros((200, 200), dtype=bool)
        grid[0:5, :] = True
        grid[-5:, :] = True
        grid[:, 0:5] = True
        grid[:, -5:] = True
        return grid


    def _generate_z_map(self):
        grid = np.zeros((200, 200), dtype=bool)
        grid[0:5, :] = True
        grid[-5:, :] = True
        grid[:, 0:5] = True
        grid[:, -5:] = True

        grid[30:40, 0:120] = True
        grid[80:95, 80:200] =  True
        grid[120:130, 0:120] = True
        grid[160:170, 100:200] = True
        return grid
    
    def _generate_star_map(self):
        grid = np.zeros((200, 200), dtype=bool)
        grid[0:5, :] = True
        grid[-5:, :] = True
        grid[:, 0:5] = True
        grid[:, -5:] = True

        num_per_side = 4  # 每行/每列的点数
        x_coords = np.linspace(35, 165, num_per_side, dtype=int)
        y_coords = np.linspace(35, 165, num_per_side, dtype=int)
        for x in x_coords:
            for y in y_coords:
                grid[y-4:y+5, x-4:x+5] = True
        return grid
    
    def _generate_hybrid_map(self):
        grid = np.zeros((200, 200), dtype=bool)

        # 边界墙
        grid[0:5, :] = True
        grid[-5:, :] = True
        grid[:, 0:5] = True
        grid[:, -5:] = True

        # 右半边放规则柱阵
        x_coords = np.linspace(120, 170, 3, dtype=int)   # 右侧三列柱子
        y_coords = np.linspace(35, 165, 4, dtype=int)    # 四行柱子
        for x in x_coords:
            for y in y_coords:
                grid[y-4:y+5, x-4:x+5] = True

        return grid
    

    def _world_to_grid(self, x, y):
        height, width = self.map_grid.shape
        pixel_x = int(x / self.map_resolution + width / 2)
        pixel_y = int(y / self.map_resolution + height / 2)
        return pixel_x, pixel_y

    def _is_obstacle(self, x, y):
        px, py = self._world_to_grid(x, y)
        height, width = self.map_grid.shape
        if 0 <= px < width and 0 <= py < height:
            return self.map_grid[py, px]
        return True
    
    def get_communication_mask(self, positions, max_dist=10.0, drop_rate=0.1):
        """
        计算物理通信的 Mask 矩阵 (N x N)
        专门针对 follower 之间的通信网络。
        """
        num_agents = len(positions)
        mask = np.ones((num_agents, num_agents), dtype=np.float32)

        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue  # 自己和自己通信全通

                # -----------------------------------
                # 1. 距离衰减 (Distance-based Loss)
                # -----------------------------------
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist > max_dist:
                    mask[i, j] = 0.0
                    continue

                # -----------------------------------
                # 2. 随机丢包 (Stochastic Packet Drop)
                # -----------------------------------
                if np.random.rand() < drop_rate:
                    mask[i, j] = 0.0
                    continue

                # -----------------------------------
                # 3. 视距遮挡 (Line-of-Sight Blockage)
                # -----------------------------------
                # 传入两辆车的坐标，检查是否被墙遮挡
                if self._check_los_blockage(positions[i], positions[j]):
                    mask[i, j] = 0.0

        return mask
    
    def _check_los_blockage(self, pos1, pos2):
        """
        使用 Bresenham 算法检查两个连续坐标之间是否有障碍物遮挡
        """
        # ✅ 直接调用你环境里现成的坐标转换函数！
        x0, y0 = self._world_to_grid(pos1[0], pos1[1])
        x1, y1 = self._world_to_grid(pos2[0], pos2[1])

        # 射线追踪算法
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        # 使用你环境里最精准的未膨胀地图作为遮挡判定标准
        grid = self.map_grid 
        height, width = grid.shape
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                # 边界保护
                if 0 <= y < height and 0 <= x < width:
                    if grid[y, x]:  # True 代表这里是障碍物（墙），信号中断！
                        return True 
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if 0 <= y < height and 0 <= x < width:
                    if grid[y, x]:  
                        return True
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        # 检查终点
        if 0 <= y < height and 0 <= x < width:
            if grid[y, x]:
                return True
            
        return False
    
    def _compute_mean_formation_error(self):
        forward_vec = np.array([np.cos(self.leader_yaw), np.sin(self.leader_yaw)])
        left_vec = np.array([-forward_vec[1], forward_vec[0]])

        # 和 reward_fn 保持一致
        leader_lidar = self._simulate_radar(self.leader_pos, self.leader_yaw)
        feature_extractor = formation_core.FeatureExtractor(8)
        leader_features = feature_extractor.extract_features(leader_lidar)
        corridor_width = leader_features.corridor_width
        alpha = self.reward_fn.compute_formation_alpha(corridor_width)

        v_angle = np.pi / 4
        line_spacing = 0.9

        errs = []
        for i, pos in enumerate(self.follower_pos):
            global_i = i + 1
            rel_pos = pos - self.leader_pos
            local_x = np.dot(rel_pos, forward_vec)
            local_y = np.dot(rel_pos, left_vec)

            line_x = - global_i * line_spacing
            line_y = 0.0

            row = (global_i + 1) // 2
            side_multiplier = 1.0 if global_i % 2 != 0 else -1.0
            v_x = - row * self.reward_fn.target_distance * np.cos(v_angle)
            v_y = side_multiplier * row * self.reward_fn.target_distance * np.sin(v_angle)

            target_local_x = (1.0 - alpha) * line_x + alpha * v_x
            target_local_y = (1.0 - alpha) * line_y + alpha * v_y

            err = np.sqrt((local_x - target_local_x) ** 2 + (local_y - target_local_y) ** 2)
            errs.append(err)

        return float(np.mean(errs)) if errs else 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        while True:
            start_idx = self._get_random_free_point_with_clearance(2.5)
            # world_start = self._grid_to_world(*start_idx)
            # print("World start:", world_start)
            goal_idx = self._get_random_free_point()
            
            # start_pos = [-7.5,-8.3]
            # goal_pos = [7.5, -8.3]
            # start_idx = self._world_to_grid(*start_pos)
            # goal_idx = self._world_to_grid(*goal_pos)


            path_indices = self._plan_path(start_idx, goal_idx)
            if len(path_indices) > 10: 
                break
                
        self.path = [self._grid_to_world(px, py) for px, py in path_indices]
        self.leader_pos = self.path.pop(0)

        self.leader_yaw = 0.0
        if len(self.path) > 0:
            direction = self.path[0] - self.leader_pos
            self.leader_yaw = np.arctan2(direction[1], direction[0])

            self.prev_leader_yaw = self.leader_yaw
            self.leader_yaw_rate = 0.0

        self.follower_pos = np.zeros((self.num_followers, 2), dtype=np.float32)
        
        for i in range(self.num_followers):
            valid_spawn = False
            attempts = 0
            while not valid_spawn and attempts < 100:
                # 在老大周围 2 米的圆环内随机找个点
                angle = np.random.uniform(-np.pi, np.pi)
                radius = np.random.uniform(0.5, 2.0)
                candidate_pos = self.leader_pos + np.array([np.cos(angle), np.sin(angle)]) * radius
                
                # 1. 检查是否在墙里（使用膨胀地图保证绝对安全）
                px, py = self._world_to_grid(candidate_pos[0], candidate_pos[1])
                if 0 <= px < self.inflated_map_grid.shape[1] and 0 <= py < self.inflated_map_grid.shape[0]:
                    if not self.inflated_map_grid[py, px]:  # 如果不在障碍物膨胀区内
                        
                        # 2. 检查和其他已经生成的小弟是否重叠
                        too_close = False
                        for j in range(i):
                            dist = np.linalg.norm(candidate_pos - self.follower_pos[j])
                            if dist < self.safety_threshold * 1.5: # 保证比最小距离还要宽裕一点
                                too_close = True
                                break
                                
                        if not too_close:
                            self.follower_pos[i] = candidate_pos
                            valid_spawn = True
                            
                attempts += 1
                
            if not valid_spawn:
                # 极端情况：如果尝试了100次都没找到好位置（比如走廊太窄）
                # 就强行排在老大正后方的一条直线上
                fallback_direction = -np.array([np.cos(self.leader_yaw), np.sin(self.leader_yaw)])
                self.follower_pos[i] = self.leader_pos + fallback_direction * (i + 1) * 0.5
        
        self.follower_yaw[:] = self.leader_yaw
        self.prev_leader_yaw = self.leader_yaw
        self.leader_yaw_rate = 0.0

        self.step_count = 0
        self.prev_follower_pos = self.follower_pos.copy()

        # ✅ 5. 循环构建所有小弟的初始观测字典
        obs_dict = {}
        for i, agent_id in enumerate(self._agent_ids):
            obs_dict[agent_id] = self._get_single_follower_obs(i)

        if self.render_mode:
            self.render()

        return obs_dict, {}

    def step(self, action_dict, graph_dict, graphs_soft_dict):
        """
        ✅ 6. 核心步进函数改造，接收动作字典
        action_dict 格式: {"follower_0": [dx, dy], "follower_1": [dx, dy], ...}

        """
        # ==========================================
        # 1. 领航者按预设轨迹移动 (与原来一样)
        # ==========================================
        leader_done = False
        old_leader_yaw = self.leader_yaw
        if len(self.path) > 0:
            target_pos = self.path[0]
            direction = target_pos - self.leader_pos
            distance_to_target = np.linalg.norm(direction)
            move_step = 0.1
            if distance_to_target < move_step:
                self.leader_pos = target_pos
                self.path.pop(0)
            else:
                self.leader_pos += (direction / distance_to_target) * move_step
         
                target_yaw = np.arctan2(direction[1], direction[0])
                yaw_diff = (target_yaw - self.leader_yaw + np.pi) % (2 * np.pi) - np.pi
                self.leader_yaw += np.clip(yaw_diff, -0.15, 0.15)
        else:
            leader_done = True

        yaw_delta = (self.leader_yaw - old_leader_yaw + np.pi) % (2 * np.pi) - np.pi
        self.leader_yaw_rate = yaw_delta
        self.prev_leader_yaw = old_leader_yaw

        # ==========================================
        # 2. 跟随者根据动作生成下一个路点 (纯路径规划)
        # ==========================================
        for i, agent_id in enumerate(self._agent_ids):
            if agent_id not in action_dict:
                continue
                
            ax, ay = action_dict[agent_id] 
            ax = (ax + 1) / 2.0

            max_step_dist = 0.2 
            
            my_yaw = self.follower_yaw[i] # 获取小弟自己的朝向

            # 把局部的 ax(前后), ay(左右) 旋转为全局的 dx, dy
            dx = (ax * np.cos(my_yaw) - ay * np.sin(my_yaw)) * max_step_dist
            dy = (ax * np.sin(my_yaw) + ay * np.cos(my_yaw)) * max_step_dist
            
            new_pos = self.follower_pos[i] + np.array([dx, dy], dtype=np.float32)

            # 只有当移动距离足够大时，才更新朝向，防止原地抖动导致朝向乱转
            if np.linalg.norm([dx, dy]) > 0.001:
               
                target_yaw = np.arctan2(dy, dx)
                
                # 模拟底盘旋转的物理惯性 (限制单步最大转角，比如和 leader 保持一致的 0.15 弧度)
                yaw_diff = (target_yaw - my_yaw + np.pi) % (2 * np.pi) - np.pi
                max_yaw_rate = 0.15 
                
                # 平滑滤波：限制每步能扭动的最大角度
                self.follower_yaw[i] = my_yaw + np.clip(yaw_diff, -max_yaw_rate, max_yaw_rate)
                
                # 规范化到 [-pi, pi]
                self.follower_yaw[i] = (self.follower_yaw[i] + np.pi) % (2 * np.pi) - np.pi
                
            self.follower_pos[i] = new_pos

        self.step_count += 1
        is_timeout = self.step_count >= self.max_steps # 超时结束

        # 直接根据 N x N 的图矩阵翻译连线
        dynamic_connections = {}
        for i, agent_id in enumerate(self._agent_ids):
            global_follower_id = i + 1
            dynamic_connections[global_follower_id] = []
            
            if agent_id in graph_dict:
                hard_weights = graph_dict[agent_id] # 这是长度为 N 的数组
                soft_weights = np.array(graphs_soft_dict[agent_id]).flatten()
                
                # 遍历图输出的 N 个目标槽位，直接对应所有的 follower
                for j in range(self.num_followers):
                    if hard_weights[j] > 0 and j != i: # 如果建立了硬连接，且不是自己连自己
                        target_global_id = j + 1 # leader 是 0，follower 从 1 开始
                        current_soft_weight = soft_weights[j]
                        dynamic_connections[global_follower_id].append((target_global_id, current_soft_weight))

        positions_3d = np.zeros((self.num_robots, 3), dtype=np.float32)
        positions_3d[0, :2] = self.leader_pos
        positions_3d[1:, :2] = self.follower_pos

        in_danger_zone = []
        in_crash_zone = []
        for i in range(self.num_robots):
            pos = positions_3d[i, :2]
            px, py = self._world_to_grid(pos[0], pos[1])
            if 0 <= px < self.inflated_map_grid.shape[1] and 0 <= py < self.inflated_map_grid.shape[0]:
                in_danger_zone.append(bool(self.inflated_map_grid[py, px]))
                in_crash_zone.append(bool(self.map_grid[py, px]))
            else:
                in_danger_zone.append(True)
                in_crash_zone.append(True)

        # print("in_crash_zone:", in_crash_zone)

        leader_lidar = self._simulate_radar(self.leader_pos, self.leader_yaw)
        feature_extractor = formation_core.FeatureExtractor(8)
        leader_features = feature_extractor.extract_features(leader_lidar)
        current_corridor_width = leader_features.corridor_width

        # ✅ 4. 把翻译好的连线扔给裁判！
        team_reward, reward_details = self.reward_fn.compute(
            leader_pos=self.leader_pos,
            follower_positions=self.follower_pos,
            prev_follower_positions=self.prev_follower_pos,
            leader_yaw=self.leader_yaw,
            follower_yaw=self.follower_yaw,
            in_danger_zone=in_danger_zone, # 传入每个智能体是否在危险区的信息(指的是inflated_map_grid)
            in_crash_zone=in_crash_zone, # 传入每个智能体是否在撞击区的信息(指的是map_grid)
            dynamic_connections=dynamic_connections, # 传入动态图！
            corridor_width=current_corridor_width # 传入当前走廊宽度，方便奖励函数设计更复杂的几何形状奖励
        )

        # ==========================================
        # 3. 计算每个智能体的观测、奖励和结束状态
        # ==========================================
        obs_dict = {}
        reward_dict = {}
        terminated_dict = {}
        truncated_dict = {}
        info_dict = {}

        # 如果任何一个小弟撞墙了，或者老大到终点了，这局就结束
        any_follower_crashed = any(in_crash_zone[1:])
        # print("any_follower_crashed:", any_follower_crashed)
        episode_done = leader_done or any_follower_crashed
        if leader_done:
            team_reward += 50.0 # 成功到达终点奖励
        # episode_done = leader_done

        for i, agent_id in enumerate(self._agent_ids): #agenr_ids指的就是跟随者的ID列表
            obs_dict[agent_id] = self._get_single_follower_obs(i)
            # print(f"obs_dict[{agent_id}]:", obs_dict[agent_id])
            # 所有人吃大锅饭：拿到完全一样的团队总分！
            reward_dict[agent_id] = team_reward 
            terminated_dict[agent_id] = episode_done
            truncated_dict[agent_id] = is_timeout
            
            info_payload = dict(reward_details)
            info_payload["leader_done"] = bool(leader_done)
            info_payload["any_follower_crashed"] = bool(any_follower_crashed)
            info_payload["timeout"] = bool(is_timeout)
            info_payload["success"] = bool(leader_done and not any_follower_crashed)
            info_payload["collision"] = bool(any_follower_crashed)
            info_payload["episode_done"] = bool(episode_done)
            info_payload["formation_error"] = float(self._compute_mean_formation_error())

            info_dict[agent_id] = info_payload

        terminated_dict["__all__"] = episode_done
        truncated_dict["__all__"] = is_timeout

        # ✅ 物理断网模拟：提取所有小弟的当前坐标，计算 N x N 的通信可用矩阵
        # 设置最大通信距离为 8.0 米，基础丢包率为 10%
        physical_comm_mask = np.ones((len(self.follower_pos), len(self.follower_pos)), dtype=np.float32)

        if getattr(self.args, 'interruption_loss_packet', False):
            physical_comm_mask = self.get_communication_mask(
                self.follower_pos, 
                max_dist=8.0, 
                drop_rate=0.1
            )

        # 把物理网络状态存入 info_dict，让主训练循环能拿到
        for i, agent_id in enumerate(self._agent_ids): 
            # 你可以把它塞进 __all__ 里，或者直接让第一个 agent 携带
            info_dict[agent_id]["physical_comm_mask"] = physical_comm_mask

        self.prev_follower_pos = self.follower_pos.copy()

        if self.render_mode:
            self.render()

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def _get_single_follower_obs(self, follower_idx):
        my_pos = self.follower_pos[follower_idx]
        my_yaw = self.follower_yaw[follower_idx] 

        # 算雷达 (传入自己的朝向)
        lidar_data = self._simulate_radar(my_pos, my_yaw)
        
        ranges = np.array(lidar_data.ranges)

        front_rays = np.concatenate([ranges[175:180], ranges[0:6]])
        front_clearance = float(np.min(front_rays))
        left_clearance = float(np.min(ranges[40:51]))
        right_clearance = float(np.min(ranges[130:141]))
        corridor_width = left_clearance + right_clearance
        obstacle_density = float(np.sum(ranges < 3.0) / len(ranges))
        
        core_features = np.array([
            corridor_width, front_clearance, left_clearance, right_clearance, obstacle_density
        ], dtype=np.float32)


        num_downsample = 36
        sectors = np.array_split(ranges, num_downsample)
        downsampled_lidar = np.array([np.min(sec) for sec in sectors], dtype=np.float32)
        
        # 拼起来组成新的雷达观测 (5 + 36 = 41 维)
        lidar_obs = np.concatenate([core_features, downsampled_lidar])
        # lidar_obs = np.round(lidar_obs, 2)

        cos_y, sin_y = np.cos(my_yaw), np.sin(my_yaw) 
        def global_to_local(vec):
            return np.array([
                vec[0] * cos_y + vec[1] * sin_y,
                -vec[0] * sin_y + vec[1] * cos_y
            ], dtype=np.float32)

        # 2. 算老大的相对位置 (转为局部)
        leader_rel_global = self.leader_pos - my_pos
        leader_rel = global_to_local(leader_rel_global)

        # ==========================================
        # ✅ 3. 核心升级：KNN + 感知半径 过滤兄弟
        # ==========================================
        entities = []
        myopic_radius = 1.5

        # 考察老大和所有兄弟 (不再区分是谁)
        all_others_pos = [self.leader_pos] + [self.follower_pos[k] for k in range(self.num_followers) if k != follower_idx]
        
        for pos in all_others_pos:
            rel_global = pos - my_pos
            dist = np.linalg.norm(rel_global)
            
            # 只有当队友极其靠近（即将发生碰撞）时，本地雷达才能看到
            if dist <= myopic_radius:
                entities.append({'dist': dist, 'rel_pos': global_to_local(rel_global)})

        # 排序并截取最近的 3 个
        entities = sorted(entities, key=lambda x: x['dist'])[:self.max_visible_teammates]

        # 填充到观测数组里喂给网络
        teammates_obs = np.zeros((self.max_visible_teammates * 3), dtype=np.float32)
        for i, e in enumerate(entities):
            idx = i * 3
            teammates_obs[idx] = e['rel_pos'][0]
            teammates_obs[idx+1] = e['rel_pos'][1]
            teammates_obs[idx+2] = 1.0   # 真实存在的标记

        side = 1.0 if follower_idx % 2 == 0 else -1.0
        rank = (follower_idx + 2) // 2
        max_rank = max(1, (self.num_followers + 1) // 2)
        rank_norm = rank / max_rank
        role_code = np.array([side, rank_norm], dtype=np.float32)

        formation_alpha = self.reward_fn.compute_formation_alpha(corridor_width)
        formation_alpha_obs = np.array([formation_alpha], dtype=np.float32)

        rel_yaw = (self.leader_yaw - my_yaw + np.pi) % (2 * np.pi) - np.pi

        leader_yaw_rate_norm = self.leader_yaw_rate / 0.15  
        # 告诉网络：老大的车头相对我的车头偏了多少度
        leader_heading_obs = np.array(
            [
                np.cos(rel_yaw),
                np.sin(rel_yaw),
                leader_yaw_rate_norm,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([
            lidar_obs,
            leader_rel,
            teammates_obs,
            role_code,
            formation_alpha_obs,
            leader_heading_obs,
        ])

        return obs

    def _simulate_radar(self, origin_pos, current_yaw): 
        num_rays = self.radar_rays 
        max_distance = 10.0
        step_size = self.map_resolution 
        
        lidar_data = formation_core.LidarData(num_rays, max_distance)
        angle_list = []
        range_list = []
        
        for i in range(num_rays):
            # 雷达朝向使用传入的角，不再用 leader_yaw
            angle = current_yaw + (2 * np.pi * i / num_rays)
            angle_list.append(angle)
            
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            hit_distance = max_distance
            for r in np.arange(0, max_distance, step_size):
                test_x = origin_pos[0] + r * dx
                test_y = origin_pos[1] + r * dy
                if self._is_obstacle(test_x, test_y):
                    hit_distance = round(r, 1)
                    # print("hit_distance:", hit_distance)
                    break
                    
            range_list.append(hit_distance)

        lidar_data.angles = angle_list
        lidar_data.ranges = range_list

        # print("lidar_data.ranges:", lidar_data.ranges)
            
        return lidar_data

    # ==========================================
    # 以下绘图和路径规划辅助函数基本保持不变
    # ==========================================
    def render(self, mode='human'):
        if not self.render_mode:
            return
        self.ax.clear()
        
        height, width = self.map_grid.shape
        extent = [-width/2 * self.map_resolution, width/2 * self.map_resolution,
                -height/2 * self.map_resolution, height/2 * self.map_resolution]

        display_grid = np.ones((height, width), dtype=np.float32)
        display_grid[self.inflated_map_grid] = 0.75
        display_grid[self.map_grid] = 0.0

        self.ax.imshow(display_grid, extent=extent, origin='lower', 
                    cmap='Greys', alpha=0.6, vmin=0.0, vmax=1.0)
        
        if hasattr(self, 'path') and len(self.path) > 0:
            self.ax.plot([p[0] for p in self.path], [p[1] for p in self.path], 
                        'y--', alpha=0.8, linewidth=2, label='Global Path')

        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self.ax.set_aspect('equal')

        self.ax.plot(self.leader_pos[0], self.leader_pos[1], 'ro', markersize=12, label='Leader')
        
        arrow_length = 0.8
        dx = arrow_length * np.cos(self.leader_yaw)
        dy = arrow_length * np.sin(self.leader_yaw)
        self.ax.arrow(self.leader_pos[0], self.leader_pos[1], dx, dy,
                    head_width=0.3, head_length=0.4, fc='r', ec='r', alpha=0.8)
        
        for pos in self.follower_pos:
            self.ax.plot(pos[0], pos[1], 'bo', markersize=8)
            
        # 画小弟及其独立的朝向
        for i, pos in enumerate(self.follower_pos):
            self.ax.plot(pos[0], pos[1], 'bo', markersize=8)
            
            # 给小弟加个短一点的方向箭头 (蓝色)
            follower_arrow_len = 0.5
            f_dx = follower_arrow_len * np.cos(self.follower_yaw[i])
            f_dy = follower_arrow_len * np.sin(self.follower_yaw[i])
            self.ax.arrow(pos[0], pos[1], f_dx, f_dy,
                        head_width=0.2, head_length=0.3, fc='b', ec='b', alpha=0.8)

        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_title(f'MARL Formation 2D Env | Step: {self.step_count}')
        plt.pause(0.01)

    def close(self):
        if self.render_mode:
            plt.close(self.fig)

    def _get_random_free_point(self):
        height, width = self.inflated_map_grid.shape
        while True:
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            if not self.inflated_map_grid[y, x]:
                return x, y

    def _plan_path(self, start_idx, goal_idx):
        queue = deque([(start_idx, [start_idx])])
        visited = set([start_idx])
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        
        while queue:
            current, path = queue.popleft()
            if current == goal_idx:
                return path
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < self.inflated_map_grid.shape[1] and 0 <= ny < self.inflated_map_grid.shape[0]:
                    if not self.inflated_map_grid[ny, nx] and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(nx, ny)]))
        return []
        
    def _grid_to_world(self, px, py):
        height, width = self.map_grid.shape
        x = (px - width / 2) * self.map_resolution
        y = (py - height / 2) * self.map_resolution
        return np.array([x, y], dtype=np.float32)
    
    def _inflate_map(self, grid, margin_meters):
        from scipy.ndimage import binary_dilation
        inflation_pixels = int(margin_meters / self.map_resolution)
        if inflation_pixels <= 0:
            return grid.copy()
        y, x = np.ogrid[-inflation_pixels:inflation_pixels+1, -inflation_pixels:inflation_pixels+1]
        kernel = x**2 + y**2 <= inflation_pixels**2
        inflated_grid = binary_dilation(grid, structure=kernel)
        return inflated_grid
    
    def _get_random_free_point_with_clearance(self, clearance_meters):
        """
        随机找一个自由点，并保证该点周围 clearance_meters 半径内没有障碍物
        """
        height, width = self.map_grid.shape
        clearance_pixels = int(clearance_meters / self.map_resolution)

        while True:
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)

            # 点本身必须可用
            if self.map_grid[y, x]:
                continue

            # 边界也要留足，不然圆形检查会越界
            if (
                x - clearance_pixels < 0 or x + clearance_pixels >= width or
                y - clearance_pixels < 0 or y + clearance_pixels >= height
            ):
                continue

            # 检查圆形邻域内是否有障碍物
            is_clear = True
            for dx in range(-clearance_pixels, clearance_pixels + 1):
                for dy in range(-clearance_pixels, clearance_pixels + 1):
                    if dx * dx + dy * dy <= clearance_pixels * clearance_pixels:
                        nx, ny = x + dx, y + dy
                        if self.map_grid[ny, nx]:
                            is_clear = False
                            break
                if not is_clear:
                    break

            if is_clear:
                return x, y