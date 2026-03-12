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
    def __init__(self, config):
        super().__init__()
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
        self.map_grid = self._load_or_generate_map()
        self.planning_safe_margin = 1.0 # 膨胀层的安全边距 (米)，根据实际情况调整
        self.inflated_map_grid = self._inflate_map(self.map_grid, self.planning_safe_margin)

        self.sensing_radius = config.get("sensing_radius", 5.0)  # 最大感知半径 (米)
        self.max_visible_teammates = 3  # 网络最多只管最近的 3 个兄弟
        
        self.reward_fn = MADDPGFormationReward()
        # 密码本，用于记录网络视野槽位对应的真实全局ID
        self.obs_mapping = {i: [] for i in range(self.num_followers)}
        
        self.leader_velocity = [0.0, 0.1]  
        self.leader_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.follower_pos = np.random.uniform(-1, 1, size=(self.num_followers, 2)).astype(np.float32)

        # ==========================================
        # ✅ 4. 彻底重构动作与状态空间 (针对单个智能体)
        # ==========================================
        # 动作空间：极其干净！就只有 X 和 Y 方向的相对位置指令
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # 维度 = 自己的雷达(21) + 老大的位置(2) + 最多3个兄弟的信息(3 * 3)
        # 兄弟信息为什么是 3 维？因为除了相对位移 (dx, dy)，我们还需要一个标志位 (is_valid)
        # 来告诉网络“这个槽位是不是真实存在的兄弟”（防止填 0 时被网络误认为是坐标原点的兄弟）
        obs_dim = 21 + 2 + (self.max_visible_teammates * 3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.render_mode = config.get("render", False)
        if self.render_mode:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()

        self.step_count = 0

    def _load_or_generate_map(self):
        map_path = "/home/lpp/formation_test/maps/underground_garage.pgm"
        if os.path.exists(map_path):
            img = Image.open(map_path).convert('L') 
            grid = np.array(img) < 250 
            return grid
        else:
            grid = np.zeros((200, 200), dtype=bool) 
            grid[0:5, :] = True; grid[-5:, :] = True
            grid[:, 0:5] = True; grid[:, -5:] = True
            grid[100:120, 0:80] = True
            grid[130:150, 120:200] = True
            grid[160:170, 90:110] = True
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

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        while True:
            start_idx = self._get_random_free_point()
            goal_idx = self._get_random_free_point()
            path_indices = self._plan_path(start_idx, goal_idx)
            if len(path_indices) > 10: 
                break
                
        self.path = [self._grid_to_world(px, py) for px, py in path_indices]
        self.leader_pos = self.path.pop(0)

        self.leader_yaw = 0.0
        if len(self.path) > 0:
            direction = self.path[0] - self.leader_pos
            self.leader_yaw = np.arctan2(direction[1], direction[0])

        self.follower_pos = np.random.uniform(-1, 1, size=(self.num_followers, 2)).astype(np.float32)
        self.follower_pos += self.leader_pos 
        
        self.step_count = 0
        self.prev_follower_pos = self.follower_pos.copy()

        # ✅ 5. 循环构建所有小弟的初始观测字典
        obs_dict = {}
        for i, agent_id in enumerate(self._agent_ids):
            obs_dict[agent_id] = self._get_single_follower_obs(i)

        if self.render_mode:
            self.render()

        return obs_dict, {}

    def step(self, action_dict, graph_dict):
        """
        ✅ 6. 核心步进函数改造，接收动作字典
        action_dict 格式: {"follower_0": [dx, dy], "follower_1": [dx, dy], ...}

        """
        # ==========================================
        # 1. 领航者按预设轨迹移动 (与原来一样)
        # ==========================================
        leader_done = False
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
                self.leader_yaw = np.arctan2(direction[1], direction[0])
        else:
            leader_done = True

        # ==========================================
        # 2. 跟随者根据动作字典各自移动
        # ==========================================
        cos_yaw = np.cos(self.leader_yaw)
        sin_yaw = np.sin(self.leader_yaw)
        
        for i, agent_id in enumerate(self._agent_ids):
            # 如果某个智能体挂了或者没给出动作，保持原地或采取默认动作
            if agent_id not in action_dict:
                continue
                
            local_action = np.array(action_dict[agent_id]) * 2.0 # 还原范围
            
            # 从局部坐标（相对于老大的车头）转为全局相对坐标
            global_dx = local_action[0] * cos_yaw - local_action[1] * sin_yaw
            global_dy = local_action[0] * sin_yaw + local_action[1] * cos_yaw
            
            # 更新该机器人的物理绝对坐标
            self.follower_pos[i] = self.leader_pos + np.array([global_dx, global_dy])

        self.step_count += 1
        is_timeout = self.step_count >= self.max_steps # 超时结束（没设计）

        # ✅ 3. 新增翻译官逻辑：根据 graph_dict 和密码本，生成全局连线列表
        dynamic_connections = {}
        for i, agent_id in enumerate(self._agent_ids):
            global_follower_id = i + 1
            dynamic_connections[global_follower_id] = []
            
            if agent_id in graph_dict:
                hard_weights = graph_dict[agent_id] # 网络吐出的比如 [1, 0, 0]
                mapping = self.obs_mapping[i]       # 密码本记录的比如 [0, 3, 4]
                
                for k, weight in enumerate(hard_weights):
                    if k < len(mapping) and weight > 0:
                        target_global_id = mapping[k]
                        dynamic_connections[global_follower_id].append(target_global_id)

        positions_3d = np.zeros((self.num_robots, 3), dtype=np.float32)
        positions_3d[0, :2] = self.leader_pos
        positions_3d[1:, :2] = self.follower_pos

        in_danger_zone = []
        for i in range(self.num_robots):
            pos = positions_3d[i, :2]
            px, py = self._world_to_grid(pos[0], pos[1])
            if 0 <= px < self.inflated_map_grid.shape[1] and 0 <= py < self.inflated_map_grid.shape[0]:
                in_danger_zone.append(bool(self.inflated_map_grid[py, px]))
            else:
                in_danger_zone.append(True)

        # ✅ 4. 把翻译好的连线扔给裁判！
        team_reward, reward_details = self.reward_fn.compute(
            leader_pos=self.leader_pos,
            follower_positions=self.follower_pos,
            prev_follower_positions=self.prev_follower_pos,
            leader_yaw=self.leader_yaw,
            in_danger_zone=in_danger_zone,
            dynamic_connections=dynamic_connections # 传入动态图！
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
        any_follower_crashed = any(in_danger_zone[1:])
        episode_done = leader_done or any_follower_crashed

        for i, agent_id in enumerate(self._agent_ids):
            obs_dict[agent_id] = self._get_single_follower_obs(i)
            # 所有人吃大锅饭：拿到完全一样的团队总分！
            reward_dict[agent_id] = team_reward 
            terminated_dict[agent_id] = episode_done
            truncated_dict[agent_id] = is_timeout
            info_dict[agent_id] = reward_details # 把扣分明细传出去，方便你写日志

        terminated_dict["__all__"] = episode_done
        truncated_dict["__all__"] = is_timeout

        self.prev_follower_pos = self.follower_pos.copy()

        if self.render_mode:
            self.render()

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def _get_single_follower_obs(self, follower_idx):
        my_pos = self.follower_pos[follower_idx]
        
        # 1. 算雷达 (保持不变)
        lidar_data = self._simulate_radar(my_pos)
        feature_extractor = formation_core.FeatureExtractor(8)
        features = feature_extractor.extract_features(lidar_data)
        
        lidar_obs = np.zeros(21, dtype=np.float32)
        lidar_obs[0] = features.corridor_width 
        lidar_obs[1] = features.front_clearance
        lidar_obs[2] = features.left_clearance  
        lidar_obs[3] = features.right_clearance 
        lidar_obs[4] = features.obstacle_density  
        lidar_obs[4:12] = np.array(features.sector_min_dists, dtype=np.float32)
        lidar_obs[13:21] = np.array(features.sector_avg_dists, dtype=np.float32)

        # 2. 算老大的相对位置
        leader_rel = self.leader_pos - my_pos
        
        # ==========================================
        # ✅ 3. 核心升级：KNN + 感知半径 过滤兄弟
        # ==========================================
        # ✅ 核心升级：把老大和其他小弟，统一放进“感知候选池”里
        entities = []
        
        # 1. 考察老大 (Global ID: 0)
        dist_to_leader = np.linalg.norm(self.leader_pos - my_pos)
        if dist_to_leader <= self.sensing_radius:
            entities.append({'dist': dist_to_leader, 'rel_pos': self.leader_pos - my_pos, 'global_id': 0})
            
        # 2. 考察其他兄弟 (Global ID: j + 1)
        for j in range(self.num_followers):
            if j != follower_idx:
                rel = self.follower_pos[j] - my_pos
                dist = np.linalg.norm(rel)
                if dist <= self.sensing_radius:
                    entities.append({'dist': dist, 'rel_pos': rel, 'global_id': j + 1})

        # 排序并截取最近的 3 个
        entities = sorted(entities, key=lambda x: x['dist'])[:self.max_visible_teammates]
        
        # ✅ 记录翻译密码本：这 3 个槽位分别是谁？
        self.obs_mapping[follower_idx] = [e['global_id'] for e in entities]

        # 填充到观测数组里喂给网络
        teammates_obs = np.zeros((self.max_visible_teammates * 3), dtype=np.float32)
        for i, e in enumerate(entities):
            idx = i * 3
            teammates_obs[idx] = e['rel_pos'][0]
            teammates_obs[idx+1] = e['rel_pos'][1]
            teammates_obs[idx+2] = 1.0 # 真实存在的标记
            
        obs = np.concatenate([lidar_obs, leader_rel, teammates_obs])
        return obs

    def _simulate_radar(self, origin_pos):
        num_rays = self.radar_rays
        max_distance = 10.0
        step_size = self.map_resolution
        
        lidar_data = formation_core.LidarData(num_rays, max_distance)
        lidar_data.max_range = max_distance
        lidar_data.num_beams = num_rays
        
        for i in range(num_rays):
            angle = self.leader_yaw + (2 * np.pi * i / num_rays) # 假设雷达朝向跟老大一致
            lidar_data.angles.append(angle)
            
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            hit_distance = max_distance
            for r in np.arange(0, max_distance, step_size):
                test_x = origin_pos[0] + r * dx
                test_y = origin_pos[1] + r * dy
                if self._is_obstacle(test_x, test_y):
                    hit_distance = r
                    break
                    
            lidar_data.ranges.append(hit_distance)
            
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
            
        # TODO: 由于环境不再直接接收控制图了，我们暂时画星型图，等 G2ANet 引入后再动态画线
        for pos in self.follower_pos:
            self.ax.plot([self.leader_pos[0], pos[0]],
                        [self.leader_pos[1], pos[1]], 'k-', alpha=0.5, linewidth=1.5)

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