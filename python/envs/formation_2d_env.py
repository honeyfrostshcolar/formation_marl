import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import formation_core 
from envs.reward_fn import FormationReward
import os
from PIL import Image
from collections import deque

class Formation2DEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.num_robots = config.get("num_robots", 3)
        self.num_followers = self.num_robots - 1
        self.max_robots = config.get("max_robots", 10)
        self.max_followers = self.max_robots - 1
        self.max_steps = config.get("max_steps", 100)
        self.radar_rays = config.get("radar_rays", 180)
        self.safety_threshold = config.get("safety_threshold", 0.5)
        self.max_comm_distance = config.get("max_comm_distance", 5.0)
        self.candidate_graphs = config.get("candidate_graphs", [])

        # ==========================================
        # ✅ 新增：地图相关配置
        # ==========================================
        self.map_resolution = 0.1  # 地图分辨率：每个像素代表 0.1 米
        self.map_origin = np.array([0.0, 0.0])  # 物理坐标系的原点 (x, y) 对应地图的中心
        self.map_grid = self._load_or_generate_map() # 获取二维数组地图 (True=障碍, False=空地)

        self.planning_safe_margin = 0.8  # 设置距离障碍物 0.8 米的安全冗余
        self.inflated_map_grid = self._inflate_map(self.map_grid, self.planning_safe_margin)
        
        self.reward_fn = FormationReward()
        self.leader_velocity = [0.0, 0.1]  
        self.leader_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.follower_pos = np.random.uniform(-1, 1, size=(self.num_followers, 2)).astype(np.float32)

        self.action_space = spaces.Dict({
            "graph_idx": spaces.Discrete(len(self.candidate_graphs)),
            "positions": spaces.Box(
                low=-1.0, high=1.0, 
                shape=(self.max_followers * 2,),
                dtype=np.float32
            )
        })
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        self.render_mode = config.get("render", False)
        if self.render_mode:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()

        self.step_count = 0

    def _load_or_generate_map(self):
        """加载 PGM 占据栅格地图，如果没有则生成一个默认的复杂地图"""
        map_path = "map.pgm"  # ✅ 改为你的 pgm 文件名
        if os.path.exists(map_path):
            print(f"Loading ROS standard PGM map from {map_path}...")
            # PIL 原生支持 PGM 读取
            img = Image.open(map_path).convert('L') 
            
            # 将图像转为 Numpy 数组
            # 设定阈值：只要像素值小于 250 (包含黑色障碍物和灰色未知区域)，就视为不可通行的 True
            grid = np.array(img) < 250 
            return grid
        else:
            print(f"No {map_path} found, generating default procedural map...")
            # 生成一个 200x200 像素的默认地图 (对应物理尺寸 20米 x 20米)
            grid = np.zeros((200, 200), dtype=bool) 
            
            # 增加四周墙壁
            grid[0:5, :] = True; grid[-5:, :] = True
            grid[:, 0:5] = True; grid[:, -5:] = True
            
            # 默认的障碍物块
            grid[100:120, 0:80] = True
            grid[130:150, 120:200] = True
            grid[160:170, 90:110] = True
            return grid

    def _world_to_grid(self, x, y):
        """物理坐标 (米) 转换为 像素索引"""
        # 假设 (0,0) 米在地图的正中心
        height, width = self.map_grid.shape
        pixel_x = int(x / self.map_resolution + width / 2)
        pixel_y = int(y / self.map_resolution + height / 2)
        return pixel_x, pixel_y

    def _is_obstacle(self, x, y):
        """检查某个物理坐标是否是障碍物"""
        px, py = self._world_to_grid(x, y)
        height, width = self.map_grid.shape
        if 0 <= px < width and 0 <= py < height:
            return self.map_grid[py, px]
        return True # 超出地图边界视为障碍物

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        # ✅ 1. 随机生成起点和终点，并规划路径
        while True:
            start_idx = self._get_random_free_point()
            goal_idx = self._get_random_free_point()
            path_indices = self._plan_path(start_idx, goal_idx)
            # 确保生成的路径足够长，否则重新生成
            if len(path_indices) > 10: 
                break
                
        # ✅ 2. 把像素路径点转化为物理坐标点
        self.path = [self._grid_to_world(px, py) for px, py in path_indices]
        
        # ✅ 3. 初始化领航者位置在起点
        self.leader_pos = self.path.pop(0)

        self.leader_yaw = 0.0
        if len(self.path) > 0:
            direction = self.path[0] - self.leader_pos
            self.leader_yaw = np.arctan2(direction[1], direction[0])

        self.follower_pos = np.random.uniform(-1, 1, size=(self.num_followers, 2)).astype(np.float32)
        self.follower_pos += self.leader_pos 

        
        
        self.step_count = 0
        obs = self._get_leader_observation()
        if self.render_mode:
            self.render()

        self.prev_follower_pos = self.follower_pos.copy()
        self.prev_graph_idx = None

        return obs, {}

    def step(self, action):
        done = False
        graph_idx = action["graph_idx"]
        action_graph = self.candidate_graphs[graph_idx]

        # 1. 判断拓扑图是否发生切换
        is_graph_changed = False
        if hasattr(self, 'prev_graph_idx') and self.prev_graph_idx is not None:
            is_graph_changed = (graph_idx != self.prev_graph_idx)
        self.prev_graph_idx = graph_idx

        if len(self.path) > 0:
            target_pos = self.path[0]
            direction = target_pos - self.leader_pos
            distance_to_target = np.linalg.norm(direction)
            move_step = 0.1  # 领航者的移动步长 (速度)
            
            if distance_to_target < move_step:
                # 如果离目标点很近了，直接踩上去，并把这个路径点从列表中剔除
                self.leader_pos = target_pos
                self.path.pop(0)
            else:
                # 否则，朝着目标点方向走一步
                self.leader_pos += (direction / distance_to_target) * move_step
                self.leader_yaw = np.arctan2(direction[1], direction[0])
        else:
            # 路径走完了，可以提前结束回合或者原地等待
            done = True

        pos_action = action["positions"]
        active_pos_action = pos_action[:self.num_followers * 2] 
        local_rel = active_pos_action.reshape(self.num_followers, 2) * 2.0
        
        cos_yaw = np.cos(self.leader_yaw)
        sin_yaw = np.sin(self.leader_yaw)
        
        global_rel = np.zeros_like(local_rel)
        # 假设动作的 local_rel[:, 0] 是沿着车头方向(X)，local_rel[:, 1] 是垂直于车头方向(Y)
        global_rel[:, 0] = local_rel[:, 0] * cos_yaw - local_rel[:, 1] * sin_yaw
        global_rel[:, 1] = local_rel[:, 0] * sin_yaw + local_rel[:, 1] * cos_yaw
        
        self.follower_pos = self.leader_pos + global_rel

        positions_3d = np.zeros((self.num_robots, 3), dtype=np.float32)
        positions_3d[0, :2] = self.leader_pos
        positions_3d[1:, :2] = self.follower_pos

        prev_positions_3d = np.zeros((self.num_robots, 3), dtype=np.float32)
        prev_positions_3d[0, :2] = self.leader_pos # 领航者历史位置不影响跟随者相对计算
        prev_positions_3d[1:, :2] = getattr(self, 'prev_follower_pos', self.follower_pos)

        # ✅ 3. 检测危险区 (Inflated Map)
        in_danger_zone = []
        for i in range(self.num_robots):
            pos = positions_3d[i, :2]
            px, py = self._world_to_grid(pos[0], pos[1])
            if 0 <= px < self.inflated_map_grid.shape[1] and 0 <= py < self.inflated_map_grid.shape[0]:
                in_danger_zone.append(bool(self.inflated_map_grid[py, px]))
            else:
                in_danger_zone.append(True) # 越界也是危险区

        # ✅ 4. 调用全新的终极奖励函数
        reward, reward_details = self.reward_fn.compute(
            positions_3d, prev_positions_3d, action_graph, 
            is_graph_changed, self.leader_yaw, in_danger_zone
        )

        # 更新记忆，供下一帧使用
        self.prev_follower_pos = self.follower_pos.copy()

        self.step_count += 1
        # done = self.step_count >= self.max_steps

        # ==========================================
        # ✅ 更新：严格的碰撞检测（机器人互碰 + 撞墙）
        # ==========================================
        # 1. 检测机器人间互撞
        for i in range(self.num_followers):
            for j in range(i+1, self.num_followers):
                dist = np.linalg.norm(self.follower_pos[i] - self.follower_pos[j])
                if dist < self.safety_threshold:
                    reward -= 5.0
                    
        # 2. 检测撞墙 (所有真实机器人)
        for i in range(self.num_robots):
            pos = positions_3d[i, :2]
            if self._is_obstacle(pos[0], pos[1]):
                reward -= 10.0 # 严厉的撞墙惩罚！
                done = True

        obs = self._get_leader_observation()
        info = {
            "reward_details": reward_details,
            "leader_pos": self.leader_pos.copy(),
            "follower_positions": self.follower_pos.copy(),
            "chosen_graph": int(graph_idx)
        }

        if self.render_mode:
            self.render()
        return obs, float(reward), done, False, info

    def _get_leader_observation(self):
        obs = np.zeros(21, dtype=np.float32)
        lidar_data = self._simulate_radar(self.leader_pos)
        feature_extractor = formation_core.FeatureExtractor(8)
        features = feature_extractor.extract_features(lidar_data)

        obs[0] = features.corridor_width 
        obs[1] = features.front_clearance
        obs[2] = features.left_clearance  
        obs[3] = features.right_clearance 
        obs[4] = features.obstacle_density  

        min_dists = np.array(features.sector_min_dists, dtype=np.float32)
        avg_dists = np.array(features.sector_avg_dists, dtype=np.float32)
        obs[4:12] = min_dists  
        obs[13:21] = avg_dists 
        return obs

    def _simulate_radar(self, origin_pos):
        """
        ✅ 更新：基于占据栅格地图的真实射线步进算法 (Ray Casting)
        """
        num_rays = self.radar_rays # 180条射线覆盖 360 度
        max_distance = 10.0
        step_size = self.map_resolution # 步长与像素大小一致0.1
        
        lidar_data = formation_core.LidarData(num_rays, max_distance)
        lidar_data.max_range = max_distance
        lidar_data.num_beams = num_rays
        
        for i in range(num_rays):
            angle = self.leader_yaw + (2 * np.pi * i / num_rays)
            lidar_data.angles.append(angle)
            
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # 射线步进检测
            hit_distance = max_distance
            for r in np.arange(0, max_distance, step_size):
                test_x = origin_pos[0] + r * dx
                test_y = origin_pos[1] + r * dy
                if self._is_obstacle(test_x, test_y):
                    hit_distance = r
                    break # 遇到障碍物，结束当前射线的检测
                    
            lidar_data.ranges.append(hit_distance)
            
        return lidar_data

    def render(self, mode='human'):
        if not self.render_mode:
            return
        
        # 1. 彻底清除上一帧的绘图内容，防止多帧叠加产生残影
        self.ax.clear()
        
        # ==========================================
        # ✅ 核心升级：构建分层 Costmap 可视化背景
        # ==========================================
        # 获取地图尺寸和物理边界 (extent)
        height, width = self.map_grid.shape
        extent = [-width/2 * self.map_resolution, width/2 * self.map_resolution,
                -height/2 * self.map_resolution, height/2 * self.map_resolution]

        # 创建一张用于显示的彩色底图 grid (默认全白 1.0)
        # 形状为 (height, width)，数据类型为 float，方便表示灰度
        display_grid = np.ones((height, width), dtype=np.float32)

        # 逻辑层 1：将膨胀区域 (inflated_map_grid) 涂成浅灰色 (0.75)
        # inflated_map_grid 包括了真实障碍物及其四周，所以这一步是把整个不可规划区变灰
        display_grid[self.inflated_map_grid] = 0.75

        # 逻辑层 2：将真实的障碍物 (map_grid) 涂成纯黑色 (0.0)
        # 这一步会覆盖掉灰色层中属于真实墙壁的部分
        display_grid[self.map_grid] = 0.0

        # 使用 imshow 显示合成后的三色代价地图
        # origin='lower' 确保数学坐标系与图片坐标系一致
        # cmap='Greys' 映射：0.0->黑，0.75->灰，1.0->白
        # vmin/vmax 定死颜色映射范围，防止自动调整导致灰色不统一
        self.ax.imshow(display_grid, extent=extent, origin='lower', 
                    cmap='Greys', alpha=0.6, vmin=0.0, vmax=1.0)
        
        # ==========================================
        # ✅ 2. 绘制规划路径 (黄色虚线)
        # ==========================================
        # 加上 linewidth 增强可视性
        if hasattr(self, 'path') and len(self.path) > 0:
            self.ax.plot([p[0] for p in self.path], [p[1] for p in self.path], 
                        'y--', alpha=0.8, linewidth=2, label='Global Path')

        # ✅ 3. 设置坐标轴范围和物理比例
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self.ax.set_aspect('equal') # 极其重要：保证 1米:1米 的真实物理比例

        # ✅ 4. 绘制编队中的机器人
        # 绘制领航者（红色大圆点）
        self.ax.plot(self.leader_pos[0], self.leader_pos[1], 'ro', markersize=12, label='Leader')
        
        # ✅ 新增：为领航者添加前进朝向箭头 (以车头为基准，X-Forward)
        # 箭头长度可以根据需要调整，例如 0.8 米
        arrow_length = 0.8
        # 箭头向量 dx, dy：基于 leader_yaw 计算
        dx = arrow_length * np.cos(self.leader_yaw)
        dy = arrow_length * np.sin(self.leader_yaw)
        
        # 使用 ax.arrow 绘制
        # (x, y) 为起点，(dx, dy) 为向量
        self.ax.arrow(self.leader_pos[0], self.leader_pos[1], dx, dy,
                    head_width=0.3, head_length=0.4, fc='r', ec='r', alpha=0.8)
        
        # 绘制跟随者（蓝色小圆点）
        for pos in self.follower_pos:
            self.ax.plot(pos[0], pos[1], 'bo', markersize=8)
            
        # 绘制拓扑连线，加上 alpha 防止遮挡障碍物细节
        for pos in self.follower_pos:
            self.ax.plot([self.leader_pos[0], pos[0]],
                        [self.leader_pos[1], pos[1]], 'k-', alpha=0.5, linewidth=1.5)

        # ✅ 5. 加上网格和标题
        self.ax.grid(True, linestyle='--', alpha=0.3)
        # 标题增加朝向指示的说明
        self.ax.set_title(f'Formation 2D Env | X-Forward (ROS Standard) | Leader Heading')
        
        # 如果提供了 label，显示图例
        # self.ax.legend(loc='upper right', fontsize='small')

        # 刷新 matplotlib 窗口
        plt.pause(0.01)

    def close(self):
        if self.render_mode:
            plt.close(self.fig)

    def _get_random_free_point(self):
        """在地图上随机寻找一个没有障碍物的空地"""
        height, width = self.inflated_map_grid.shape
        while True:
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            if not self.inflated_map_grid[y, x]: # 如果是 False (空地)
                return x, y

    def _plan_path(self, start_idx, goal_idx):
        """使用 BFS 算法在占据栅格地图上规划最短路径"""
        queue = deque([(start_idx, [start_idx])])
        visited = set([start_idx])
        # 允许 8 邻域移动 (包含斜向)
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        
        while queue:
            current, path = queue.popleft()
            if current == goal_idx:
                return path
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                # 检查边界和碰撞
                if 0 <= nx < self.inflated_map_grid.shape[1] and 0 <= ny < self.inflated_map_grid.shape[0]:
                    if not self.inflated_map_grid[ny, nx] and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(nx, ny)]))
        return [] # 如果找不到路，返回空
        
    def _grid_to_world(self, px, py):
        """像素索引 转换为 物理坐标 (米)"""
        height, width = self.map_grid.shape
        x = (px - width / 2) * self.map_resolution
        y = (py - height / 2) * self.map_resolution
        return np.array([x, y], dtype=np.float32)
    
    def _inflate_map(self, grid, margin_meters):
        """将地图上的障碍物膨胀一定半径，生成专用的安全规划地图"""
        from scipy.ndimage import binary_dilation
        
        inflation_pixels = int(margin_meters / self.map_resolution)
        if inflation_pixels <= 0:
            return grid.copy()
            
        # 创建一个圆形的膨胀核 (保证四面八方的安全距离都一致)
        y, x = np.ogrid[-inflation_pixels:inflation_pixels+1, -inflation_pixels:inflation_pixels+1]
        kernel = x**2 + y**2 <= inflation_pixels**2
        
        # 使用 scipy 进行二值膨胀：只要靠近障碍物 inflation_pixels 个像素内的区域，全变成 True (不可通行)
        inflated_grid = binary_dilation(grid, structure=kernel)
        return inflated_grid