
import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from envs.reward_fn import FormationReward

class FormationPyBulletEnv(gym.Env):
    """单智能体环境：领航者根据局部观测决策全局编队"""
    
    def __init__(self, 
                 num_robots=3, 
                 max_steps=100,
                 radar_rays=18,      # 雷达射线数量（对应18维障碍物特征）
                 safety_threshold=0.5,
                 max_comm_distance=5.0):
        super().__init__()
        
        self.num_robots = num_robots
        self.num_followers = num_robots - 1
        self.max_steps = max_steps
        self.radar_rays = radar_rays
        self.safety_threshold = safety_threshold
        self.max_comm_distance = max_comm_distance

        self.reward_fn = FormationReward()
        
        # 动作 = [编队图索引(离散), 跟随者1_x, 跟随者1_y, ..., 跟随者N_x, 跟随者N_y]
        # 实际训练中：编队图索引由策略网络采样（离散），位置由网络输出（连续）
        # RLlib中通过自定义模型处理混合动作，环境接收已采样的具体值
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1 + 2 * self.num_followers,),  # [graph_idx_norm, pos_x1, pos_y1, ...]
            dtype=np.float32
        )
        
        # [0:3] = 领航者位置(x,y,z)
        # [3:21] = 18维雷达障碍物距离（归一化）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )
        
        # 初始化PyBullet
        self.physics_client = p.connect(p.DIRECT)  # 无GUI加速训练
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setTimeStep(0.01)
        
        self._create_obstacles()
        
        self.reset()
    
    def _create_obstacles(self):
        """创建环境障碍物（用于雷达扫描）"""
        # 示例：创建4面墙
        wall_height = 2.0
        wall_thickness = 0.2
        # 左墙
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, 10, wall_height])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(...), basePosition=[-10, 0, wall_height/2])
        # ... 其他墙（略）
        # 可扩展：从CSV加载障碍物配置
    
    def reset(self):
        """重置环境，返回领航者观测（21维）"""
        p.resetSimulation() # 重置仿真
        p.setGravity(0, 0, -9.8) # 重置重力
        self._create_obstacles() # 重新创建障碍物
        
        # 重新创建机器人（领航者 + 跟随者）
        self.robot_ids = []
        for i in range(self.num_robots):
            robot_id = p.createMultiBody(
                baseMass=1.0 if i == 0 else 0.5,  # 领航者质量稍大
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.2),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1,0,0,1] if i==0 else [0,0,1,1]),
                basePosition=[0, 0, 0.3] if i==0 else [0, 0, 0.1],  # 领航者初始位置
                baseOrientation=[0,0,0,1]
            )
            self.robot_ids.append(robot_id)
        
        self.step_count = 0
        self.done = False
        
        # === 关键修正3：仅返回领航者观测 ===
        return self._get_leader_observation()
    
    def _get_leader_observation(self):
        """获取领航者的21维局部观测"""
        # 1. 领航者自身位姿 (3维: x, y, z)
        leader_pos, leader_orn = p.getBasePositionAndOrientation(self.robot_ids[0])
        obs = np.zeros(21, dtype=np.float32)
        obs[0:3] = leader_pos[:3]  # 仅取x,y,z（z轴高度）
        
        # 2. 雷达扫描障碍物 (18维: 18个方向的最近障碍物距离)
        # === 关键修正4：实现真实雷达扫描 ===
        radar_distances = self._scan_radar(leader_pos)
        obs[3:21] = radar_distances  # 归一化到[0,1]或保留原始距离
        
        return obs
    
    def _scan_radar(self, origin_pos):
        """射线检测模拟雷达（18个方向）"""
        num_rays = self.radar_rays
        max_distance = 10.0  # 雷达最大探测距离
        distances = np.zeros(num_rays, dtype=np.float32)
        
        # 生成18个水平方向的射线（360度均匀分布）
        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays
            direction = [
                np.cos(angle), 
                np.sin(angle), 
                0  # 水平扫描
            ]
            # 射线检测
            result = p.rayTest(
                origin_pos, 
                np.array(origin_pos) + np.array(direction) * max_distance
            )[0]
            
            # 解析结果：hitFraction=0表示击中自身，1表示无碰撞
            if result[0] != -1 and result[2] > 0.1:  # 排除自身碰撞
                distances[i] = result[2] * max_distance
            else:
                distances[i] = max_distance  # 无障碍物设为最大距离
        
        # 可选：归一化到[0,1]
        # distances = distances / max_distance
        return distances
    
    def step(self, action):
        """
        执行领航者决策：
        action = [graph_idx_norm, follower1_x, follower1_y, ..., followerN_x, followerN_y]
        """
        # === 关键修正5：解析动作 ===
        # 编队图索引（归一化值 -> 实际索引）
        graph_idx_norm = action[0]  # [-1,1] -> 映射到[0, num_graphs-1]
        # 实际使用时：在自定义模型中完成离散采样，此处接收具体索引值
        # 为简化，此处假设graph_idx已由外部处理为整数（训练时需注意）
        
        # 跟随者目标相对位置（归一化值 -> 实际坐标）
        follower_positions = []
        for i in range(self.num_followers):
            rel_x = action[1 + 2*i] * 5.0  # 缩放因子：将[-1,1]映射到[-5,5]米
            rel_y = action[1 + 2*i + 1] * 5.0
            follower_positions.append([rel_x, rel_y])
        
        # === 关键修正6：环境执行编队（非智能体决策）===
        # 根据领航者当前位置 + 相对位置，计算跟随者绝对位置
        leader_pos, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        for i, rel_pos in enumerate(follower_positions):
            abs_pos = [
                leader_pos[0] + rel_pos[0],
                leader_pos[1] + rel_pos[1],
                0.1  # 固定高度
            ]
            # 直接设置跟随者位置（环境执行，非物理驱动）
            p.resetBasePositionAndOrientation(
                self.robot_ids[i+1], 
                abs_pos, 
                [0,0,0,1]
            )
        
        # 推进物理仿真（仅领航者受物理影响，跟随者由环境重置位置）
        p.stepSimulation()
        self.step_count += 1
        
        # === 关键修正7：奖励基于全局编队状态（但由领航者获得）===
        reward, reward_details = self._compute_global_reward()
        
        # 终止条件
        done = self.step_count >= self.max_steps
        # 可扩展：碰撞检测、任务完成等
        
        # 返回：仅领航者观测 + 标量奖励
        obs = self._get_leader_observation()
        info = {
            "reward_details": reward_details,
            "leader_pos": leader_pos,
            "follower_positions": follower_positions
        }
        
        return obs, reward, done, info
    
    def _compute_reward(self, robot_id):
        """计算特定机器人的奖励"""
        # 获取所有机器人的位置
        positions = []
        for i in range(self.num_robots):
            pos, _ = p.getBasePositionAndOrientation(self.robot_ids[i])
            positions.append(pos)
        positions = np.array(positions)
        
        # 获取当前机器人ID的图（这里简化，使用固定图）
        graph = np.eye(self.num_robots)
        
        # 计算奖励
        reward, _ = self.reward_fn.compute(robot_id, positions, graph)
        return reward
    
    def close(self):
        """关闭环境"""
        p.disconnect(self.physics_client)