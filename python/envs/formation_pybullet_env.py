import pybullet as p
import pybullet_data
import numpy as np
import gym
import formation_core
from gym import spaces
from envs.reward_fn import FormationReward

class FormationPyBulletEnv(gym.Env):
    """单智能体环境：领航者根据局部观测决策全局编队"""
    
    def __init__(self, config):
        # 解析配置
        self.num_robots = config.get("num_robots", 3)
        self.num_followers = self.num_robots - 1
        self.max_steps = config.get("max_steps", 100)
        self.radar_rays = config.get("radar_rays", 180)
        self.safety_threshold = config.get("safety_threshold", 0.5)
        self.max_comm_distance = config.get("max_comm_distance", 5.0)
        self.candidate_graphs = config.get("candidate_graphs", [])

        self.reward_fn = FormationReward()
        self.leader_velocity = [0.0, 0.1, 0.0]
        self.leader_pos = [0.0, 0.0, 0.3]
        self.obstacle_ids = [10]

        # 根据 render 参数决定连接模式（训练时通常为 False）
        render = config.get("render", False)
        if render:
            self.physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0])
        else:
            self.physics_client = p.connect(p.DIRECT)

        # 初始化 PyBullet 设置
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setTimeStep(0.01)

        # 动作空间和观测空间
        self.action_space = spaces.Dict({
            "graph_idx": spaces.Discrete(len(self.candidate_graphs)),
            "positions": spaces.Box(low=-1.0, high=1.0, shape=(self.num_followers * 2,), dtype=np.float32)
        })
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        self.robot_ids = []
        self.step_count = 0
        self.done = False

        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # 加载地面
        p.loadURDF("plane.urdf")

        # 创建机器人
        self.robot_ids = []
        for i in range(self.num_robots):
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2) # 碰撞形状：球形，半径0.2米（物理引擎用，决定碰撞检测）
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1,0,0,1] if i==0 else [0,0,1,1]) # 视觉形状：球形，半径0.2米（渲染用，和碰撞形状一致）
            robot_id = p.createMultiBody(
                baseMass=1.0 if i == 0 else 0.5,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[i * 0.5, 0, 0.5],
                baseOrientation=[0,0,0,1]
            )
            self.robot_ids.append(robot_id)

        if self.physics_client == p.GUI:
            p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0])
        p.stepSimulation()

        self.step_count = 0
        self.done = False
        return self._get_leader_observation()
    
    def _get_leader_observation(self):
        leader_pos, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        obs = np.zeros(21, dtype=np.float32)

        lidar_data = self._scan_radar(leader_pos)
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
        obs[12:20] = avg_dists
        
        return obs
    
    def _scan_radar(self, origin_pos):
        num_rays = self.radar_rays
        max_distance = 10.0
        lidar_data = formation_core.LidarData(num_rays, max_distance)
        lidar_data.max_range = max_distance
        lidar_data.num_beams = num_rays
        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays
            lidar_data.angles.append(angle)
            direction = [np.cos(angle), np.sin(angle), 0]
            result = p.rayTest(
                origin_pos,
                np.array(origin_pos) + np.array(direction) * max_distance
            )[0]
            if result[0] != -1 and result[2] > 0.1:
                lidar_data.ranges.append(result[2] * max_distance)
            else:
                lidar_data.ranges.append(max_distance)
        return lidar_data
    
    def step(self, action):
        graph_idx = action["graph_idx"]
        action_graph = self.candidate_graphs[graph_idx]
        pos_action = action["positions"]
        graph_idx_norm = action[0]
        graph_idx = int((graph_idx_norm + 1) / 2 * (len(self.candidate_graphs) - 1))
        graph_idx = np.clip(graph_idx, 0, len(self.candidate_graphs) - 1)
        action_graph = self.candidate_graphs[graph_idx]

        # 领航者移动
        self.leader_pos[0] += self.leader_velocity[0]
        self.leader_pos[1] += self.leader_velocity[1]
        p.resetBasePositionAndOrientation(self.robot_ids[0], self.leader_pos, [0,0,0,1])
        
        #print("action:", action)
        # import sys
        # sys.exit(0)
        # 跟随者目标相对位置
        follower_positions = []
        for i in range(self.num_followers):
            rel_x = action[1 + 2*i] * 2.0 # 映射米的物理范围
            rel_y = action[1 + 2*i + 1] * 2.0
            follower_positions.append([rel_x, rel_y])
        
        leader_pos, _ = p.getBasePositionAndOrientation(self.robot_ids[0])
        for i, rel_pos in enumerate(follower_positions):
            abs_pos = [
                leader_pos[0] + rel_pos[0],
                leader_pos[1] + rel_pos[1],
                0.1
            ]
            p.resetBasePositionAndOrientation(self.robot_ids[i+1], abs_pos, [0,0,0,1])

        robot_positions = []
        for i in range(self.num_robots):
            pos, _ = p.getBasePositionAndOrientation(self.robot_ids[i])
            robot_positions.append(pos)
        positions = np.array(robot_positions)
        
        p.stepSimulation()
        self.step_count += 1

        #print("positions:", positions)
        
        reward, reward_details = self.reward_fn.compute(positions, action_graph)
        done = self.step_count >= self.max_steps

        if self._check_collision_with_obstacles():
            done = True
            reward += -10.0

        obs = self._get_leader_observation()
        info = {
            "reward_details": reward_details,
            "leader_pos": leader_pos,
            "follower_positions": follower_positions
        }

        # if self.step_count % 5 == 0:
        #     print(f"Step {self.step_count}: Graph={graph_idx}, Distances={distances}")

        return obs, reward, done, info
    
    def close(self):
        p.disconnect(self.physics_client)

    def _check_collision_with_obstacles(self):
        for robot_id in self.robot_ids:
            # 获取所有接触点
            contact_points = p.getContactPoints(robot_id)
            for cp in contact_points:
                # 检查对方物体是否是障碍物（可以根据ID或名称判断）
                if cp[2] in self.obstacle_ids:
                    return True
        return False