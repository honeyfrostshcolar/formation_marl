import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class FormationDataset(Dataset):
    """编队学习数据集"""
    
    def __init__(self, csv_file, feature_columns, transform=None):
        """
        Args:
            csv_file: CSV文件路径
            feature_columns: 特征列名列表
            transform: 数据变换
        """
        self.data = pd.read_csv(csv_file)
        self.feature_columns = feature_columns #[]
        self.transform = transform
        
        # 确保数据包含必要的列
        required_columns = ['env_type', 'robot_x', 'robot_y', 'robot_theta', 'has_expert_label'] #robot_x、robot_y、robot_theta是领航的初始位置和方向
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def __len__(self):
        return len(self.data)
    
    # 索引获取样本
    def __getitem__(self, idx):
        sample = self.data.iloc[idx] # 获取第idx行的样本
        
        # 环境特征
        features = torch.tensor(sample[self.feature_columns].values.astype(np.float32)) #[feature_dim]
        
        # 机器人位姿
        leader_pose = {
            'position': {'x': sample['robot_x'], 'y': sample['robot_y']},
            'orientation': sample['robot_theta']
        }
        
        # 环境类型
        environment_type = sample['env_type']
        
        # 专家标注（如果有）
        has_expert = sample['has_expert_label']
        if has_expert:

            ctrlGraph_str = sample["controlGraph"] 
            row_strs = ctrlGraph_str.split("；")
            ctrlGraph_list = []
            for row_str in row_strs:
                # 拆分列并转数值（处理空值/异常值）
                col_vals = [float(val.strip()) for val in row_str.split("|") if val.strip()]
                ctrlGraph_list.append(col_vals)

            ctrlGraph = torch.tensor(ctrlGraph_list, dtype=torch.float32)

            expert_positions_str = sample["robot_positions"] 
            pos_row_strs = expert_positions_str.split("；")
            expert_positions_list = []
            for row_str in pos_row_strs:
                # 拆分列并转数值（处理空值/异常值）
                col_vals = [float(val.strip()) for val in row_str.split("|") if val.strip()]
                expert_positions_list.append(col_vals)

            expert_positions = torch.tensor(expert_positions_list, dtype=torch.float32)

            # expert_graph_idx = int(sample['expert_graph_index'])
            
            # # 根据编队索引生成专家位置（这里简化处理）
            # if expert_graph_idx == 0:  # 三角形编队
            #     expert_positions = np.array([[0, 0], [0.5, 0.87], [0.5, -0.87]], dtype=np.float32)
            # elif expert_graph_idx == 1:  # 线性编队
            #     expert_positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
            # else:  # 紧凑编队
            #     expert_positions = np.array([[0, 0], [0.7, 0], [0, 0.7]], dtype=np.float32)
            
            # expert_positions = torch.tensor(expert_positions)
            
        else:
            # 现在不处理这种情况
            expert_graph_idx = -1
            expert_positions = torch.zeros(3, 2, dtype=torch.float32) # 这里3应该是机器人数量
        
        sample_dict = {
            'features': features,
            'leader_pose': leader_pose,
            'environment_type': environment_type,
            'has_expert': has_expert,
            'controlGraph': ctrlGraph,
            'expert_positions': expert_positions
        }
        
        # 不打算设计
        if self.transform:
            sample_dict = self.transform(sample_dict)
        
        return sample_dict

class CurriculumDataLoader:
    """
    课程学习数据加载器
    按照环境复杂度逐步增加训练难度
    """
    
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir # 数据目录
        self.batch_size = batch_size
        
        # 课程学习阶段定义（现假设只有一个阶段）
        # self.curriculum_stages = [
        #     {
        #         'name': 'stage1_easy',
        #         'env_types': ['open_space'],
        #         'expert_ratio': 0.8,
        #         'epochs': 50,
        #         'imitation_weight': 0.8
        #     },
        #     {
        #         'name': 'stage2_medium', 
        #         'env_types': ['open_space', 'corridor'],
        #         'expert_ratio': 0.5,
        #         'epochs': 100,
        #         'imitation_weight': 0.5
        #     },
        #     {
        #         'name': 'stage3_hard',
        #         'env_types': ['open_space', 'corridor', 'complex_obstacles'],
        #         'expert_ratio': 0.2,
        #         'epochs': 150,
        #         'imitation_weight': 0.2
        #     }
        # ]
        self.curriculum_stages = [
            {
                'name': 'stage1_easy',
                'env_types': ['open_space'],
                'expert_ratio': 0.8,
                'epochs': 50,
                'imitation_weight': 0.8
            }
        ]
        
        # 特征列定义（根据实际CSV文件调整），通过feature_columns里的特征来从CSV文件中提取数据
        self.feature_columns = [
            'corridor_width', 'front_clearance', 'left_clearance', 'right_clearance', 
            'obstacle_density'
        ] + [f'sector_{i}_min' for i in range(8)] + [f'sector_{i}_avg' for i in range(8)]
    
    def get_stage_dataloader(self, stage_idx, phase='train'):
        """获取指定阶段的数据加载器"""
        if stage_idx >= len(self.curriculum_stages):
            raise ValueError(f"Invalid stage index: {stage_idx}")
        
        stage = self.curriculum_stages[stage_idx]
        
        # 加载对应阶段的数据文件（假设现在就只有一个阶段数据，直接加载）
        csv_file = f"{self.data_dir}/formation_data_{stage['name']}_{phase}.csv" 
        # csv_file = "/home/lpp/formation_test/data/formation_training_data_open_space.csv" 
        
        try:
            dataset = FormationDataset(csv_file, self.feature_columns)
        except FileNotFoundError:
            # 如果文件不存在，使用完整数据集并过滤
            # full_dataset = FormationDataset(f"{self.data_dir}/formation_training_{phase}.csv", self.feature_columns)
            # dataset = self._filter_dataset_by_stage(full_dataset, stage)

            # 如果不存在直接报错
            raise FileNotFoundError(f"错误：未找到指定文件 '{csv_file}'，程序终止运行！")
        
        # 创建数据加载器
        # 循环切分数据成批次；
        # 手动打乱数据顺序；
        # 单进程加载数据（速度慢）；
        # 手动处理数据加载的异常（比如样本数不是 batch_size 整数倍）；
        # 这些都是重复且易出错的工作，DataLoader 全帮你封装好了。
        # 训练时直接 for batch in dataloader 就能循环取批次数据

        # 训练时打乱数据，验证时不打乱
        shuffle = True if phase == 'train' else False
        dataloader = DataLoader(  
            dataset, 
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4
        )
        
        return dataloader, stage
    
    def _filter_dataset_by_stage(self, dataset, stage):
        """根据阶段过滤数据集"""
        filtered_indices = []
        
        for idx in range(len(dataset)):
            sample = dataset.data.iloc[idx]
            if sample['env_type'] in stage['env_types']:
                # 根据专家比例随机选择
                if np.random.random() < stage['expert_ratio'] or sample['has_expert_label']:
                    filtered_indices.append(idx)
        
        # 创建子集
        from torch.utils.data import Subset
        return Subset(dataset, filtered_indices) # 返回根据阶段过滤后的数据集子集
    
    def get_total_epochs(self):
        """获取总训练轮数"""
        return sum(stage['epochs'] for stage in self.curriculum_stages)