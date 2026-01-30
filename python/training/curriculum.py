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

        #总机器人数量
        robot_nums = sample['ctrlnums']
        
        # 环境类型
        environment_type = sample['env_type']
        # print(environment_type)
        
        sample_dict = {
            'features': features,
            'leader_pose': leader_pose,
            'environment_type': environment_type,
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
        self.data_dir = data_dir # 数据目录(假设data_dir是具体的CSV文件路径)
        self.batch_size = batch_size

        # 特征列定义（根据实际CSV文件调整），通过feature_columns里的特征来从CSV文件中提取数据
        self.feature_columns = [
            'corridor_width', 'front_clearance', 'left_clearance', 'right_clearance', 
            'obstacle_density'
        ] + [f'sector_{i}_min' for i in range(8)] + [f'sector_{i}_avg' for i in range(8)]
    
    def get_stage_dataloader(self):
        """获取指定数据加载器"""
        
        try:
            dataset = FormationDataset(self.data_dir, self.feature_columns)
        except FileNotFoundError:

            raise FileNotFoundError(f"错误：未找到指定文件 '{self.data_dir}'，程序终止运行！")
        
        # 创建数据加载器
        # 循环切分数据成批次；
        # 手动打乱数据顺序；
        # 单进程加载数据（速度慢）；
        # 手动处理数据加载的异常（比如样本数不是 batch_size 整数倍）；
        # 这些都是重复且易出错的工作，DataLoader 全帮你封装好了。
        # 训练时直接 for batch in dataloader 就能循环取批次数据

        # 训练时打乱数据，验证时不打乱
        dataloader = DataLoader(  
            dataset, 
            batch_size=self.batch_size, # 32
            shuffle=True,
            num_workers=4
        )
        
        return dataloader
    
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