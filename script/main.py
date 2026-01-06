import torch
import torch.optim as optim
import pandas as pd
import argparse
import os
import sys
import cv2
import numpy as np

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from python.models.formation_net import ConstrainedFormationNet, HybridLoss
from python.training.trainer import FormationTrainer
from python.training.curriculum import CurriculumDataLoader

def main():
    # parser = argparse.ArgumentParser(description='Formation Learning Training')
    # parser.add_argument('--data-dir', type=str, required=True, help='Training data directory') # 训练数据目录
    # parser.add_argument('--output-dir', type=str, default='/data', help='Output directory') # 输出目录
    # parser.add_argument('--num-robots', type=int, default=3, help='Number of robots') # 机器人数量
    # parser.add_argument('--num-graphs', type=int, default=3, help='Number of control graphs') # 控制图数量
    # parser.add_argument('--feature-dim', type=int, default=26, help='Feature dimension') # 特征维度
    # parser.add_argument('--batch-size', type=int, default=32, help='Batch size') # 批次大小
    # parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate') # 学习率
    # parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint') # 恢复训练的检查点
    
    # args = parser.parse_args()
    # 初始化C++模块
    # cpp_evaluator = formation_core.FormationEvaluator(0.4, 0.3, 0.3) #FormationConfig的评分模块，既有位置，又有图的评估
    # cpp_enumerator = formation_core.FormationEnumerator(robot_num) #cpp编队枚举模块
    # cpp_enumerator.get_all_formations()



    class Args:
        pass
    args = Args()
    
    # 手动赋值所有参数
    args.data_dir = "/home/lpp/formation_test/data"  # 替换为你的训练数据实际路径
    args.map = "/home/lpp/formation_test/data/env_map.pgm"  # 替换为你的PGM文件实际路径
    df = pd.read_csv(args.data_dir, encoding="utf-8")
    args.output_dir = "/data"                      # 输出目录（可改）
    args.feature_dim = 26                          # 特征维度
    args.batch_size = 32                           # 批次大小
    args.lr = 1e-3                                 # 学习率
    args.resume = None                             # 恢复训练的检查点（无需恢复则为None）
    args.num_robots = 3                            # 机器人数量
    args.num_graphs = df.iloc[0, 6]                # 控制图数量

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化C++模块（需要先编译）
    try:

        import formation_core

        try:
            # 读取PGM图像（OpenCV自动识别PGM格式）
            pgm_path = args.map  # 替换为你的PGM文件实际路径
            obstacle_threshold = 128  # 灰度阈值，低于该值视为障碍物
            img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        
            if img is None:
                print(f"错误：无法读取PGM文件 {pgm_path}（可能路径错误或格式不支持）")
            
            # 灰度值转栅格值
            grid_map = np.where(img < obstacle_threshold, 1, 0).tolist()
    
        except Exception as e:
            print(f"读取PGM失败：{str(e)}")
            return []
        
        cpp_evaluator = formation_core.FormationEvaluator(0.4, 0.3, 0.3)
        print("Successfully loaded C++ core module")
    except ImportError:
        print("Warning: C++ core module not found, using dummy evaluator")
        cpp_evaluator = None
    
    # 初始化模型
    model = ConstrainedFormationNet(
        feature_dim=args.feature_dim,
        num_graphs=args.num_graphs,
        num_robots=args.num_robots,
        min_distance=0.5, # 跟随者与领航者的最小距离
        max_distance=3.0, # 跟随者与领航者的最大距离
        angle_range=(-90, 90) # 跟随者与领航者的角度范围
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # Adam优化器（用于更新模型参数
    loss_fn = HybridLoss(imitation_weight=0.7, rl_weight=0.3, diversity_weight=0.1)
    
    # 初始化训练器
    trainer = FormationTrainer(model, optimizer, device, cpp_evaluator)
    
    # 恢复训练（加载上一次未训练完的内容，继续开始训练）
    start_epoch = 0
    current_stage = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, current_stage = trainer.load_checkpoint(args.resume)
        start_epoch += 1  # 从下一轮开始
    
    # 初始化课程学习数据加载器
    curriculum_loader = CurriculumDataLoader(args.data_dir, args.batch_size)
    
    # 训练循环
    total_epochs = curriculum_loader.get_total_epochs()
    global_epoch = start_epoch
    
    for stage_idx in range(current_stage, len(curriculum_loader.curriculum_stages)):
        stage_info = curriculum_loader.curriculum_stages[stage_idx]
        stage_epochs = stage_info['epochs']
        
        print(f"\n=== Starting Curriculum Stage {stage_idx + 1}: {stage_info['name']} ===")
        print(f"Environment types: {stage_info['env_types']}")
        print(f"Expert ratio: {stage_info['expert_ratio']}")
        print(f"Epochs: {stage_epochs}")
        print(f"Imitation weight: {stage_info['imitation_weight']}")
        
        # 获取当前阶段的数据加载器
        train_loader, _ = curriculum_loader.get_stage_dataloader(stage_idx, 'train') # 训练数据加载器（打乱，加载、处理用于数据的训练）
        val_loader, _ = curriculum_loader.get_stage_dataloader(stage_idx, 'val') # 验证数据加载器（不打乱，加载、处理用于数据的验证）
        
        # 阶段内训练循环
        for stage_epoch in range(stage_epochs):
            epoch = global_epoch + stage_epoch
            
            print(f"\nEpoch {epoch + 1}/{total_epochs} (Stage {stage_idx + 1}.{stage_epoch + 1})")
            
            # 确定训练阶段和权重
            # 整个训练过程中，模型的神经网络权重是连续迭代更新的，后一阶段完全基于前一阶段的学习成果继续优化
            if stage_epoch < stage_epochs * 0.3:  # 前30%：主要模仿学习（也不是只有模仿学习，也有强化学习）
                training_phase = "imitation"
                imitation_weight = stage_info['imitation_weight']
            elif stage_epoch < stage_epochs * 0.7:  # 中间40%：混合学习
                training_phase = "mixed" 
                imitation_weight = stage_info['imitation_weight'] * 0.5
            else:  # 后30%：主要强化学习
                training_phase = "rl_finetune"
                imitation_weight = stage_info['imitation_weight'] * 0.2
            
            # 训练一个epoch
            train_loss, loss_components = trainer.train_epoch(
                train_loader, loss_fn, training_phase, 
                imitation_weight, stage_idx, grid_map
            )
            
            # 验证
            if stage_epoch % 10 == 0 or stage_epoch == stage_epochs - 1:
                val_score = trainer.validate(val_loader, stage_idx) # val_loader是验证数据加载器（validate函数现在还没有写）
                print(f"Validation Score: {val_score:.4f}")
            
            # 保存检查点
            if stage_epoch % 20 == 0 or stage_epoch == stage_epochs - 1:
                checkpoint_path = os.path.join(
                    args.output_dir, 
                    f"checkpoint_stage{stage_idx+1}_epoch{epoch+1}.pth"
                )
                trainer.save_checkpoint(checkpoint_path, epoch, stage_idx)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        global_epoch += stage_epochs
    
    # 最终评估和绘图
    print("\n=== Training Completed ===")
    final_val_score = trainer.validate(val_loader, len(curriculum_loader.curriculum_stages) - 1)
    print(f"Final Validation Score: {final_val_score:.4f}")
    
    # 绘制训练进度
    plot_path = os.path.join(args.output_dir, "training_progress.png")
    trainer.plot_training_progress(plot_path)
    print(f"Training progress plot saved: {plot_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    trainer.save_checkpoint(final_model_path, total_epochs - 1, len(curriculum_loader.curriculum_stages) - 1)
    print(f"Final model saved: {final_model_path}")

if __name__ == '__main__':
    main()