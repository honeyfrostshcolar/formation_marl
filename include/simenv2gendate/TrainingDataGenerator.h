#ifndef TRAINING_DATA_GENERATOR_H
#define TRAINING_DATA_GENERATOR_H

#include "types.h"
#include "LidarSimulator.h"
#include "FeatureExtractor.h"
#include "EnvironmentGenerator.h"
#include <vector>
#include <string>
#include <random>

class TrainingDataGenerator {
private:
    LidarSimulator lidar_sim_;
    EnvironmentGenerator env_generator_;
    std::vector<FormationConfig> control_graphs_;
    FeatureExtractor feature_extractor_;
    std::mt19937 rng_;
    
    // 生成专家标注（简化版本）
    FormationConfig generateExpertFormation(const std::string& env_type, 
                                           const EnvironmentFeatures& features,
                                           double& confidence);
    
public:
    TrainingDataGenerator(const std::vector<FormationConfig>& graphs);
    
    // 生成单个训练样本
    TrainingSample generateSample(const std::string& env_type, 
                                 int map_width, int map_height, double resolution);
    
    // 批量生成训练数据
    std::vector<TrainingSample> generateDataset(int num_samples, 
                                               int map_width = 50, 
                                               int map_height = 50,
                                               double resolution = 0.1);
    
    // 保存数据集到文件
    void saveDataset(const std::vector<TrainingSample>& dataset, const std::string& filename);
};

// 示例控制图定义
std::vector<FormationConfig> createControlGraphs();

#endif