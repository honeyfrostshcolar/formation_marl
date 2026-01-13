#include "simenv2gendate/TrainingDataGenerator.h"
#include "enumeration/formation_enumeration.h"
#include <iostream>
#include <map>
#include <algorithm>

int main() {
    
    std::cout << "=== 训练数据生成系统 ===" << std::endl;
    
    // 1. 创建控制图
    formation::FormationEnumerator enumerator(3);
    std::vector<formation::ControlGraph> formations = enumerator.enumerateAllFormations();
    
    std::vector<FormationConfig> control_graphs;
    for(const auto& form : formations){
        FormationConfig config;
        config.controlGraph = form.getAdjacencyMatrix();
        control_graphs.push_back(config);
    }
    
    std::cout << "创建了 " << control_graphs.size() << " 个控制图" << std::endl;
    
    TrainingDataGenerator data_generator(control_graphs);
    
    // 3. 生成训练数据集
    std::cout << "开始生成训练数据..." << std::endl;
    // std::cout << "开始生成测试数据..." << std::endl;
    std::vector<TrainingSample> dataset = data_generator.generateDataset(500, 250, 250, 0.1); //假设左上角坐标为(0.0,0.0)
    // std::vector<TrainingSample> dataset = data_generator.generateDataset(1000, 200, 200, 0.1);

    // 4. 保存数据集
    std::string filename = "/home/lpp/formation_test/data/formation_data_stage1_easy_val.csv";
    // std::string filename = "/home/lpp/formation_test/data/formation_data_stage1_easy_train.csv";
    data_generator.saveDataset(dataset, filename);
    
    // // 5. 显示统计信息
    // std::cout << "\n=== 数据集统计 ===" << std::endl;
    // std::cout << "总样本数: " << dataset.size() << std::endl;
    
    // // 统计各环境类型的样本数量
    // std::map<std::string, int> env_counts;
    // std::map<int, int> formation_counts;
    
    // for (const auto& sample : dataset) {
    //     env_counts[sample.environment_type]++;
    //     formation_counts[sample.expert_formation.graph_index]++;
    // }
    
    // std::cout << "环境类型分布:" << std::endl;
    // for (const auto& pair : env_counts) {
    //     std::cout << "  " << pair.first << ": " << pair.second << std::endl;
    // }
    
    // std::cout << "编队选择分布:" << std::endl;
    // for (const auto& pair : formation_counts) {
    //     std::cout << "  编队 " << pair.first << ": " << pair.second << std::endl;
    // }
    
    // // 6. 显示前几个样本的特征
    // std::cout << "\n=== 前3个样本的特征 ===" << std::endl;
    // for (int i = 0; i < std::min(3, static_cast<int>(dataset.size())); ++i) {
    //     const auto& sample = dataset[i];
    //     std::cout << "样本 " << i + 1 << ":" << std::endl;
    //     std::cout << "  环境: " << sample.environment_type << std::endl;
    //     std::cout << "  位置: (" << sample.robot_pose.position.x << ", " 
    //               << sample.robot_pose.position.y << "), 朝向: " 
    //               << sample.robot_pose.orientation << std::endl;
    //     std::cout << "  专家选择编队: " << sample.expert_formation.graph_index << std::endl;
    //     std::cout << "  通道宽度: " << sample.features.corridor_width << std::endl;
    //     std::cout << "  障碍物密度: " << sample.features.obstacle_density << std::endl;
    //     std::cout << std::endl;
    // }
    
    return 0;
}
