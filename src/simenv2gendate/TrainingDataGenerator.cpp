#include "simenv2gendate/TrainingDataGenerator.h"
#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <Eigen/LU>

TrainingDataGenerator::TrainingDataGenerator(const std::vector<FormationConfig>& graphs) 
    : control_graphs_(graphs),
      feature_extractor_(8){

    std::random_device rd;
    rng_.seed(rd()); // 调用rng_()生成随机数种子
}

// 生成单个训练样本
TrainingSample TrainingDataGenerator::generateSample(const std::string& env_type, 
                                                   int map_width, int map_height,
                                                   double resolution) {
    TrainingSample sample;
    sample.environment_type = env_type;
    
    // 生成环境
    std::vector<std::vector<int>> grid_map;
    if (env_type == "corridor") {
        grid_map = env_generator_.createCorridor(map_width, map_height);
    } else if (env_type == "open_space") {
        grid_map = env_generator_.createOpenSpace(map_width, map_height);
    } else if (env_type == "complex_obstacles") {
        grid_map = env_generator_.createComplexObstacles(map_width, map_height);
    } else {
        grid_map = env_generator_.createRandomEnvironment(map_width, map_height);
    }
    
    lidar_sim_.setGridMap(grid_map, resolution);
    
    // 在自由空间随机选择机器人位置
    std::vector<Point2D> free_positions;
    for (int y = 0; y < map_height; ++y) {
        for (int x = 0; x < map_width; ++x) {
            if (grid_map[y][x] == 0) {
                double x_pos = x * resolution;
                double y_pos = -y * resolution;
                free_positions.push_back(Point2D(x_pos, y_pos));
            }
        }
    }
    
    if (free_positions.empty()) {
        // 如果没有自由空间，返回默认样本
        return sample;
    }
    
    std::uniform_int_distribution<int> dist_pos(0, free_positions.size() - 1); //每次调用 dist_pos(rng_)会从区间里等概率返回一个 int 整数。
    std::uniform_real_distribution<double> dist_angle(0, 2 * M_PI);//每次调用 dist_angle(rng_)会从区间里等概率返回一个 double 浮点数。
    std::uniform_real_distribution<double> expert_prob(0.0, 1.0);
    
    //领航者位置和朝向
    Point2D selected_pos = free_positions[dist_pos(rng_)];
    double selected_angle = dist_angle(rng_); //我觉得可以直接用领航者的角度
    
    sample.leader_pose = RobotState(selected_pos.x, selected_pos.y, selected_angle);
    
    // 模拟激光雷达扫描，默认参数int num_beams = 360, double max_range = 10.0
    LidarData lidar_data = lidar_sim_.simulateScan(
        selected_pos.x, selected_pos.y, selected_angle, 360, 10.0);
    
    // 提取特征
    sample.features = feature_extractor_.extractFeatures(lidar_data);

    // 生成专家标注
    // ============== 是否生成专家标注 ============== //
    // if(expert_prob(rng_) < 0.4){
    //     sample.expert_formation = generateExpertFormation(env_type, sample.features, sample.expert_confidence);
    // } 
    
    sample.expert_formation.ctrlnums = control_graphs_.size();

    sample.has_expert_label = (sample.expert_formation.controlGraph.sum() != 0); //是否有专家数据
    
    return sample;
}

std::vector<TrainingSample> TrainingDataGenerator::generateDataset(int num_samples, 
                                                                 int map_width, 
                                                                 int map_height,
                                                                 double resolution) {
    std::vector<TrainingSample> dataset;
    //std::vector<std::string> env_types = {"corridor", "open_space", "complex_obstacles"}; // 环境类型（走廊、开放空间、复杂障碍物）
    std::string env_type = "open_space";
    
    //std::uniform_int_distribution<int> dist_env(0, env_types.size() - 1);
    
    for (int i = 0; i < num_samples; ++i) {

        TrainingSample sample = generateSample(env_type, map_width, map_height, resolution);
        
        if (!sample.environment_type.empty()) {  // 有效的样本
            dataset.push_back(sample);
        }
        
        if (i % 100 == 0) {
            std::cout << "Generated " << i << "/" << num_samples << " samples" << std::endl;
        }
    }
    
    return dataset;
}

void TrainingDataGenerator::saveDataset(const std::vector<TrainingSample>& dataset, 
                                    const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    // 更新标题行
    file << "env_type,robot_x,robot_y,robot_theta,has_expert_label,expert_confidence,";
    file << "ctrlnums,controlGraph,robot_positions,";
    file << "corridor_width,front_clearance,left_clearance,right_clearance,obstacle_density,";

    for (int i = 0; i < 8; ++i) {
        file << "sector_" << i << "_min,";
    }
    for (int i = 0; i < 8; ++i) {   
        file << "sector_" << i << "_avg" << (i < 7 ? "," : "");
    }
    file << std::endl;

    // 写入数据
    for (const auto& sample : dataset) {
        file << sample.serialize() << ",";

        // 写入特征
        const auto& features = sample.features.toVector();
        for (size_t i = 0; i < features.size(); ++i) {
            file << features[i] << (i < features.size() - 1 ? "," : "");
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Dataset saved to " << filename << std::endl;
}

FormationConfig TrainingDataGenerator::generateExpertFormation(const std::string& env_type, 
                                                              const EnvironmentFeatures& features,
                                                              double& confidence) { //我先暂时不去管置信度
    FormationConfig expert;
    // expert.ctrlnums = control_graphs_.size();
    
    confidence = 1.0;
    // 简化规则：根据环境类型选择编队
    if (env_type == "corridor" || features.corridor_width < 3.0) {
        // 狭窄环境：选择线性编队
        expert.graph_index = 1;  // 假设索引1是线性编队
        expert.controlGraph = control_graphs_[expert.graph_index].controlGraph;
        expert.positions = {//都是相对位置
            {0, 0},   // 领航者
            {1, 0},   // 跟随者1
            {2, 0}    // 跟随者2
        };
    } else if (env_type == "open_space") {
        // 开阔环境：选择三角形编队
        //前方最小距离
        if(features.right_clearance < 0.5 || features.left_clearance < 0.5){//太小，用直线，一字型
            for(auto& control_graph : control_graphs_){
                
                Eigen::FullPivLU<Eigen::MatrixXd> r(control_graph.controlGraph.cast<double>());//Eigen强转为double类型
                int rank = r.rank();

                if(rank == control_graph.controlGraph.rows() - 1){
                    expert.controlGraph = control_graph.controlGraph;
                    break;
                }
            }
            expert.positions = {//都是相对位置右手坐标系，小车前进为x轴，左转为y轴
                {0, 0},        // 领航者
                {-0.5, 0.0},   // 跟随者1
                {-1.0, 0.0}   // 跟随者2
            };
             
        }else{
            for(auto& control_graph : control_graphs_){
                Eigen::FullPivLU<Eigen::MatrixXd> r(control_graph.controlGraph.cast<double>());
                int rank = r.rank();

                if( rank == 1 ){
                    expert.controlGraph = control_graph.controlGraph;
                    break;
                }
            }
            expert.positions = {//都是相对位置右手坐标系，小车前进为x轴，左转为y轴
                {0, 0},        // 领航者
                {-0.5, 0.5},   // 跟随者1
                {-0.5, -0.5}   // 跟随者2
            };
        }
    } else {
        // 中等环境：选择紧凑编队
        expert.graph_index = 2;  // 假设索引2是紧凑编队
        expert.controlGraph = control_graphs_[expert.graph_index].controlGraph;
        expert.positions = {
            {0, 0},    // 领航者
            {0.7, 0},  // 跟随者1
            {0, 0.7}   // 跟随者2
        };
    }
    
    return expert;
}
