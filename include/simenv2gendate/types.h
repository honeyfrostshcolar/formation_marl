#ifndef TYPES_H
#define TYPES_H

#define _USE_MATH_DEFINES   
#include <cmath>

#include <vector>
#include <string>
#include <Eigen/Core>



// 二维点结构
struct Point2D {
    double x, y;
    Point2D(double x = 0, double y = 0) : x(x), y(y) {}
    
    // 计算两点间距离
    double distanceTo(const Point2D& other) const;
};

// 机器人状态
struct RobotState {
    Point2D position;  // 绝对坐标
    double orientation; // 朝向（弧度）
    
    RobotState(double x = 0, double y = 0, double theta = 0);
};

// 激光雷达数据
struct LidarData {
    std::vector<double> ranges;           // 距离测量值
    std::vector<double> angles;           // 对应的角度（弧度），我们以车头方向为0，顺时针为正
    double max_range;                     // 最大测量范围
    int num_beams;                        // 光束数量

    LidarData(int beams, double max_r) ;
};

// 环境特征（从激光雷达数据提取）
struct EnvironmentFeatures {
    double corridor_width;                // 估计的通道宽度
    double front_clearance;               // 前方最小距离
    double left_clearance;                // 左侧最小距离
    double right_clearance;               // 右侧最小距离
    double obstacle_density;              // 障碍物密度
    std::vector<double> sector_min_dists; // 各扇形区域最小距离
    std::vector<double> sector_avg_dists; // 各扇形区域平均距离
    
    // 转换为特征向量
    std::vector<double> toVector() const;
};

// 编队配置
struct FormationConfig {
    //std::vector<std::vector<int>> controlGraph; // 控制图邻接矩阵
    int ctrlnums;                            // 控制图总数
    Eigen::MatrixXi controlGraph; // 控制图邻接矩阵
    std::vector<Point2D> positions;             // 相对位置配置（相对于领航者）
    int graph_index;                            // 控制图索引
    
    FormationConfig();
};

// 训练数据样本
struct TrainingSample {
    EnvironmentFeatures features;          // 环境特征
    FormationConfig expert_formation;      // 专家选择的编队
    RobotState leader_pose;                 // 领航者机器人的位姿
    std::string environment_type;          // 环境类型

    bool has_expert_label;                 // 标记是否有专家标注
    double expert_confidence;              // 专家置信度
    
    TrainingSample() : has_expert_label(false), expert_confidence(0.0) {}
    // 序列化用于保存
    std::string serialize() const;
};

#endif