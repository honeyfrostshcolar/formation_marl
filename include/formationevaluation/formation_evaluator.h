#ifndef FORMATION_EVALUATOR_H
#define FORMATION_EVALUATOR_H

#include <vector>
#include <cmath>
#include <limits>
#include <simenv2gendate/types.h>

// 二维点结构
// struct Point2D {
//     double x, y;
//     Point2D(double x = 0, double y = 0) : x(x), y(y) {}
// };

// // 机器人状态
// struct RobotState {
//     Point2D position;  // 绝对坐标
//     double orientation; // 朝向（弧度）
// };

// // 编队配置
// struct FormationConfig {
//     std::vector<std::vector<int>> controlGraph; // 控制图邻接矩阵
//     std::vector<Point2D> positions;             // 相对位置配置
// };

// 环境信息
struct Environment {
    std::vector<std::vector<int>> gridMap;      // 栅格地图 (0=空闲, 1=障碍)
    RobotState leaderState;                     // 领航者状态
    double safetyThreshold;                     // 安全阈值（碰撞风险）
    double maxCommDistance;                     // 最大通信距离（通讯效率）
    std::vector<double> lidarScan;              // 激光雷达扫描数据（表示的是每个角度的障碍物距离，最大为maxRange）
    double corridorWidth;                       // 通道宽度(如果编队宽度远小于通道宽度，那么编队可能没有充分利用可用空间，可能不是最优）

    Environment(const std::vector<std::vector<int>>& grid, 
                const RobotState& state, 
                double safety, 
                double commDist);

    // 模拟激光雷达扫描（简化实现）
    void simulateLidar(int numBeams = 360, double maxRange = 10.0);
    // 射线投射
    double rayCast(const Point2D& start, double angle, double maxRange);
    // 计算通道宽度
    void calculateCorridorWidth();
        
};

class FormationEvaluator {
private:
    // 权重参数
    double w_navigation;   // 导航效率权重
    double w_collision;    // 碰撞风险权重
    double w_communication; // 通信质量权重
    
    // 将相对位置转换为绝对坐标（只是转换跟随者的，领航本来用的是绝对坐标）
    std::vector<Point2D> convertToAbsolutePositions(
        const std::vector<Point2D>& relativePositions, 
        const RobotState& leaderState);
    
    // 评估导航效率
    double evaluateNavigationEfficiency(
        const Environment& env, 
        const FormationConfig& formation,
        const std::vector<Point2D>& absolutePositions);
    
    // 计算编队宽度
    double calculateFormationWidth(
        const FormationConfig& formation,
        const std::vector<Point2D>& absolutePositions,
        double orientation);
    
    // 评估碰撞风险
    double evaluateCollisionRisk(
        const Environment& env,
        const std::vector<Point2D>& absolutePositions);
    
    // 检查附近障碍物
    bool checkNearbyObstacles(
        const Environment& env,
        const Point2D& position,
        double safetyThreshold);
    
    // 评估通信质量
    double evaluateCommunicationQuality(
        const FormationConfig& formation,
        const std::vector<Point2D>& absolutePositions,
        double maxCommDistance);
    
public:
    FormationEvaluator(double w_nav = 0.4, double w_col = 0.3, double w_comm = 0.3);
    
    // 主评估函数
    double evaluateFormation(const Environment& env, const FormationConfig& formation);
};

#endif // FORMATION_EVALUATOR_H
