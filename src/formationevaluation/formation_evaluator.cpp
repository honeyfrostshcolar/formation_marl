#include "formationevaluation/formation_evaluator.h"
#include <algorithm>
#include <iostream>


// Environment 构造函数实现
Environment::Environment(const std::vector<std::vector<int>>& grid, 
                         const RobotState& state,
                         double safety, 
                         double commDist)
    : gridMap(grid), leaderState(state), safetyThreshold(safety), maxCommDistance(commDist) {

    // 模拟激光雷达扫描
    simulateLidar(360, 10.0);
    // 计算通道宽度
    calculateCorridorWidth();

}

// 模拟激光雷达扫描（简化实现）
void Environment::simulateLidar(int numBeams , double maxRange ) {
    lidarScan.resize(numBeams);
    double angleStep = 2 * M_PI / numBeams;

    for (int i = 0; i < numBeams; ++i) {
        double angle = leaderState.orientation + i * angleStep;
        lidarScan[i] = rayCast(leaderState.position, angle, maxRange);
    }
    
    // for (int i = 0; i < numBeams; ++i) {
    //     std::cout << "lidarScan[" << i << "]：" << lidarScan[i] << std::endl;
    // }
}

// 射线投射
double Environment::rayCast(const Point2D& start, double angle, double maxRange) {
    double step = 0.1;
    double currentDist = 0;

    while (currentDist < maxRange) {
        double x = start.x + currentDist * std::cos(angle);
        double y = start.y + currentDist * std::sin(angle);

        int gridX = static_cast<int>(x);
        int gridY = static_cast<int>(y);

        if (gridX < 0 || gridX >= gridMap[0].size() || 
            gridY < 0 || gridY >= gridMap.size()) {
            return maxRange;
        }

        if (gridMap[gridY][gridX] == 1) {
            return currentDist;
        }

        currentDist += step;
    }

    return maxRange;
}

// 计算通道宽度
void Environment::calculateCorridorWidth() {
    int numBeams = lidarScan.size();
    // std::cout << "numBeams：" << numBeams << std::endl;
    // 假设激光雷达数据中，索引0为前方，索引90为左侧，索引270为右侧
    // 然后不应该计算左右两边的距离，应该再保守一点，可以计算80度和280度的距离
    int leftIndex = numBeams / 4.5;   // 80度
    int rightIndex = 3 * numBeams / 4.5; // 280度

    double leftDist = std::abs(lidarScan[leftIndex] * std::sin(80 * M_PI / 180.0)); // 计算垂直距离
    double rightDist = std::abs(lidarScan[rightIndex] * std::sin(280 * M_PI / 180.0)); // 计算垂直距离

    corridorWidth = leftDist + rightDist;

}

// 构造函数实现
FormationEvaluator::FormationEvaluator(double w_nav, double w_col, double w_comm)
    : w_navigation(w_nav), w_collision(w_col), w_communication(w_comm) {}

// 将相对位置转换为绝对坐标
std::vector<Point2D> FormationEvaluator::convertToAbsolutePositions(
    const std::vector<Point2D>& relativePositions, 
    const RobotState& leaderState) {
    
    std::vector<Point2D> absolutePositions;
    
    for (const auto& relPos : relativePositions) {
        // 考虑机器人朝向的坐标变换
        double cosTheta = std::cos(leaderState.orientation);
        double sinTheta = std::sin(leaderState.orientation);
        
        Point2D absPos;
        absPos.x = leaderState.position.x + relPos.x * cosTheta - relPos.y * sinTheta;
        absPos.y = leaderState.position.y + relPos.x * sinTheta + relPos.y * cosTheta;
        
        absolutePositions.push_back(absPos);
    }
    
    return absolutePositions;
}

// 评估导航效率
double FormationEvaluator::evaluateNavigationEfficiency(
    const Environment& env, 
    const FormationConfig& formation,
    const std::vector<Point2D>& absolutePositions) {
    
    // 计算编队宽度（在前进方向的横向宽度）
    double formationWidth = calculateFormationWidth(formation, absolutePositions, env.leaderState.orientation);
    // std::cout << "formationWidth：" << formationWidth << std::endl;
    // std::cout << "corridorWidth：" << env.corridorWidth << std::endl;

    // 计算效率得分：编队宽度与通道宽度的匹配程度
    double widthDifference = std::abs(formationWidth - env.corridorWidth);
    double efficiencyScore = std::exp(-widthDifference / env.corridorWidth);
    
    return efficiencyScore;
}

// 计算编队宽度
double FormationEvaluator::calculateFormationWidth(
    const FormationConfig& formation,
    const std::vector<Point2D>& absolutePositions,
    double orientation) {
    
    // 计算前进方向的垂直向量
    double perpX = -std::sin(orientation);
    double perpY = std::cos(orientation);
    
    // 计算所有机器人在垂直方向上的投影
    double minProjection = std::numeric_limits<double>::max();
    double maxProjection = std::numeric_limits<double>::lowest();
    
    for (const auto& pos : absolutePositions) {
        double projection = pos.x * perpX + pos.y * perpY;
        minProjection = std::min(minProjection, projection);
        maxProjection = std::max(maxProjection, projection);
    }
    
    return maxProjection - minProjection;
}

// 评估碰撞风险
double FormationEvaluator::evaluateCollisionRisk(
    const Environment& env,
    const std::vector<Point2D>& absolutePositions) {
    
    int collisionCount = 0;
    
    for (const auto& pos : absolutePositions) {
        // 将连续坐标转换为栅格坐标
        int gridX = static_cast<int>(pos.x);
        int gridY = static_cast<int>(pos.y);
        
        // 检查是否超出地图边界
        if (gridX < 0 || gridX >= env.gridMap[0].size() || 
            gridY < 0 || gridY >= env.gridMap.size()) {
            collisionCount++;
            continue;
        }
        
        // 检查是否与障碍物重叠
        if (env.gridMap[gridY][gridX] == 1) {
            collisionCount++;
            continue;
        }
        
        // 检查与障碍物的距离（可选：检查周围栅格）
        if (checkNearbyObstacles(env, pos, env.safetyThreshold)) {
            collisionCount++;
        }
    }
    
    // 碰撞风险得分：有碰撞风险的机器人比例
    return static_cast<double>(collisionCount) / absolutePositions.size();
}

// 检查附近障碍物（在范围内有障碍物则返回true）
bool FormationEvaluator::checkNearbyObstacles(
    const Environment& env,
    const Point2D& position,
    double safetyThreshold) {
    
    int checkRadius = static_cast<int>(std::ceil(safetyThreshold));
    
    for (int dy = -checkRadius; dy <= checkRadius; ++dy) {
        for (int dx = -checkRadius; dx <= checkRadius; ++dx) {
            int checkX = static_cast<int>(position.x) + dx;
            int checkY = static_cast<int>(position.y) + dy;
            
            // 检查边界
            if (checkX < 0 || checkX >= env.gridMap[0].size() || 
                checkY < 0 || checkY >= env.gridMap.size()) {
                continue;
            }
            
            // 如果是障碍物，计算实际距离
            if (env.gridMap[checkY][checkX] == 1) {
                double distance = std::sqrt(dx*dx + dy*dy);
                if (distance < safetyThreshold) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

// 评估通信质量
double FormationEvaluator::evaluateCommunicationQuality(
    const FormationConfig& formation,
    const std::vector<Point2D>& absolutePositions,
    double maxCommDistance) {
    
    int validLinks = 0;
    int totalLinks = 0;
    
    const Eigen::MatrixXi& graph = formation.controlGraph;
    int numRobots = graph.rows();
    // std::cout << "graph ：" << graph << "numRobots：" << numRobots << std::endl;
    
    // 遍历控制图的所有边
    for (int i = 0; i < numRobots; ++i) {
        for (int j = 0; j < numRobots; ++j) {
            // std::cout << "i：" << i << "j：" << j << std::endl;
            if (graph(i, j) == 1) {  // 存在控制关系
                totalLinks++;
                
                // 计算两个机器人之间的距离
                double distance = std::sqrt(
                    std::pow(absolutePositions[i].x - absolutePositions[j].x, 2) +
                    std::pow(absolutePositions[i].y - absolutePositions[j].y, 2)
                );
                
                if (distance <= maxCommDistance) {
                    validLinks++;
                }
            }
        }
    }

    // std::cout << "222222222222222" << std::endl;
    
    if (totalLinks == 0) {
        return 0.0;  // 没有通信链路
    }
    
    return static_cast<double>(validLinks) / totalLinks;
}

// 主评估函数
double FormationEvaluator::evaluateFormation(const Environment& env, const FormationConfig& formation) {

    // std::cout << "1111111111111111" << std::endl;
    // 将跟随者相对位置转换为绝对坐标
    std::vector<Point2D> absolutePositions = 
        convertToAbsolutePositions(formation.positions, env.leaderState);
    
    // 计算各项得分
    double navScore = evaluateNavigationEfficiency(env, formation, absolutePositions);
    double collisionScore = evaluateCollisionRisk(env, absolutePositions);
    double commScore = evaluateCommunicationQuality(formation, absolutePositions, env.maxCommDistance);
    
    //std::cout << "navScore：" << navScore << std::endl;

    // 综合得分
    double totalScore = w_navigation * navScore + 
                       w_collision * (1 - collisionScore) + 
                       w_communication * commScore;
    
    return totalScore;
}

// 使用示例
// int main() {
//     // 创建环境
//     Environment env;
//     env.gridMap = {
//         {0,0,0,0,0,0,0,0,0,0},
//         {0,0,0,0,0,0,0,0,0,0},
//         {0,0,1,1,1,1,1,0,0,0},
//         {0,0,0,0,0,0,0,0,0,0},
//         {0,0,0,0,0,0,0,0,0,0}
//     };
//     env.leaderState = {5.0, 2.0, 0.0};  // x, y, theta
//     env.corridorWidth = 3.0;
//     env.safetyThreshold = 0.5;
//     env.maxCommDistance = 5.0;
    
//     // 创建编队配置
//     FormationConfig formation;
//     formation.controlGraph = {
//         {0,1,0},  // 机器人0控制机器人1
//         {0,0,1},  // 机器人1控制机器人2
//         {0,0,0}   // 机器人2不控制其他
//     };
//     formation.positions = {
//         {0, 0},   // 领航者
//         {1, 0},   // 跟随者1
//         {2, 0}    // 跟随者2
//     };
    
//     // 创建评估器并评估
//     FormationEvaluator evaluator(0.4, 0.3, 0.3);
//     double score = evaluator.evaluateFormation(env, formation);
    
//     // 输出结果
//     std::cout << "Formation Score: " << score << std::endl;
    
//     return 0;
// }
