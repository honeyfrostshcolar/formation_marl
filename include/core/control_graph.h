#ifndef FORMATION_CONTROL_CONTROL_GRAPH_H
#define FORMATION_CONTROL_CONTROL_GRAPH_H

#include <vector>
#include <set>
#include <string>
#include <Eigen/Dense>

namespace formation {

/**
 * @brief 控制图类，表示机器人编队的控制关系
 * 
 * 控制图是一个有向图，其中节点表示机器人，边表示控制关系。
 * 必须满足三个约束条件：
 * 1. 存在至少一个领航机器人（入度为0）
 * 2. 标签顺序约束：边只能从小编号指向大编号
 * 3. 控制维度约束：每个机器人最多控制2个其他机器人
 */
class ControlGraph {
public:
    /**
     * @brief 构造函数
     * @param num_robots 机器人数量
     */
    explicit ControlGraph(int num_robots);

    ControlGraph() : num_robots_(0) {}  // 默认初始化机器人数量为 0
    
    /**
     * @brief 添加控制边
     * @param from 控制机器人编号
     * @param to 被控制机器人编号
     * @return 是否添加成功
     */
    bool addEdge(int from, int to);
    
    /**
     * @brief 移除控制边
     * @param from 控制机器人编号
     * @param to 被控制机器人编号
     * @return 是否移除成功
     */
    bool removeEdge(int from, int to);
    
    /**
     * @brief 获取邻接矩阵
     * @return 邻接矩阵
     */
    Eigen::MatrixXi getAdjacencyMatrix() const;
    
    /**
     * @brief 从邻接矩阵设置控制图
     * @param matrix 邻接矩阵
     * @return 是否设置成功
     */
    bool setFromAdjacencyMatrix(const Eigen::MatrixXi& matrix);
    
    /**
     * @brief 检查控制图是否有效（满足三约束）
     * @return 是否有效
     */
    bool isValid() const;
    
    /**
     * @brief 获取机器人数量
     * @return 机器人数量
     */
    int getNumRobots() const;
    
    /**
     * @brief 获取所有边（返回 const 引用，避免不必要拷贝）
     * @return 边集合
     */
    const std::set<std::pair<int, int>>& getEdges() const;
    
    /**
     * @brief 判断是否存在某条边
     */
    bool hasEdge(int from, int to) const;
    
    /**
     * @brief 判断某个机器人是否为领航机器人（入度为0）
     */
    bool isLeaderRobot(int id) const;
    
    /**
     * @brief 获取领航机器人（入度为0的机器人）
     * @return 领航机器人编号集合
     */
    std::set<int> getLeaderRobots() const;
    
    /**
     * @brief 获取字符串表示
     * @return 控制图的字符串表示
     */
    std::string toString() const;
    
    /**
     * @brief 计算控制图的哈希值
     * @return 哈希值
     */
    size_t hash() const;
    
    /**
     * @brief 比较两个控制图是否相等
     * @param other 另一个控制图
     * @return 是否相等
     */
    bool operator==(const ControlGraph& other) const;

private:
    int num_robots_;  // 机器人数量
    std::set<std::pair<int, int>> edges_;  // 边集合
    
    /**
     * @brief 检查标签顺序约束
     * @return 是否满足约束
     */
    bool checkLabelOrderConstraint() const;
    
    /**
     * @brief 检查控制维度约束
     * @return 是否满足约束
     */
    bool checkControlDimensionConstraint() const;
    
    /**
     * @brief 检查领航机器人存在性约束
     * @return 是否满足约束
     */
    bool checkLeaderExistenceConstraint() const;
};

} // namespace formation

#endif // FORMATION_CONTROL_CONTROL_GRAPH_H