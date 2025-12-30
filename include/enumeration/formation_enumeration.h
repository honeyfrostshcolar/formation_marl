#ifndef FORMATION_CONTROL_FORMATION_ENUMERATION_H
#define FORMATION_CONTROL_FORMATION_ENUMERATION_H

#include "core/control_graph.h"
#include <vector>
#include <set>
#include <functional>
#include <memory>
#include <map> 

namespace formation {

/**
 * @brief 性能数据结构体（提前定义，供 FormationEnumerator 使用）
 */
struct PerformanceData {
    int total_formations;           // 总队形数量
    int valid_formations;          // 有效队形数量
    double enumeration_time;      // 枚举时间（秒）
    size_t memory_usage;           // 内存使用（字节）
    std::map<std::string, int> type_counts; // 各类队形数量
    
    PerformanceData() 
        : total_formations(0)
        , valid_formations(0)
        , enumeration_time(0.0)
        , memory_usage(0) {}
};

/**
 * @brief 队形枚举器类，基于Polya定理实现有向图枚举
 * 
 * 按照论文中的算法，枚举所有满足三约束条件的控制图（队形）
 * 支持n=6个机器人的完整枚举和分类
 */
class FormationEnumerator {
public:
    /**
     * @brief 构造函数
     * @param num_robots 机器人数量
     */
    explicit FormationEnumerator(int num_robots);
    
    /**
     * @brief 枚举所有有效的控制图（队形）
     * @return 所有有效控制图的集合
     */
    std::vector<ControlGraph> enumerateAllFormations();
    
    /**
     * @brief 枚举特定类型的队形
     * @param formation_type 队形类型
     * @return 特定类型的控制图集合
     */
    std::vector<ControlGraph> enumerateByType(const std::string& formation_type);
    
    /**
     * @brief 获取队形统计信息
     * @return 各类队形的数量统计
     */
    std::map<std::string, int> getFormationStatistics();
    
    /**
     * @brief 分类控制图
     * @param graph 控制图
     * @return 队形类型标识
     */
    std::string classifyFormation(const ControlGraph& graph);
    
    /**
     * @brief 获取所有可能的队形类型
     * @return 队形类型列表
     */
    std::vector<std::string> getAllFormationTypes() const;
    
    /**
     * @brief 设置枚举回调函数（用于进度监控）
     * @param callback 回调函数
     */
    void setProgressCallback(std::function<void(int, int)> callback);

    int getControlGraghNums();
    
    /**
     * @brief 获取枚举性能数据
     * @return 性能数据（枚举时间、内存使用等）
     */
    PerformanceData getPerformanceData() const;

private:
    int num_robots_;
    int num_graghs_;
    std::vector<std::pair<int, int>> edge_mapping_;
    std::function<void(int, int)> progress_callback_;
    
    /**
     * @brief 递归生成所有可能的有向图
     * @param current_graph 当前控制图
     * @param edge_index 当前处理的边索引
     * @param results 结果集合
     */
    void generateAllDirectedGraphs(ControlGraph& current_graph, 
                                  int edge_index, 
                                  std::vector<ControlGraph>& results);
    
    /**
     * @brief 应用Polya定理进行图枚举
     * @return 所有非同构图
     */
    std::vector<ControlGraph> applyPolyaTheorem();
    
    /**
     * @brief 计算图的同构类
     * @param graphs 图集合
     * @return 同构类划分
     */
    std::vector<std::vector<ControlGraph>> computeIsomorphismClasses(
        const std::vector<ControlGraph>& graphs);
    
    /**
     * @brief 检查两个控制图是否同构
     * @param graph1 第一个控制图
     * @param graph2 第二个控制图
     * @return 是否同构
     */
    bool areIsomorphic(const ControlGraph& graph1, const ControlGraph& graph2);
    
    /**
     * @brief 基于拓扑结构分类队形
     * @param graph 控制图
     * @return 队形类型
     */
    std::string classifyByTopology(const ControlGraph& graph);
    
    /**
     * @brief 基于相对位置关系分类队形
     * @param graph 控制图
     * @return 队形类型
     */
    std::string classifyByRelativePositions(const ControlGraph& graph);

    void initializeEdgeMapping(int num_robots);

    bool checkPermutation(const Eigen::MatrixXi& adj1,
                            const Eigen::MatrixXi& adj2,
                            const std::vector<int>& permutation);
};

/**
 * @brief 队形类型分类器
 */
class FormationClassifier {
public:
    /**
     * @brief 分类控制图
     * @param graph 控制图
     * @return 队形类型
     */
    static std::string classify(const ControlGraph& graph);
    
    /**
     * @brief 获取队形的拓扑特征
     * @param graph 控制图
     * @return 拓扑特征向量
     */
    static std::vector<int> getTopologicalFeatures(const ControlGraph& graph);
    
    /**
     * @brief 获取队形的度序列
     * @param graph 控制图
     * @return 度序列
     */
    static std::vector<int> getDegreeSequence(const ControlGraph& graph);
    
    /**
     * @brief 计算图的连通分量
     * @param graph 控制图
     * @return 连通分量大小
     */
    static std::vector<int> getConnectedComponents(const ControlGraph& graph);
};

} // namespace formation

#endif // FORMATION_CONTROL_FORMATION_ENUMERATION_H