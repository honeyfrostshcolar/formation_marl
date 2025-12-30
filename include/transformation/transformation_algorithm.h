#pragma once

#include "core/control_graph.h"
#include "core/constraints.h"
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace formation {

// 变换操作类型枚举
enum class TransformationType {
    ADD_EDGE,      // 添加边
    REMOVE_EDGE,   // 移除边
    SWAP_EDGE,     // 交换边端点（拓扑有边 A→B 和 C→D，交换后变为 A→D 和 C→B ）
    MERGE_NODES,   // 合并节点（对应「逻辑组队/子编队」）
    SPLIT_NODE     // 分裂节点（对应「子编队拆分/队形扩展」）
};

// 变换操作结构体
struct TransformationOperation {
    TransformationType type;
    int robot1;           // 操作涉及的第一个机器人
    int robot2;           // 操作涉及的第二个机器人
    double cost;          // 变换成本
    std::string description; // 操作描述
    
    TransformationOperation(TransformationType t, int r1, int r2, double c = 1.0)
        : type(t), robot1(r1), robot2(r2), cost(c) {
        updateDescription();
    }

    TransformationOperation() = default;
    
private:
    void updateDescription();
};

// 变换路径结构体
struct TransformationPath {

    // 现有构造函数
    TransformationPath(const ControlGraph& start, const ControlGraph& target)
        : start_graph(start), target_graph(target), is_valid(false), total_cost(0.0) {}

    TransformationPath(const TransformationPath& other) = default;

    ControlGraph start_graph;                    // 起始队形
    ControlGraph target_graph;                   // 目标队形
    std::vector<TransformationOperation> operations; // 变换操作序列
    bool is_valid;                               // 路径是否有效
    double total_cost;                           // 总成本
};

// 变换算法类
class TransformationAlgorithm {
public:
    TransformationAlgorithm();
    
    // 基础变换方法
    TransformationPath findTransformationPath(const ControlGraph& start, 
                                             const ControlGraph& target);
    
    // 优化变换方法
    TransformationPath findOptimalPath(const ControlGraph& start, 
                                      const ControlGraph& target,
                                      double max_cost = 100.0);
    
    // 多目标变换
    std::vector<TransformationPath> findMultiplePaths(
        const ControlGraph& start, 
        const ControlGraph& target,
        int max_paths = 5);
    
    // 变换验证
    bool validateTransformation(const TransformationOperation& op, 
                               const ControlGraph& current_graph);
    
    // 成本计算
    double calculateOperationCost(const TransformationOperation& op, 
                                 const ControlGraph& graph);
    
    // 设置成本函数
    void setCostFunction(std::function<double(const TransformationOperation&, 
                                             const ControlGraph&)> cost_func);
    
    // 性能统计
    struct PerformanceStats {
        int total_operations;      // 总操作数
        int valid_operations;     // 有效操作数
        double average_cost;      // 平均成本
        double search_time;       // 搜索时间
        int explored_states;      // 探索的状态数
    };
    
    PerformanceStats getPerformanceStats() const;
    
    // 设置搜索参数
    void setSearchParameters(int max_depth = 10, int max_states = 1000);

    // 展示变换过程
    void displayTransformationPath(const TransformationPath& path) const;
    
private:
    // 内部搜索算法
    TransformationPath breadthFirstSearch(const ControlGraph& start, 
                                         const ControlGraph& target);
    TransformationPath aStarSearch(const ControlGraph& start, 
                                   const ControlGraph& target);
    TransformationPath beamSearch(const ControlGraph& start, 
                                 const ControlGraph& target);
    
    // 启发式函数
    double heuristic(const ControlGraph& current, const ControlGraph& target);
    
    // 生成候选操作
    std::vector<TransformationOperation> generateCandidateOperations(
        const ControlGraph& graph);
    
    // 应用变换操作
    ControlGraph applyOperation(const ControlGraph& graph, 
                              const TransformationOperation& op);
    
    // 状态比较
    bool areGraphsEquivalent(const ControlGraph& g1, const ControlGraph& g2);
    
    // 成员变量
    std::function<double(const TransformationOperation&, const ControlGraph&)> cost_function_;
    int max_search_depth_;
    int max_explored_states_;
    PerformanceStats stats_;
};

// 变换管理器类
class TransformationManager {
public:
    TransformationManager();
    
    // 单步变换
    bool performSingleTransformation(ControlGraph& graph, 
                                     TransformationType type, 
                                     int robot1, int robot2);
    
    // 批量变换
    bool performTransformationSequence(ControlGraph& graph,
                                      const std::vector<TransformationOperation>& operations);
    
    // 撤销操作
    bool undoLastTransformation(ControlGraph& graph);
    
    // 重做操作
    bool redoLastTransformation(ControlGraph& graph);
    
    // 获取历史记录
    std::vector<TransformationOperation> getTransformationHistory() const;
    
    // 清除历史记录
    void clearHistory();
    
private:
    std::vector<ControlGraph> history_stack_;
    std::vector<TransformationOperation> operation_history_;
    int current_state_index_;
};

// 变换优化器类
class TransformationOptimizer {
public:
    TransformationOptimizer();
    
    // 路径优化
    TransformationPath optimizePath(const TransformationPath& original_path);
    
    // 成本优化
    TransformationPath findMinimumCostPath(const ControlGraph& start, 
                                          const ControlGraph& target);
    
    // 步骤优化
    TransformationPath findMinimumStepPath(const ControlGraph& start, 
                                          const ControlGraph& target);
    
    // 多目标优化
    TransformationPath findBalancedPath(const ControlGraph& start, 
                                       const ControlGraph& target,
                                       double cost_weight = 0.5,
                                       double step_weight = 0.5);
    
private:
    // 优化算法
    TransformationPath localOptimization(const TransformationPath& path);
    TransformationPath geneticOptimization(const TransformationPath& path);
    TransformationPath simulatedAnnealing(const TransformationPath& path);
};

} // namespace formation