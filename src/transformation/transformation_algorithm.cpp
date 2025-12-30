#include "transformation/transformation_algorithm.h"
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <chrono>
#include <functional>
#include <cmath>
#include <unordered_map>
#include <iterator>
#include <iostream>

namespace formation {

void TransformationOperation::updateDescription() {
    switch (type) {
        case TransformationType::ADD_EDGE:
            description = "添加边: 机器人" + std::to_string(robot1) + " → 机器人" + std::to_string(robot2);
            break;
        case TransformationType::REMOVE_EDGE:
            description = "移除边: 机器人" + std::to_string(robot1) + " → 机器人" + std::to_string(robot2);
            break;
        case TransformationType::SWAP_EDGE:
            description = "交换边: 机器人" + std::to_string(robot1) + " ↔ 机器人" + std::to_string(robot2);
            break;
        case TransformationType::MERGE_NODES:
            description = "合并节点: 机器人" + std::to_string(robot1) + " + 机器人" + std::to_string(robot2);
            break;
        case TransformationType::SPLIT_NODE:
            description = "分裂节点: 机器人" + std::to_string(robot1) + " → 新机器人";
            break;
        default:
            description = "未知操作";
    }
}

TransformationAlgorithm::TransformationAlgorithm() 
    : max_search_depth_(10), max_explored_states_(1000) {
    
    // 默认成本函数
    cost_function_ = [](const TransformationOperation& op, const ControlGraph& graph) {
        return op.cost;
    };
    
    // 初始化统计信息
    stats_ = {0, 0, 0.0, 0.0, 0};
}

TransformationPath TransformationAlgorithm::findTransformationPath(
    const ControlGraph& start, const ControlGraph& target) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 使用A*搜索算法
    TransformationPath path = aStarSearch(start, target);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    stats_.search_time = duration.count() / 1000.0;
    
    return path;
}

TransformationPath TransformationAlgorithm::findOptimalPath(
    const ControlGraph& start, const ControlGraph& target, double max_cost) {
    
    auto path = findTransformationPath(start, target);
    
    if (path.total_cost > max_cost) {
        path.is_valid = false;
    }
    
    return path;
}

std::vector<TransformationPath> TransformationAlgorithm::findMultiplePaths(
    const ControlGraph& start, const ControlGraph& target, int max_paths) {
    
    std::vector<TransformationPath> paths;
    
    // 使用波束搜索获取多条路径
    TransformationPath main_path = beamSearch(start, target);
    if (main_path.is_valid) {
        paths.push_back(main_path);
    }
    
    // 这里可以添加其他搜索策略来获取更多路径
    // 例如：随机搜索、遗传算法等
    
    return paths;
}

// 验证变换操作是否合法
bool TransformationAlgorithm::validateTransformation(
    const TransformationOperation& op, const ControlGraph& current_graph) {
    
    stats_.total_operations++;
    
    // 检查操作是否在有效范围内
    int num_robots = current_graph.getNumRobots();
    if (op.robot1 < 0 || op.robot1 >= num_robots || 
        op.robot2 < 0 || op.robot2 >= num_robots) {
        return false;
    }
    
    // 根据操作类型进行验证
    switch (op.type) {
        case TransformationType::ADD_EDGE:
            // 必须满足标签顺序约束：from < to
            if (op.robot1 >= op.robot2) {
                return false;
            }
            // 检查边是否已存在
            if (current_graph.hasEdge(op.robot1, op.robot2)) {
                return false;
            }
            // 检查出度限制（最多允许2个）
            {
                std::vector<int> out_degree(num_robots, 0);
                const auto& edges = current_graph.getEdges();
                for (const auto& e : edges) out_degree[e.first]++;
                if (out_degree[op.robot1] >= 2) {
                    return false;
                }
            }
            break;
            
        case TransformationType::REMOVE_EDGE:
            // 检查边是否存在
            if (!current_graph.hasEdge(op.robot1, op.robot2)) {
                return false;
            }
            break;
            
        // case TransformationType::SWAP_EDGE:
        //     // 需要两条不同的节点进行交换，且涉及的边应存在/合法
        //     if (op.robot1 == op.robot2) {
        //         return false;
        //     }
        //     // 简单检查：至少其中一条边存在（更复杂的逻辑可按需扩展）
        //     if (!current_graph.hasEdge(op.robot1, op.robot2) && !current_graph.hasEdge(op.robot2, op.robot1)) {
        //         return false;
        //     }
        //     break;
            
        // case TransformationType::MERGE_NODES:
        // case TransformationType::SPLIT_NODE:
        //     // 这些操作需要更复杂的验证
        //     // 暂时允许，但上层应谨慎使用
        //     break;
    }
    
    stats_.valid_operations++;
    return true;
}

double TransformationAlgorithm::calculateOperationCost(
    const TransformationOperation& op, const ControlGraph& graph) { //图为目前图，操作为对他的操作
    
    return cost_function_(op, graph);
}

void TransformationAlgorithm::setCostFunction(
    std::function<double(const TransformationOperation&, const ControlGraph&)> cost_func) {
    
    cost_function_ = cost_func;
}

TransformationAlgorithm::PerformanceStats TransformationAlgorithm::getPerformanceStats() const {
    return stats_;
}

void TransformationAlgorithm::setSearchParameters(int max_depth, int max_states) {
    max_search_depth_ = max_depth;
    max_explored_states_ = max_states;
}

// ControlGraph 哈希/相等适配器，供 unordered_map 使用
struct ControlGraphHasher {
    size_t operator()(const ControlGraph& g) const noexcept {
        return g.hash();
    }
};
struct ControlGraphEqual {
    bool operator()(const ControlGraph& a, const ControlGraph& b) const noexcept {
        return a == b;
    }
};

// A*搜索算法实现
TransformationPath TransformationAlgorithm::aStarSearch(
    const ControlGraph& start, const ControlGraph& target) {
    
    TransformationPath path(start, target);
    
    // 使用优先队列进行A*搜索
    using SearchNode = std::pair<double, ControlGraph>; // <f值, 图状态>
    struct NodeCompare {
        bool operator()(const SearchNode& a, const SearchNode& b) const {
            return a.first > b.first; // 小的 f 值优先
        }
    };
    std::priority_queue<SearchNode, std::vector<SearchNode>, NodeCompare> open_set;
    
    // 使用 unordered_map 代替 map ，需要哈希与相等函数
    std::unordered_map<ControlGraph, double, ControlGraphHasher, ControlGraphEqual> g_score; // 从起点到当前节点的实际成本
    std::unordered_map<ControlGraph, ControlGraph, ControlGraphHasher, ControlGraphEqual> came_from; // 记录路径
    std::unordered_map<ControlGraph, TransformationOperation, ControlGraphHasher, ControlGraphEqual> operation_used; // 使用的操作
    
    // 初始化
    g_score[start] = 0.0;
    open_set.push({heuristic(start, target), start});
    
    int states_explored = 0;
    
    while (!open_set.empty() && states_explored < max_explored_states_) {
        auto current = open_set.top().second;
        open_set.pop();
        states_explored++;
        
        // 检查是否到达目标
        if (areGraphsEquivalent(current, target)) {
            // 重建路径
            path.is_valid = true;
            ControlGraph temp = current;
            
            while (came_from.find(temp) != came_from.end()) {
                path.operations.push_back(operation_used[temp]);
                path.total_cost += operation_used[temp].cost;
                temp = came_from[temp];
            }
            
            std::reverse(path.operations.begin(), path.operations.end());
            break;
        }
        
        // 生成候选操作
        auto candidates = generateCandidateOperations(current);
        
        for (const auto& op : candidates) {
            if (validateTransformation(op, current)) {
                ControlGraph neighbor = applyOperation(current, op);
                
                double tentative_g_score = g_score[current] + calculateOperationCost(op, current);
                
                auto it = g_score.find(neighbor);
                if (it == g_score.end() || tentative_g_score < it->second) {
                    
                    came_from[neighbor] = current;
                    operation_used[neighbor] = op;
                    g_score[neighbor] = tentative_g_score;
                    double f_score = tentative_g_score + heuristic(neighbor, target);
                    open_set.push({f_score, neighbor});
                }
            }
        }
    }
    
    stats_.explored_states = states_explored;
    return path;
}

void TransformationAlgorithm::displayTransformationPath(const TransformationPath& path) const {
    std::cout << "变换路径：" << std::endl;
    std::cout << "起始队形：" << std::endl;
    std::cout << path.start_graph.toString() << std::endl;
    for (size_t i = 0; i < path.operations.size(); ++i) {
        std::cout << "步骤 " << i + 1 << ": " << path.operations[i].description 
                  << " (成本: " << path.operations[i].cost << ")" << std::endl;
    }
    std::cout << "目标队形：" << std::endl;
    std::cout << path.target_graph.toString() << std::endl;
    std::cout << "总成本: " << path.total_cost << std::endl;
    std::cout << "路径有效性: " << (path.is_valid ? "有效" : "无效") << std::endl;
}

// 波束搜索实现
TransformationPath TransformationAlgorithm::beamSearch(
    const ControlGraph& start, const ControlGraph& target) {
    
    // 简化的波束搜索实现
    // 实际实现需要更复杂的波束宽度管理
    return aStarSearch(start, target);
}

// 启发式函数
double TransformationAlgorithm::heuristic(const ControlGraph& current, const ControlGraph& target) {
    // 基于图差异的启发式函数
    double diff = 0.0;
    
    // 比较边集合
    auto current_edges = current.getEdges();
    auto target_edges = target.getEdges();
    
    // 计算边差异
    std::set<std::pair<int, int>> current_set(current_edges.begin(), current_edges.end());
    std::set<std::pair<int, int>> target_set(target_edges.begin(), target_edges.end());
    
    std::vector<std::pair<int, int>> symmetric_diff;
    std::set_symmetric_difference(current_set.begin(), current_set.end(),
                                 target_set.begin(), target_set.end(),
                                 std::back_inserter(symmetric_diff));
    
    diff += symmetric_diff.size() * 1.0; // 每条差异边的成本
    
    return diff;
}

// 生成候选操作
std::vector<TransformationOperation> TransformationAlgorithm::generateCandidateOperations(
    const ControlGraph& graph) {
    
    std::vector<TransformationOperation> candidates;
    int num_robots = graph.getNumRobots();
    
    // 生成所有可能的添加边操作：只考虑 i < j（符合标签顺序约束）
    for (int i = 0; i < num_robots; ++i) {
        for (int j = i + 1; j < num_robots; ++j) {
            if (!graph.hasEdge(i, j)) {
                candidates.emplace_back(TransformationType::ADD_EDGE, i, j);
            }
        }
    }
    
    // 生成所有可能的移除边操作
    auto edges = graph.getEdges();
    for (const auto& edge : edges) {
        candidates.emplace_back(TransformationType::REMOVE_EDGE, edge.first, edge.second);
    }
    
    // 注意：不再自动生成 REVERSE_EDGE（在标签顺序约束下通常无效）
    
    return candidates;
}

// 应用变换操作
ControlGraph TransformationAlgorithm::applyOperation(
    const ControlGraph& graph, const TransformationOperation& op) {
    
    ControlGraph result = graph;
    
    switch (op.type) {
        case TransformationType::ADD_EDGE:
            result.addEdge(op.robot1, op.robot2);
            break;
            
        case TransformationType::REMOVE_EDGE:
            result.removeEdge(op.robot1, op.robot2);
            break;
            
        case TransformationType::SWAP_EDGE:
            // 简化实现：交换两条边
            // 实际实现需要更复杂的逻辑
            break;
            
        case TransformationType::MERGE_NODES:
        case TransformationType::SPLIT_NODE:
            // 这些操作需要更复杂的实现
            break;
    }
    
    return result;
}

// 图等价性检查
bool TransformationAlgorithm::areGraphsEquivalent(const ControlGraph& g1, const ControlGraph& g2) {
    // 简化的等价性检查
    // 实际实现需要更复杂的图同构算法
    
    if (g1.getNumRobots() != g2.getNumRobots()) {
        return false;
    }
    
    auto edges1 = g1.getEdges();
    auto edges2 = g2.getEdges();
    
    if (edges1.size() != edges2.size()) {
        return false;
    }
    
    std::set<std::pair<int, int>> set1(edges1.begin(), edges1.end());
    std::set<std::pair<int, int>> set2(edges2.begin(), edges2.end());
    
    return set1 == set2;
}

// 不同图所确定到的队形（在图中添加距离这个额外信息）
// void TransformationAlgorithm::generaterPosForGraph(ControlGraph& graph, const Formation& formation) {
    
// }


// TransformationManager 实现
TransformationManager::TransformationManager() : current_state_index_(-1) {}

bool TransformationManager::performSingleTransformation(
    ControlGraph& graph, TransformationType type, int robot1, int robot2) {
    
    TransformationOperation op(type, robot1, robot2);
    
    // 保存当前状态
    history_stack_.push_back(graph);
    operation_history_.push_back(op);
    current_state_index_ = history_stack_.size() - 1;
    
    // 应用变换
    // 这里需要调用TransformationAlgorithm来应用操作
    // 简化实现：直接修改图
    
    switch (type) {
        case TransformationType::ADD_EDGE:
            return graph.addEdge(robot1, robot2);
        case TransformationType::REMOVE_EDGE:
            return graph.removeEdge(robot1, robot2);
        default:
            return false;
    }
}

bool TransformationManager::performTransformationSequence(
    ControlGraph& graph, const std::vector<TransformationOperation>& operations) {
    
    for (const auto& op : operations) {
        if (!performSingleTransformation(graph, op.type, op.robot1, op.robot2)) {
            return false;
        }
    }
    
    return true;
}

bool TransformationManager::undoLastTransformation(ControlGraph& graph) {
    if (current_state_index_ > 0) {
        current_state_index_--;
        graph = history_stack_[current_state_index_];
        return true;
    }
    return false;
}

bool TransformationManager::redoLastTransformation(ControlGraph& graph) {
    if (current_state_index_ < static_cast<int>(history_stack_.size()) - 1) {
        current_state_index_++;
        graph = history_stack_[current_state_index_];
        return true;
    }
    return false;
}

std::vector<TransformationOperation> TransformationManager::getTransformationHistory() const {
    return operation_history_;
}

void TransformationManager::clearHistory() {
    history_stack_.clear();
    operation_history_.clear();
    current_state_index_ = -1;
}

// TransformationOptimizer 实现
TransformationOptimizer::TransformationOptimizer() {}

TransformationPath TransformationOptimizer::optimizePath(const TransformationPath& original_path) {
    // 简化实现：返回原路径
    return original_path;
}

TransformationPath TransformationOptimizer::findMinimumCostPath(
    const ControlGraph& start, const ControlGraph& target) {
    
    TransformationAlgorithm algorithm;
    return algorithm.findTransformationPath(start, target);
}

TransformationPath TransformationOptimizer::findMinimumStepPath(
    const ControlGraph& start, const ControlGraph& target) {
    
    TransformationAlgorithm algorithm;
    // 设置成本函数，使所有操作成本相同
    algorithm.setCostFunction([](const TransformationOperation& op, const ControlGraph& graph) {
        return 1.0; // 所有操作成本为1
    });
    
    return algorithm.findTransformationPath(start, target);
}

TransformationPath TransformationOptimizer::findBalancedPath(
    const ControlGraph& start, const ControlGraph& target,
    double cost_weight, double step_weight) {
    
    // 简化实现：返回最小成本路径
    return findMinimumCostPath(start, target);
}

} // namespace formation