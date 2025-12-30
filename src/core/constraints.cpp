#include "core/constraints.h"
#include <queue>
#include <algorithm>
#include <sstream>
#include <iterator> // 为 set_symmetric_difference/inserter 提供支持

namespace formation {

ValidationResult ConstraintValidator::validate(const ControlGraph& graph) {
    ValidationResult result;
    // 确保初始为合法状态，后续发现问题再置为 false
    result.is_valid = true;
    result.leader_constraint_violated = false;
    result.label_order_constraint_violated = false;
    result.control_dimension_constraint_violated = false;
    
    // 检查领航机器人存在性约束
    auto leaders = graph.getLeaderRobots();
    if (leaders.empty()) {
        result.is_valid = false;
        result.leader_constraint_violated = true;
        result.details.push_back("违反领航机器人存在性约束：没有领航机器人");
    }
    
    // 检查标签顺序约束
    const auto& edges = graph.getEdges();
    for (const auto& edge : edges) {
        if (edge.first >= edge.second) {
            result.is_valid = false;
            result.label_order_constraint_violated = true;
            std::stringstream ss;
            ss << "违反标签顺序约束：边(" << edge.first << "->" << edge.second << ")中from >= to";
            result.details.push_back(ss.str());
            break;
        }
    }
    
    // 检查控制维度约束
    std::vector<int> out_degree(graph.getNumRobots(), 0);
    for (const auto& edge : edges) {
        out_degree[edge.first]++;
    }
    
    for (int i = 0; i < graph.getNumRobots(); ++i) {
        if (out_degree[i] > 2) {
            result.is_valid = false;
            result.control_dimension_constraint_violated = true;
            std::stringstream ss;
            ss << "违反控制维度约束：机器人" << i << "控制" << out_degree[i] << "个其他机器人（最多允许2个）";
            result.details.push_back(ss.str());
            break;
        }
    }
    
    // 如果所有约束都满足，添加成功信息
    if (result.is_valid) {
        result.details.push_back("所有约束检查通过");
    }
    
    return result;
}

bool ConstraintValidator::validateTransformationStep(const ControlGraph& from, const ControlGraph& to) {
    // 检查两个图的机器人数量是否相同
    if (from.getNumRobots() != to.getNumRobots()) {
        return false;
    }
    
    // 检查目标图是否有效
    if (!to.isValid()) {
        return false;
    }
    
    // 计算变换矩阵
    auto from_matrix = from.getAdjacencyMatrix();
    auto to_matrix = to.getAdjacencyMatrix();
    auto transform_matrix = to_matrix - from_matrix;
    
    return transform_matrix.sum() == 0;
}

std::vector<std::string> ConstraintValidator::getViolationDetails(const ControlGraph& graph) {
    auto result = ConstraintValidator::validate(graph);
    return result.details;
}

bool ConstraintValidator::isConnected(const ControlGraph& graph) {
    int num_robots = graph.getNumRobots();
    if (num_robots == 0) return true;
    
    // 使用BFS检查连通性
    std::vector<bool> visited(num_robots, false);
    std::queue<int> q;
    
    // 从任意一个领航机器人开始
    auto leaders = graph.getLeaderRobots();
    if (leaders.empty()) return false;
    
    int start = *leaders.begin();
    q.push(start);
    visited[start] = true;
    
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        
        // 遍历所有边，找到从当前机器人出发的边
        const auto& edges = graph.getEdges();
        for (const auto& edge : edges) {
            if (edge.first == current && !visited[edge.second]) {
                visited[edge.second] = true;
                q.push(edge.second);
            }
        }
    }
    
    // 检查是否所有机器人都被访问到
    return std::all_of(visited.begin(), visited.end(), [](bool v) { return v; });
}

bool ConstraintValidator::isAcyclic(const ControlGraph& graph) {
    int num_robots = graph.getNumRobots();
    
    // 计算入度
    std::vector<int> in_degree(num_robots, 0);
    const auto& edges = graph.getEdges();
    for (const auto& edge : edges) {
        in_degree[edge.second]++;
    }
    
    // 拓扑排序检查是否有环
    std::queue<int> q;
    std::vector<int> in_degree_copy = in_degree;
    
    // 入度为0的节点入队
    for (int i = 0; i < num_robots; ++i) {
        if (in_degree_copy[i] == 0) {
            q.push(i);
        }
    }
    
    int visited_count = 0;
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        visited_count++;
        
        // 减少邻居的入度
        for (const auto& edge : edges) {
            if (edge.first == current) {
                in_degree_copy[edge.second]--;
                if (in_degree_copy[edge.second] == 0) {
                    q.push(edge.second);
                }
            }
        }
    }
    
    // 如果访问的节点数等于总节点数，说明无环
    return visited_count == num_robots;
}

int ConstraintValidator::calculateDepth(const ControlGraph& graph) {
    int num_robots = graph.getNumRobots();
    if (num_robots == 0) return 0;
    
    // 计算入度
    std::vector<int> in_degree(num_robots, 0);
    const auto& edges = graph.getEdges();
    for (const auto& edge : edges) {
        in_degree[edge.second]++;
    }
    
    // 使用拓扑排序计算最长路径（DAG 上最长路径）
    std::vector<int> distance(num_robots, 0);
    std::queue<int> q;
    std::vector<int> in_degree_copy = in_degree;
    
    // 入度为0的节点入队
    for (int i = 0; i < num_robots; ++i) {
        if (in_degree_copy[i] == 0) {
            q.push(i);
        }
    }
    
    int max_depth = 0;
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        
        for (const auto& edge : edges) {
            if (edge.first == current) {
                int neighbor = edge.second;
                // 更新最长距离
                distance[neighbor] = std::max(distance[neighbor], distance[current] + 1);
                max_depth = std::max(max_depth, distance[neighbor]);
                // 减少入度并在入度为0时入队（拓扑顺序）
                in_degree_copy[neighbor]--;
                if (in_degree_copy[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }
    }
    
    return max_depth;
}

bool ConstraintValidator::canTransformInOneStep(const ControlGraph& graph1, const ControlGraph& graph2) {
    // 检查机器人数量是否相同
    if (graph1.getNumRobots() != graph2.getNumRobots()) {
        return false;
    }
    
    // 检查两个图是否都有效
    if (!graph1.isValid() || !graph2.isValid()) {
        return false;
    }
    
    // 计算边差异
    const auto& edges1 = graph1.getEdges();
    const auto& edges2 = graph2.getEdges();
    
    // 找出不同的边
    std::set<std::pair<int, int>> diff;//diff1.first是diff1.second的领航机器人
    std::set_symmetric_difference(  //所有只在edges1中出现，或只在edges2中出现的边
        edges1.begin(), edges1.end(),
        edges2.begin(), edges2.end(),
        std::inserter(diff, diff.begin())
    );
    
    if(diff.size() != 2) return false;
    auto it = diff.begin();
    std::pair<int,int> edge1 = *it;
    std::pair<int,int> edge2 = *(++it);

    if( edge1.second != edge2.second) return false;       

    // 单步变换应该只改变一条边
    return true;
}

} // namespace formation