#include "enumeration/formation_enumeration.h"
#include "core/constraints.h"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <queue>
#include <functional>

namespace formation {

FormationEnumerator::FormationEnumerator(int num_robots) 
    : num_robots_(num_robots) {
    if (num_robots <= 0) {
        throw std::invalid_argument("机器人数量必须大于0");
    }

    initializeEdgeMapping(num_robots_);
}

// 枚举所有可能的控制图
std::vector<ControlGraph> FormationEnumerator::enumerateAllFormations() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<ControlGraph> all_graphs;
    
    // 生成所有可能的有向图
    ControlGraph initial_graph(num_robots_);
    generateAllDirectedGraphs(initial_graph, 0, all_graphs);
    
    // for(const auto& form : all_graphs){
    //     std::cout << form.getAdjacencyMatrix() << std::endl;
    //     std::cout<<"--------------"<< std::endl;
    // }

    // 过滤有效的控制图
    std::vector<ControlGraph> valid_graphs;
    for (const auto& graph : all_graphs) {
        if (graph.isValid()) {
            valid_graphs.push_back(graph);
        }
    }
    
    // 应用Polya定理进行同构类划分
    std::vector<std::vector<ControlGraph>> isomorphism_classes = computeIsomorphismClasses(valid_graphs);
    
    // 从每个同构类中选择一个代表
    std::vector<ControlGraph> representatives;
    for (const auto& class_graphs : isomorphism_classes) {
        if (!class_graphs.empty()) {
            representatives.push_back(class_graphs[0]);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end_time - start_time;
    //std::cout << "枚举所有队形耗时: " << duration.count() << " 秒" << std::endl;

    // 调用进度回调
    if (progress_callback_) {
        progress_callback_(representatives.size(), representatives.size());
    }

    num_graghs_ = representatives.size();
    
    return representatives;
}

// 按队形类型枚举
std::vector<ControlGraph> FormationEnumerator::enumerateByType(const std::string& formation_type) {
    auto all_formations = enumerateAllFormations();
    std::vector<ControlGraph> filtered_formations;
    
    for (const auto& formation : all_formations) {
        if (classifyFormation(formation) == formation_type) {
            filtered_formations.push_back(formation);
        }
    }
    
    return filtered_formations;
}

// 统计所有队形类型的数量
std::map<std::string, int> FormationEnumerator::getFormationStatistics() {
    auto all_formations = enumerateAllFormations();
    std::map<std::string, int> statistics;
    
    for (const auto& formation : all_formations) {
        std::string type = classifyFormation(formation);
        statistics[type]++;
    }
    
    return statistics;
}

int FormationEnumerator::getControlGraghNums(){

    return num_graghs_;

}

std::string FormationEnumerator::classifyFormation(const ControlGraph& graph) {
    // 优先使用拓扑分类
    std::string topology_type = classifyByTopology(graph);
    
    // 如果需要更细粒度的分类，可以结合相对位置关系
    if (topology_type == "复杂结构") {
        std::string position_type = classifyByRelativePositions(graph);
        return topology_type + "-" + position_type;
    }
    
    return topology_type;
}

std::vector<std::string> FormationEnumerator::getAllFormationTypes() const {
    return {
        "链式结构",
        "树状结构", 
        "星型结构",
        "环形结构",
        "复杂结构"
    };
}

void FormationEnumerator::setProgressCallback(std::function<void(int, int)> callback) {
    progress_callback_ = callback;
}

// 获取性能数据（模拟实现）
PerformanceData FormationEnumerator::getPerformanceData() const {
    // 这里返回模拟数据，实际实现中需要记录真实性能数据
    PerformanceData data;
    data.total_formations = 100;  // 模拟值
    data.valid_formations = 50;   // 模拟值
    data.enumeration_time = 1.5; // 模拟值
    data.memory_usage = 1024 * 1024; // 模拟值
    
    return data;
}

void FormationEnumerator::generateAllDirectedGraphs(ControlGraph& current_graph, 
                                                   int edge_index, 
                                                   std::vector<ControlGraph>& results) {
    int num_robots = current_graph.getNumRobots();
    
    // 计算所有可能的边数量
    int max_edges = num_robots * (num_robots - 1) / 2;
    
    if (edge_index >= max_edges ) {
        // 到达叶子节点，保存当前图
        if(current_graph.isValid()){
            results.push_back(current_graph);
        }
        
        return;
    }
    
    int from, to;
    from = edge_mapping_[edge_index].first;
    to = edge_mapping_[edge_index].second;
    
    // 分支1：不添加这条边（是添加边，不是保存图）
    generateAllDirectedGraphs(current_graph, edge_index + 1, results);
    
    // 分支2：添加这条边（如果有效）
    ControlGraph new_graph = current_graph;

    std::cout << "欲添加边: " << from << " -> " << to << std::endl;
    if (new_graph.addEdge(from, to)) {
        std::cout << "成功添加边: " << from << " -> " << to << std::endl;
        generateAllDirectedGraphs(new_graph, edge_index + 1, results);
    }

}

std::vector<ControlGraph> FormationEnumerator::applyPolyaTheorem() {
    // 简化的Polya定理实现
    // 实际实现需要更复杂的群论计算
    
    std::vector<ControlGraph> all_graphs;
    ControlGraph initial_graph(num_robots_);
    generateAllDirectedGraphs(initial_graph, 0, all_graphs);
    
    // 过滤有效的图
    std::vector<ControlGraph> valid_graphs;
    for (const auto& graph : all_graphs) {
        if (graph.isValid()) {
            valid_graphs.push_back(graph);
        }
    }
    
    std::vector<std::vector<ControlGraph>> isomorphism_classes = computeIsomorphismClasses(valid_graphs); // 计算同构类
    std::vector<ControlGraph> representatives;
    for (const auto& class_graphs : isomorphism_classes) {
        if(!class_graphs.empty()){
           // 选择第一个作为代表
        representatives.push_back(class_graphs[0]); 
        } 
    }

    return representatives;
}

std::vector<std::vector<ControlGraph>> FormationEnumerator::computeIsomorphismClasses(
    const std::vector<ControlGraph>& graphs) {
    
    std::vector<std::vector<ControlGraph>> classes;
    std::vector<bool> assigned(graphs.size(), false);
    
    for (size_t i = 0; i < graphs.size(); ++i) {
        if (!assigned[i]) {
            std::vector<ControlGraph> current_class;
            current_class.push_back(graphs[i]);
            assigned[i] = true;
            
            for (size_t j = i + 1; j < graphs.size(); ++j) {
                if (!assigned[j] && areIsomorphic(graphs[i], graphs[j])) {
                    current_class.push_back(graphs[j]);
                    assigned[j] = true;
                }
            }
            
            classes.push_back(current_class);
        }
    }
    
    return classes;
}

bool FormationEnumerator::areIsomorphic(const ControlGraph& graph1, const ControlGraph& graph2) {
    int n = graph1.getNumRobots();
    if (n != graph2.getNumRobots()) {
        return false;
    }
    
    // 1. 快速检查：边数不同肯定不同构
    if (graph1.getEdges().size() != graph2.getEdges().size()) {
        return false;
    }
    
    // 2. 检查入度序列和出度序列（分别检查，不合并）

    std::vector<int> in_degree1(n, 0), out_degree1(n, 0);
    std::vector<int> in_degree2(n, 0), out_degree2(n, 0);
    for (const auto& edge : graph1.getEdges()) {
        out_degree1[edge.first]++;
        in_degree1[edge.second]++;
    }
    for (const auto& edge : graph2.getEdges()) {
        out_degree2[edge.first]++;
        in_degree2[edge.second]++;
    }
   
    std::sort(in_degree1.begin(), in_degree1.end());
    std::sort(in_degree2.begin(), in_degree2.end());
    std::sort(out_degree1.begin(), out_degree1.end());
    std::sort(out_degree2.begin(), out_degree2.end());
    
    if (in_degree1 != in_degree2 || out_degree1 != out_degree2) {
        return false;
    }
    
    // 3. 获取邻接矩阵并尝试所有排列
    //邻接矩阵
    Eigen::MatrixXi adj1 = graph1.getAdjacencyMatrix();
    Eigen::MatrixXi adj2 = graph2.getAdjacencyMatrix();
    
    // 生成所有可能的顶点排列
    std::vector<int> permutation(n);
    for (int i = 0; i < n; ++i) permutation[i] = i;
    
    do {
        if (checkPermutation(adj1, adj2, permutation)) {
            return true;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    
    return false;
}

bool FormationEnumerator::checkPermutation(const Eigen::MatrixXi& adj1,
                                            const Eigen::MatrixXi& adj2,
                                            const std::vector<int>& permutation) {
    int n = adj1.rows();

    // 检查排列后的邻接矩阵是否匹配
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // 如果 adj1[i][j] 不等于 adj2[permutation[i]][permutation[j]]，则排列无效
            if (adj1(i,j) != adj2(permutation[i],permutation[j])) {
                return false;
            }
        }
    }
    return true;
}

std::string FormationEnumerator::classifyByTopology(const ControlGraph& graph) {
    std::vector<int> features = FormationClassifier::getTopologicalFeatures(graph);
    
    int num_edges = graph.getEdges().size();
    int num_robots = graph.getNumRobots();
    
    // 简单拓扑分类规则
    if (num_edges == num_robots - 1) {
        return "链式结构";
    } else if (num_edges == num_robots) {
        return "树状结构";
    } else if (num_edges == num_robots - 1 && features[0] == 1) {
        return "星型结构";
    } else if (num_edges == num_robots && features[1] > 0) {
        return "环形结构";
    } else {
        return "复杂结构";
    }
}

std::string FormationEnumerator::classifyByRelativePositions(const ControlGraph& graph) {
    // 基于相对位置关系的分类
    // 这里实现简化的分类逻辑
    
    auto leaders = graph.getLeaderRobots();
    if (leaders.size() == 1) {
        return "单领导";
    } else {
        return "多领导";
    }
}

void FormationEnumerator::initializeEdgeMapping(int num_robots) {
    edge_mapping_.clear();
    for (int i = 0; i < num_robots; ++i) {
        for (int j = i + 1; j < num_robots; ++j) {
            edge_mapping_.push_back({i, j});
        }
    }

    for(const auto& edge : edge_mapping_){
        std::cout << edge.first << "->" << edge.second << std::endl;
    }
}

// FormationClassifier 的实现
std::string FormationClassifier::classify(const ControlGraph& graph) {
    FormationEnumerator enumerator(graph.getNumRobots());
    return enumerator.classifyFormation(graph);
}

//
std::vector<int> FormationClassifier::getTopologicalFeatures(const ControlGraph& graph) {
    std::vector<int> features;
    
    // 特征1：领航机器人数量
    auto leaders = graph.getLeaderRobots();
    features.push_back(leaders.size());
    
    // 特征2：图深度（控制链的最大长度）
    int depth = ConstraintValidator::calculateDepth(graph);
    features.push_back(depth);
    
    // 特征3：最大出度（控制负载的集中程度，反映某个节点的重量）
    std::vector<int> out_degree(graph.getNumRobots(), 0);
    for (const auto& edge : graph.getEdges()) {
        out_degree[edge.first]++;
    }
    int max_out_degree = *std::max_element(out_degree.begin(), out_degree.end());
    features.push_back(max_out_degree);
    
    return features;
}

std::vector<int> FormationClassifier::getDegreeSequence(const ControlGraph& graph) {
    std::vector<int> in_degree(graph.getNumRobots(), 0);
    std::vector<int> out_degree(graph.getNumRobots(), 0);
    
    for (const auto& edge : graph.getEdges()) {
        out_degree[edge.first]++;
        in_degree[edge.second]++;
    }
    
    std::vector<int> degree_sequence;
    for (int i = 0; i < graph.getNumRobots(); ++i) {
        degree_sequence.push_back(in_degree[i] + out_degree[i]);
    }
    
    return degree_sequence;
}

std::vector<int> FormationClassifier::getConnectedComponents(const ControlGraph& graph) {
    int num_robots = graph.getNumRobots();
    std::vector<bool> visited(num_robots, false);
    std::vector<int> component_sizes;
    
    for (int i = 0; i < num_robots; ++i) {
        if (!visited[i]) {
            std::queue<int> q;
            q.push(i);
            visited[i] = true;
            int size = 1;
            
            while (!q.empty()) {
                int current = q.front();
                q.pop();
                
                // 遍历所有边
                for (const auto& edge : graph.getEdges()) {
                    if (edge.first == current && !visited[edge.second]) {
                        visited[edge.second] = true;
                        q.push(edge.second);
                        size++;
                    }
                    if (edge.second == current && !visited[edge.first]) {
                        visited[edge.first] = true;
                        q.push(edge.first);
                        size++;
                    }
                }
            }
            
            component_sizes.push_back(size);
        }
    }
    
    return component_sizes;
}

} // namespace formation