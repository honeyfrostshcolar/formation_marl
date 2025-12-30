#include "core/control_graph.h"
#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <iostream> 
#include <Eigen/Dense> 

namespace formation {

ControlGraph::ControlGraph(int num_robots) 
    : num_robots_(num_robots) {
    if (num_robots <= 0) {
        throw std::invalid_argument("机器人数量必须大于0");
    }
}

bool ControlGraph::addEdge(int from, int to) {
    // 检查机器人编号有效性
    if (from < 0 || from >= num_robots_ || to < 0 || to >= num_robots_) {
        return false;
    }
    
    // 检查标签顺序约束：只能从小编号指向大编号
    if (from >= to) {
        return false;
    }
     
    // 检查边是否已存在
    if (edges_.find({from, to}) != edges_.end()) {
        return false;
    }
    
    // 添加边（先尝试，再验证，必要时回滚）
    edges_.insert({from, to});
    
    return true;
}

bool ControlGraph::removeEdge(int from, int to) {
    auto it = edges_.find({from, to});
    if (it == edges_.end()) {
        return false;
    }
    
    edges_.erase(it);
    return true;
}

Eigen::MatrixXi ControlGraph::getAdjacencyMatrix() const {
    Eigen::MatrixXi matrix = Eigen::MatrixXi::Zero(num_robots_, num_robots_);
    
    for (const auto& edge : edges_) {
        matrix(edge.first, edge.second) = 1;
    }
    
    return matrix;
}

// 测试用例函数
bool ControlGraph::setFromAdjacencyMatrix(const Eigen::MatrixXi& matrix) {
    if (matrix.rows() != num_robots_ || matrix.cols() != num_robots_) {
        return false;
    }
    
    // 备份现有边
    auto old_edges = edges_;
    edges_.clear();
    
    // 从矩阵添加边
    for (int i = 0; i < num_robots_; ++i) {
        for (int j = 0; j < num_robots_; ++j) {
            if (matrix(i, j) != 0) {
                if (!addEdge(i, j)) {
                    // 如果添加失败，恢复原状态
                    edges_ = std::move(old_edges);
                    return false;
                }
            }
        }
    }
    
    return isValid();
}

bool ControlGraph::isValid() const {

    if(edges_.empty()){
        return false;
    }

    return checkLabelOrderConstraint() && 
           checkControlDimensionConstraint() && 
           checkLeaderExistenceConstraint();
}

std::set<int> ControlGraph::getLeaderRobots() const {
    std::set<int> leaders;
    
    // 计算每个机器人的入度
    std::vector<int> in_degree(num_robots_, 0);
    for (const auto& edge : edges_) {
        in_degree[edge.second]++;
    }
    
    // 入度为0的机器人是领航机器人
    for (int i = 0; i < num_robots_; ++i) {
        if (in_degree[i] == 0) {
            leaders.insert(i);
        }
    }
    
    return leaders;
}

// 转换为字符串表示（用于调试）
std::string ControlGraph::toString() const {
    std::stringstream ss;
    ss << "ControlGraph with " << num_robots_ << " robots:\n";
    ss << "Edges: ";
    
    for (const auto& edge : edges_) {
        ss << "(" << edge.first << "->" << edge.second << ") ";
    }
    
    ss << "\nLeaders: ";
    auto leaders = getLeaderRobots();
    for (int leader : leaders) {
        ss << leader << " ";
    }
    
    ss << "\nValid: " << (isValid() ? "Yes" : "No");
    
    return ss.str();
}

size_t ControlGraph::hash() const {
    size_t seed = 0;
    
    // 使用边的集合来计算哈希值
    for (const auto& edge : edges_) {
        seed ^= std::hash<int>{}(edge.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(edge.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    
    return seed;
}

bool ControlGraph::operator==(const ControlGraph& other) const {
    if (num_robots_ != other.num_robots_) {
        return false;
    }
    
    return edges_ == other.edges_;
}

// 新增：获取边集合的访问接口
const std::set<std::pair<int,int>>& ControlGraph::getEdges() const {
    return edges_;
}

// 新增：获取机器人数量
int ControlGraph::getNumRobots() const {
    return num_robots_;
}

// 新增：判断是否存在某条边
bool ControlGraph::hasEdge(int from, int to) const {
    return edges_.find({from, to}) != edges_.end();
}

// 新增：判断某个机器人是否为领航机器人（入度为0）
bool ControlGraph::isLeaderRobot(int id) const {
    if (id < 0 || id >= num_robots_) return false;
    // 通过计算入度判断
    std::vector<int> in_degree(num_robots_, 0);
    for (const auto& edge : edges_) {
        in_degree[edge.second]++;
    }
    return in_degree[id] == 0;
}

//必须从 “小编号机器人” 指向 “大连号机器人”
bool ControlGraph::checkLabelOrderConstraint() const {
    //std::cout << getAdjacencyMatrix() << std::endl;
    
    for (const auto& edge : edges_) {
        if (edge.first >= edge.second) {
            return false;
        }
    }
    return true;
}

//机器人可依赖的领导者数量不超过 2(且目前先暂定为可依赖的领导者数量不小于1)
//规定：指向谁，就是控制谁
bool ControlGraph::checkControlDimensionConstraint() const {
    
    //std::cout << getAdjacencyMatrix() << std::endl;
    
    // 计算每个机器人的出度
    std::vector<int> out_degree(num_robots_, 0);
    std::vector<int> in_degree(num_robots_, 0);
    for (const auto& edge : edges_) {
        out_degree[edge.first]++;
        in_degree[edge.second]++;
    }
    
    //检查每个机器人的出度是否不超过2(且目前先暂定为可依赖的领导者数量不小于1)
    for(int i = 1; i < num_robots_; ++i){
        if(in_degree[i] < 1){
            // std::cout << "可依赖的领导者数量小于1" << std::endl;
            // std::cout << getAdjacencyMatrix() << std::endl;
            return false;
        }
    }

    for (int degree : out_degree) {
        if (degree > 2 ) {
            // std::cout << "控制维度大于2" << std::endl;
            // std::cout << getAdjacencyMatrix() << std::endl;
            return false;
        }
    } 
    
    return true;
}

//领航机器人无入边
//领航机器人至少有一条出边
bool ControlGraph::checkLeaderExistenceConstraint() const {

    //std::cout << getAdjacencyMatrix() << std::endl;
    
    // 计算领航者的出入度
    std::vector<int> out_degree(num_robots_, 0);
    std::vector<int> in_degree(num_robots_, 0);
    for (const std::pair<int, int>& edge : edges_) {
        in_degree[edge.second]++;
        out_degree[edge.first]++;
    }   
    
    // bool mark = (in_degree[0] == 0 && out_degree[0] > 0);
    // std::cout << mark << std::endl;

    // 检查领航者的出度至少为1，入度为0
    if (in_degree[0] == 0 && out_degree[0] > 0) {
        //std::cout << "有效" << std::endl;
        return true;
    }

    return false;
}

} // namespace formation