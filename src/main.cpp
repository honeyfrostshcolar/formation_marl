#include <iostream>
#include <iomanip>
#include "core/control_graph.h"
#include "core/constraints.h"

using namespace formation;

void testBasicControlGraph() {
    std::cout << "=== 基础控制图测试 ===" << std::endl;
    
    // 创建包含4个机器人的控制图
    ControlGraph graph(4);
    
    // 添加一些边
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 3);
    
    std::cout << graph.toString() << std::endl;
    
    // 显示邻接矩阵
    auto matrix = graph.getAdjacencyMatrix();
    std::cout << "邻接矩阵:" << std::endl;
    std::cout << matrix << std::endl;
}

void testConstraintValidation() {
    std::cout << "\n=== 约束验证测试 ===" << std::endl;
    
    // 测试有效控制图
    ControlGraph valid_graph(3);
    valid_graph.addEdge(0, 1);
    valid_graph.addEdge(0, 2);
    
    auto result = ConstraintValidator::validate(valid_graph);
    std::cout << "有效控制图验证结果:" << std::endl;
    for (const auto& detail : result.details) {
        std::cout << "  - " << detail << std::endl;
    }
    
    // 测试无效控制图（违反标签顺序约束）
    ControlGraph invalid_graph(3);
    invalid_graph.addEdge(1, 0);  // 违反标签顺序约束
    
    auto invalid_result = ConstraintValidator::validate(invalid_graph);
    std::cout << "无效控制图验证结果:" << std::endl;
    for (const auto& detail : invalid_result.details) {
        std::cout << "  - " << detail << std::endl;
    }
}

void testGraphProperties() {
    std::cout << "\n=== 图属性测试 ===" << std::endl;
    
    ControlGraph graph(5);
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 3);
    graph.addEdge(2, 4);
    
    std::cout << "连通性: " << (ConstraintValidator::isConnected(graph) ? "是" : "否") << std::endl;
    std::cout << "无环性: " << (ConstraintValidator::isAcyclic(graph) ? "是" : "否") << std::endl;
    std::cout << "图深度: " << ConstraintValidator::calculateDepth(graph) << std::endl;
}

void testTransformationValidation() {
    std::cout << "\n=== 变换验证测试 ===" << std::endl;
    
    // 创建两个控制图
    ControlGraph graph1(4);
    graph1.addEdge(0, 1);
    graph1.addEdge(0, 2);
    graph1.addEdge(1, 3);
    
    ControlGraph graph2(4);
    graph2.addEdge(0, 1);
    graph2.addEdge(0, 3);  // 改变一条边
    graph2.addEdge(1, 2);
    
    bool can_transform = ConstraintValidator::canTransformInOneStep(graph1, graph2);
    std::cout << "能否单步变换: " << (can_transform ? "是" : "否") << std::endl;
    
    if (can_transform) {
        bool valid_step = ConstraintValidator::validateTransformationStep(graph1, graph2);
        std::cout << "变换步骤是否合法: " << (valid_step ? "是" : "否") << std::endl;
    }
}

int main() {
    std::cout << "机器人编队控制图系统 - 核心数据结构测试" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        testBasicControlGraph();
        testConstraintValidation();
        testGraphProperties();
        testTransformationValidation();
        
        std::cout << "\n=== 所有测试完成 ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}