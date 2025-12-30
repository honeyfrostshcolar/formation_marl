#include <gtest/gtest.h>
#include "core/control_graph.h"
#include "core/constraints.h"
#include <iostream>

using namespace formation;

// 测试控制图创建
TEST(ControlGraphTest, Creation) {
    ControlGraph graph(3);
    EXPECT_EQ(graph.getNumRobots(), 3);
    EXPECT_TRUE(graph.getEdges().empty());
    EXPECT_TRUE(graph.isValid());  // 空图应该有效
}

// 测试边操作
TEST(ControlGraphTest, EdgeOperations) {
    ControlGraph graph(4);
    
    // 测试添加有效边
    EXPECT_TRUE(graph.addEdge(0, 1));
    EXPECT_TRUE(graph.addEdge(0, 2));
    EXPECT_TRUE(graph.addEdge(1, 3));
    EXPECT_EQ(graph.getEdges().size(), 3);
    
    // 测试添加无效边（违反标签顺序）
    EXPECT_FALSE(graph.addEdge(2, 1));  // from >= to
    EXPECT_EQ(graph.getEdges().size(), 3);
    
    // 测试添加重复边
    EXPECT_FALSE(graph.addEdge(0, 1));  // 重复边
    EXPECT_EQ(graph.getEdges().size(), 3);
    
    // 测试移除边
    EXPECT_TRUE(graph.removeEdge(0, 1));
    EXPECT_EQ(graph.getEdges().size(), 2);
    EXPECT_FALSE(graph.removeEdge(0, 1));  // 边已不存在
}

// 测试约束验证
TEST(ConstraintTest, Validation) {
    // 测试有效控制图
    ControlGraph valid_graph(3);
    valid_graph.addEdge(0, 1);
    valid_graph.addEdge(0, 2);
    
    auto result = ConstraintValidator::validate(valid_graph);
    EXPECT_TRUE(result.is_valid);
    EXPECT_FALSE(result.leader_constraint_violated);
    EXPECT_FALSE(result.label_order_constraint_violated);
    EXPECT_FALSE(result.control_dimension_constraint_violated);
    
    // 测试无效控制图（没有领航机器人）
    // 注意：在当前约束系统下，由于标签顺序约束，无法形成真正的环
    // 所以这个测试可能需要调整
    ControlGraph no_leader_graph(3);
    // 这里无法添加 (2,0) 因为违反标签顺序约束
    // 所以无法测试真正的"没有领航者"情况
    
    // 测试无效控制图（违反控制维度约束）
    ControlGraph dimension_violation_graph(4);
    dimension_violation_graph.addEdge(0, 1);
    dimension_violation_graph.addEdge(0, 2);
    // 第三条边应该添加失败，所以图仍然是有效的
    EXPECT_FALSE(dimension_violation_graph.addEdge(0, 3));
    
    auto dimension_result = ConstraintValidator::validate(dimension_violation_graph);
    // 因为第三条边添加失败，图仍然是有效的
    EXPECT_TRUE(dimension_result.is_valid);
}

// 测试邻接矩阵
TEST(ControlGraphTest, AdjacencyMatrix) {
    ControlGraph graph(3);
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    
    auto matrix = graph.getAdjacencyMatrix();
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix(0, 1), 1);
    EXPECT_EQ(matrix(0, 2), 1);
    EXPECT_EQ(matrix(1, 0), 0);
    EXPECT_EQ(matrix(2, 0), 0);
    
    // 测试从邻接矩阵设置
    ControlGraph graph2(3);
    Eigen::MatrixXi new_matrix(3, 3);
    new_matrix << 0, 1, 1,
                  0, 0, 0,
                  0, 0, 0;
    
    EXPECT_TRUE(graph2.setFromAdjacencyMatrix(new_matrix));
    EXPECT_EQ(graph2.getEdges().size(), 2);
}

// 测试图属性
TEST(GraphPropertiesTest, BasicProperties) {
    ControlGraph graph(4);
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 3);
    
    // 测试连通性
    EXPECT_TRUE(ConstraintValidator::isConnected(graph));
    
    // 测试无环性
    EXPECT_TRUE(ConstraintValidator::isAcyclic(graph));
    
    // 测试图深度
    int depth = ConstraintValidator::calculateDepth(graph);
    EXPECT_EQ(depth, 2);  // 最长路径：0->1->3 或 0->2
    
    // 测试领航机器人
    auto leaders = graph.getLeaderRobots();
    EXPECT_EQ(leaders.size(), 1);
    EXPECT_NE(leaders.find(0), leaders.end());
}

//测试变换验证
//单步变换的核心是仅调整单个机器人的控制依赖关系
//即一个 follower 机器人更换其依赖的领导者（从依赖A 改为依赖B）
TEST(TransformationTest, Validation) {
    ControlGraph graph1(3);
    graph1.addEdge(0, 1);
    graph1.addEdge(0, 2);
    
    ControlGraph graph2(3);
    graph2.addEdge(0, 1);
    graph2.addEdge(1, 2);  // 改变一条边
    
    // 测试能否单步变换
    bool can_transform = ConstraintValidator::canTransformInOneStep(graph1, graph2);
    EXPECT_TRUE(can_transform);
    
    // 测试变换步骤验证
    bool valid_step = ConstraintValidator::validateTransformationStep(graph1, graph2);
    EXPECT_TRUE(valid_step);
}

// 测试边界情况
TEST(ControlGraphTest, BoundaryCases) {
    // 测试最小规模
    ControlGraph min_graph(1);
    EXPECT_EQ(min_graph.getNumRobots(), 1);
    EXPECT_TRUE(min_graph.isValid());
    
    // 测试空图的有效性
    ControlGraph empty_graph(3);
    EXPECT_TRUE(empty_graph.isValid());
    EXPECT_TRUE(empty_graph.getEdges().empty());
    
    // 测试单边图
    ControlGraph single_edge_graph(2);
    EXPECT_TRUE(single_edge_graph.addEdge(0, 1));
    EXPECT_TRUE(single_edge_graph.isValid());
}

// 测试图比较
TEST(ControlGraphTest, GraphComparison) {
    ControlGraph graph1(3);
    graph1.addEdge(0, 1);
    graph1.addEdge(0, 2);
    
    ControlGraph graph2(3);
    graph2.addEdge(0, 1);
    graph2.addEdge(0, 2);
    
    ControlGraph graph3(3);
    graph3.addEdge(0, 1);
    graph3.addEdge(1, 2);
    
    // 测试相等性
    EXPECT_TRUE(graph1 == graph2);
    EXPECT_FALSE(graph1 == graph3);
}

// 测试哈希功能
TEST(ControlGraphTest, HashFunction) {
    ControlGraph graph1(3);
    graph1.addEdge(0, 1);
    graph1.addEdge(0, 2);
    
    ControlGraph graph2(3);
    graph2.addEdge(0, 1);
    graph2.addEdge(0, 2);
    
    ControlGraph graph3(3);
    graph3.addEdge(0, 1);
    graph3.addEdge(1, 2);
    
    // 相同图应该有相同哈希值
    EXPECT_EQ(graph1.hash(), graph2.hash());
    // 不同图通常有不同的哈希值（可能有哈希冲突，但概率很低）
    EXPECT_NE(graph1.hash(), graph3.hash());
}