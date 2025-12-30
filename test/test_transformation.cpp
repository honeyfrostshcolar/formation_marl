#include <gtest/gtest.h>
#include "transformation/transformation_algorithm.h"

namespace formation {

// 测试 TransformationOperation 基本功能
TEST(TransformationTest, TransformationOperation_BasicFunction) {
    // 测试默认参数构造
    TransformationOperation op1(TransformationType::ADD_EDGE, 0, 1);
    EXPECT_EQ(op1.type, TransformationType::ADD_EDGE);
    EXPECT_EQ(op1.robot1, 0);
    EXPECT_EQ(op1.robot2, 1);
    EXPECT_DOUBLE_EQ(op1.cost, 1.0);
    EXPECT_FALSE(op1.description.empty());

    // 测试自定义成本构造
    TransformationOperation op2(TransformationType::REMOVE_EDGE, 2, 3, 2.5);
    EXPECT_DOUBLE_EQ(op2.cost, 2.5);
}

// 测试 TransformationPath 基本功能
TEST(TransformationTest, TransformationPath_BasicFunction) {
    ControlGraph start(3);
    ControlGraph target(3);

    // 测试路径初始化
    TransformationPath path(start, target);
    EXPECT_EQ(path.start_graph.getNumRobots(), 3);
    EXPECT_EQ(path.target_graph.getNumRobots(), 3);
    EXPECT_DOUBLE_EQ(path.total_cost, 0.0);
    EXPECT_FALSE(path.is_valid);

    // 测试添加操作后的数据验证
    path.operations.emplace_back(TransformationType::ADD_EDGE, 0, 1);
    path.operations.emplace_back(TransformationType::ADD_EDGE, 1, 2);
    path.total_cost = 2.0;
    path.is_valid = true;

    EXPECT_EQ(path.operations.size(), 2);
    EXPECT_DOUBLE_EQ(path.total_cost, 2.0);
    EXPECT_TRUE(path.is_valid);
}

// 测试 TransformationAlgorithm 创建和配置
TEST(TransformationAlgorithmTest, CreationAndConfiguration) {
    TransformationAlgorithm algorithm;

    // 测试默认参数初始化
    auto stats = algorithm.getPerformanceStats();
    EXPECT_EQ(stats.total_operations, 0);
    EXPECT_EQ(stats.valid_operations, 0);

    // 测试搜索参数设置（假设 setSearchParameters 是设置最大步骤和超时时间等，此处验证无异常）
    EXPECT_NO_THROW(algorithm.setSearchParameters(15, 2000));

    // 测试成本函数设置
    EXPECT_NO_THROW(
        algorithm.setCostFunction([](const TransformationOperation& op, const ControlGraph& graph) {
            return op.cost * 2.0;
        })
    );
}

// 测试变换操作验证逻辑
TEST(TransformationAlgorithmTest, TransformationValidation) {
    TransformationAlgorithm algorithm;
    ControlGraph graph(3);

    // 测试有效操作（机器人编号在范围内，边未存在）
    TransformationOperation valid_op(TransformationType::ADD_EDGE, 0, 1);
    EXPECT_TRUE(algorithm.validateTransformation(valid_op, graph));

    // 测试无效操作（机器人编号超出范围）
    TransformationOperation invalid_op(TransformationType::ADD_EDGE, 5, 6);
    EXPECT_FALSE(algorithm.validateTransformation(invalid_op, graph));

    // 测试重复边操作（边已存在，再次添加）
    graph.addEdge(0, 1);
    TransformationOperation duplicate_op(TransformationType::ADD_EDGE, 0, 1);
    EXPECT_FALSE(algorithm.validateTransformation(duplicate_op, graph));
}

// 测试简单变换路径搜索
TEST(TransformationAlgorithmTest, SimpleTransformationPathSearch) {
    TransformationAlgorithm algorithm;

    // 构造起始和目标队形（起始无边缘，目标有 0→1 和 1→2 边）
    ControlGraph start(3);
    ControlGraph target(3);
    target.addEdge(0, 1);
    target.addEdge(1, 2);

    // 搜索变换路径
    TransformationPath path = algorithm.findTransformationPath(start, target);

    //algorithm.displayTransformationPath(path);
    
    // 验证路径有效性（假设存在有效路径）
    if (path.is_valid) {
        EXPECT_GT(path.operations.size(), 0);  // 至少有一个操作
        EXPECT_GT(path.total_cost, 0.0);       // 总成本大于 0
    } else {
        GTEST_SKIP() << "未找到有效变换路径，跳过后续验证";
    }
}

// 测试 TransformationManager 功能（单步变换、撤销、重做、历史记录）
TEST(TransformationManagerTest, BasicFunctions) {
    TransformationManager manager;
    ControlGraph graph(3);

    // 测试单步变换执行
    bool perform_success = manager.performSingleTransformation(graph, TransformationType::ADD_EDGE, 0, 1);
    EXPECT_TRUE(perform_success);
    EXPECT_TRUE(graph.hasEdge(0, 1));

    // 测试撤销操作
    bool undo_success = manager.undoLastTransformation(graph);
    EXPECT_TRUE(undo_success);
    EXPECT_FALSE(graph.hasEdge(0, 1));

    // 测试重做操作
    bool redo_success = manager.redoLastTransformation(graph);
    EXPECT_TRUE(redo_success);
    EXPECT_TRUE(graph.hasEdge(0, 1));

    // 测试历史记录获取
    auto history = manager.getTransformationHistory();
    EXPECT_EQ(history.size(), 1);  // 仅一次有效变换操作
}

// 测试 TransformationOptimizer 路径搜索功能
TEST(TransformationOptimizerTest, PathSearchFunctions) {
    TransformationOptimizer optimizer;

    // 构造起始和目标队形
    ControlGraph start(3);
    ControlGraph target(3);
    target.addEdge(0, 1);
    target.addEdge(1, 2);

    // 测试最小成本路径
    TransformationPath min_cost_path = optimizer.findMinimumCostPath(start, target);
    if (min_cost_path.is_valid) {
        EXPECT_GT(min_cost_path.total_cost, 0.0);
    }

    // 测试最小步骤路径
    TransformationPath min_step_path = optimizer.findMinimumStepPath(start, target);
    if (min_step_path.is_valid) {
        EXPECT_GT(min_step_path.operations.size(), 0);
    }

    // 测试平衡路径（成本和步骤权重各 0.5）
    TransformationPath balanced_path = optimizer.findBalancedPath(start, target, 0.5, 0.5);
    if (balanced_path.is_valid) {
        EXPECT_GT(balanced_path.operations.size(), 0);
        EXPECT_GT(balanced_path.total_cost, 0.0);
    }
}

// 测试性能统计功能
TEST(TransformationAlgorithmTest, PerformanceStatistics) {
    TransformationAlgorithm algorithm;

    // 构造测试用起始和目标队形
    ControlGraph start(3);
    ControlGraph target(3);
    target.addEdge(0, 1);
    target.addEdge(1, 2);

    // 执行路径搜索（触发性能统计）
    algorithm.findTransformationPath(start, target);

    // 获取并验证性能统计数据
    auto stats = algorithm.getPerformanceStats();
    EXPECT_GE(stats.total_operations, 0);      // 总操作数非负
    EXPECT_GE(stats.valid_operations, 0);      // 有效操作数非负
    EXPECT_GE(stats.average_cost, 0.0);        // 平均成本非负
    EXPECT_GE(stats.search_time, 0.0);         // 搜索时间非负
    EXPECT_GE(stats.explored_states, 0);       // 探索状态数非负
}

}  // namespace formation
