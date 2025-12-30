#include <gtest/gtest.h>
#include "enumeration/formation_enumeration.h"
#include "core/control_graph.h"
#include <iostream>
#include <map>

using namespace formation;  

// 测试枚举器创建与基础属性
TEST(FormationEnumeratorTest, Creation) {
    // 正常创建（3个机器人）
    FormationEnumerator enumerator(3);
    EXPECT_EQ(enumerator.getAllFormationTypes().size(), 5);  // 固定5种队形类型
    EXPECT_NO_THROW(FormationEnumerator(1));                 // 最小规模（1个机器人）
    
    // 异常场景：机器人数量≤0
    EXPECT_THROW(FormationEnumerator(0), std::invalid_argument);
    EXPECT_THROW(FormationEnumerator(-2), std::invalid_argument);
}

// 测试小规模队形枚举（2个机器人）
TEST(FormationEnumeratorTest, SmallScaleEnumeration) {
    FormationEnumerator enumerator(3);
    auto formations = enumerator.enumerateAllFormations();
    
    // 验证枚举结果：2个机器人有效队形至少1种（如0→1）
    EXPECT_FALSE(formations.empty());
    EXPECT_GE(formations.size(), 2);
    
    // 验证所有枚举的队形均有效
    for (const auto& form : formations) {
        EXPECT_TRUE(form.isValid());
    }
}

// 测试队形分类功能
TEST(FormationEnumeratorTest, FormationClassification) {
    FormationEnumerator enumerator(3);
    std::vector<ControlGraph> formations = enumerator.enumerateAllFormations();
    for(const auto& form : formations){
        std::cout << form.getAdjacencyMatrix() << std::endl;
        std::cout<<"--------------"<< std::endl;
    }
    EXPECT_FALSE(formations.empty()) << "无有效队形，无法测试分类";
    
    // 分类并验证结果非空
    std::map<std::string, int> type_count;
    for (const auto& form : formations) {
        std::string type = enumerator.classifyFormation(form);
        std::cout << form.getAdjacencyMatrix() << "的队形分类结果: " << type << std::endl;
        type_count[type]++;
        EXPECT_FALSE(type.empty());  // 分类结果不能为空字符串
    }
    
    // 至少包含1种类型（如链式/星型）
    EXPECT_GE(type_count.size(), 1);
}

// 测试队形统计功能
TEST(FormationEnumeratorTest, FormationStatistics) {
    FormationEnumerator enumerator(3);
    std::map<std::string, int> stats = enumerator.getFormationStatistics();
    
    // 统计结果非空，且总数与有效队形数一致
    EXPECT_FALSE(stats.empty());
    std::vector<ControlGraph> all_formations = enumerator.enumerateAllFormations();
    
    int total = 0;
    for (const auto& [_, cnt] : stats) total += cnt;
    EXPECT_EQ(total, static_cast<int>(all_formations.size()));
}

// 测试进度回调功能
TEST(FormationEnumeratorTest, ProgressCallback) {
    FormationEnumerator enumerator(2);
    bool callback_triggered = false;
    int last_current = 0;
    int last_total = 0;
    
    // 设置回调
    enumerator.setProgressCallback([&](int current, int total) {
        callback_triggered = true;
        last_current = current;
        last_total = total;
    });
    
    // 执行枚举（触发回调）
    enumerator.enumerateAllFormations();
    
    // 验证回调触发，且进度值合理
    EXPECT_TRUE(callback_triggered);
    EXPECT_GT(last_total, 0);
    EXPECT_EQ(last_current, last_total);  // 回调应在枚举完成时触发（current=total）
}

// ------------------------------
// FormationClassifier 测试（对应你的 GraphPropertiesTest 风格）
// ------------------------------
// 测试分类器静态方法（拓扑特征、度序列、连通分量）
TEST(FormationClassifierTest, StaticMethods) {
    // 构造已知结构的控制图：3机器人链式（0→1→2）
    ControlGraph graph(3);
    EXPECT_TRUE(graph.addEdge(0, 1));
    EXPECT_TRUE(graph.addEdge(1, 2));
    
    // 1. 测试 classify 方法
    std::string type = FormationClassifier::classify(graph);
    EXPECT_FALSE(type.empty());
    
    // 2. 测试拓扑特征（3个特征：领航数、深度、最大出度）
    auto features = FormationClassifier::getTopologicalFeatures(graph);
    EXPECT_EQ(features.size(), 3);
    EXPECT_EQ(features[0], 1);    // 仅1个领航机器人（0）
    EXPECT_EQ(features[1], 2);    // 控制链深度：0→1→2（长度2）
    EXPECT_EQ(features[2], 1);    // 最大出度：每个节点最多1条出边
    
    // 3. 测试度序列（入度+出度）
    auto degree_seq = FormationClassifier::getDegreeSequence(graph);
    EXPECT_EQ(degree_seq.size(), 3);
    EXPECT_EQ(degree_seq[0], 1);  // 0：出度1，入度0 → 总1
    EXPECT_EQ(degree_seq[1], 2);  // 1：出度1，入度1 → 总2
    EXPECT_EQ(degree_seq[2], 1);  // 2：出度0，入度1 → 总1
    
    // 4. 测试连通分量（链式结构应为1个连通分量）
    auto components = FormationClassifier::getConnectedComponents(graph);
    EXPECT_EQ(components.size(), 1);
    EXPECT_EQ(components[0], 3);  // 包含所有3个机器人
}

// 测试分类器的同构辅助判断（基于度序列）
TEST(FormationClassifierTest, DegreeSequenceForIsomorphism) {
    // 构造两个同构的图（结构相同，边顺序不同）
    ControlGraph graph1(3);
    graph1.addEdge(0, 1);
    graph1.addEdge(0, 2);
    
    ControlGraph graph2(3);
    graph2.addEdge(0, 2);
    graph2.addEdge(0, 1);  // 边顺序不同，但结构相同
    
    // 度序列应一致（同构图的必要条件）
    auto seq1 = FormationClassifier::getDegreeSequence(graph1);
    auto seq2 = FormationClassifier::getDegreeSequence(graph2);
    std::sort(seq1.begin(), seq1.end());
    std::sort(seq2.begin(), seq2.end());
    EXPECT_EQ(seq1, seq2);
}
