// #include "visualization/visualizer.h"
// #include <algorithm>
// #include <cmath>
// #include <iostream>

// namespace formation {

// // Visualizer 基类实现
// Visualizer::Visualizer(const VisualizationConfig& config) 
//     : config_(config) {}

// void Visualizer::setEventCallback(std::function<void(const VisualizationEventData&)> callback) {
//     event_callback_ = callback;
// }

// VisualizationConfig& Visualizer::getConfig() {
//     return config_;
// }

// void Visualizer::setConfig(const VisualizationConfig& config) {
//     config_ = config;
// }

// void Visualizer::triggerEvent(const VisualizationEventData& event) {
//     if (event_callback_) {
//         event_callback_(event);
//     }
// }

// // ControlGraphVisualizer 实现
// ControlGraphVisualizer::ControlGraphVisualizer(const VisualizationConfig& config)
//     : Visualizer(config), current_formation_index_(0) {}

// bool ControlGraphVisualizer::initialize() {
//     // 初始化图形系统
//     // 这里使用控制台输出模拟初始化
//     std::cout << "ControlGraphVisualizer 初始化完成" << std::endl;
//     return true;
// }

// void ControlGraphVisualizer::render(const ControlGraph& graph) {
//     // 计算布局
//     auto positions = calculateLayout(graph);
    
//     // 渲染机器人
//     for (const auto& pos : positions) {
//         renderRobot(pos);
//     }
    
//     // 渲染边
//     auto edges = graph.getEdges();
//     for (const auto& edge : edges) {
//         renderEdge(edge.first, edge.second, positions);
//     }
    
//     // 渲染信息面板
//     renderInfoPanel(graph);
    
//     // 触发渲染完成事件
//     triggerEvent(VisualizationEventData(VisualizationEvent::FORMATION_CHANGED));
// }

// void ControlGraphVisualizer::renderTransformation(const TransformationPath& path, float progress) {
//     if (progress >= 1.0f) {
//         // 渲染最终状态
//         render(path.target_graph);
//         return;
//     }
    
//     // 计算中间状态
//     int current_step = static_cast<int>(progress * path.operations.size());
    
//     // 应用部分变换
//     ControlGraph intermediate = path.start_graph;
//     for (int i = 0; i < current_step; ++i) {
//         // 这里需要实现应用变换操作的方法
//         // 简化实现：直接使用目标图
//     }
    
//     // 渲染中间状态
//     render(intermediate);
    
//     // 显示进度信息
//     std::cout << "变换进度: " << (progress * 100) << "%" << std::endl;
// }

// void ControlGraphVisualizer::startTransformationAnimation(const TransformationPath& path) {
//     animation_state_.path = path;
//     animation_state_.current_progress = 0.0f;
//     animation_state_.is_playing = true;
    
//     triggerEvent(VisualizationEventData(VisualizationEvent::TRANSFORMATION_STARTED));
// }

// void ControlGraphVisualizer::stopAnimation() {
//     animation_state_.is_playing = false;
// }

// bool ControlGraphVisualizer::isAnimating() const {
//     return animation_state_.is_playing;
// }

// void ControlGraphVisualizer::handleEvents() {
//     // 处理用户输入事件
//     // 这里使用控制台输入模拟事件处理
    
//     if (animation_state_.is_playing) {
//         // 更新动画进度
//         animation_state_.current_progress += 0.1f / config_.animation_duration;
        
//         if (animation_state_.current_progress >= 1.0f) {
//             animation_state_.current_progress = 1.0f;
//             animation_state_.is_playing = false;
//             triggerEvent(VisualizationEventData(VisualizationEvent::TRANSFORMATION_COMPLETED));
//         }
        
//         triggerEvent(VisualizationEventData(VisualizationEvent::ANIMATION_FRAME));
//     }
// }

// std::vector<RobotPosition> ControlGraphVisualizer::calculateLayout(const ControlGraph& graph) {
//     if (!custom_layout_.empty()) {
//         return custom_layout_;
//     }
    
//     if (config_.auto_layout) {
//         return circularLayout(graph);
//     } else {
//         return forceDirectedLayout(graph);
//     }
// }

// void ControlGraphVisualizer::highlightRobot(int robot_id) {
//     // 高亮指定机器人
//     std::cout << "高亮机器人: " << robot_id << std::endl;
// }

// void ControlGraphVisualizer::highlightEdge(int from_robot, int to_robot) {
//     // 高亮指定边
//     std::cout << "高亮边: " << from_robot << " -> " << to_robot << std::endl;
// }

// void ControlGraphVisualizer::clearHighlights() {
//     // 清除所有高亮
//     std::cout << "清除所有高亮" << std::endl;
// }

// void ControlGraphVisualizer::setCustomLayout(const std::vector<RobotPosition>& positions) {
//     custom_layout_ = positions;
// }

// // 布局算法实现
// std::vector<RobotPosition> ControlGraphVisualizer::circularLayout(const ControlGraph& graph) {
//     std::vector<RobotPosition> positions;
//     int num_robots = graph.getNumRobots();
    
//     float center_x = config_.window_width / 2.0f;
//     float center_y = config_.window_height / 2.0f;
//     float radius = std::min(config_.window_width, config_.window_height) * 0.4f;
    
//     for (int i = 0; i < num_robots; ++i) {
//         float angle = 2.0f * M_PI * i / num_robots;
//         float x = center_x + radius * std::cos(angle);
//         float y = center_y + radius * std::sin(angle);
        
//         bool is_leader = graph.isLeaderRobot(i);
//         positions.emplace_back(i, x, y, is_leader);
//     }
    
//     return positions;
// }

// std::vector<RobotPosition> ControlGraphVisualizer::forceDirectedLayout(const ControlGraph& graph) {
//     // 简化的力导向布局算法
//     std::vector<RobotPosition> positions;
//     int num_robots = graph.getNumRobots();
    
//     // 初始化为圆形布局
//     positions = circularLayout(graph);
    
//     // 力导向迭代（简化版）
//     for (int iteration = 0; iteration < 10; ++iteration) {
//         // 这里可以实现更复杂的力导向算法
//         // 简化实现：直接使用圆形布局
//     }
    
//     return positions;
// }

// std::vector<RobotPosition> ControlGraphVisualizer::hierarchicalLayout(const ControlGraph& graph) {
//     // 层次布局算法
//     std::vector<RobotPosition> positions;
//     int num_robots = graph.getNumRobots();
    
//     // 计算每个机器人的深度
//     std::vector<int> depths(num_robots, 0);
//     // 这里需要实现深度计算逻辑
    
//     // 根据深度排列机器人
//     float start_y = config_.window_height * 0.2f;
//     float level_height = config_.window_height * 0.6f / (num_robots > 0 ? num_robots : 1);
    
//     for (int i = 0; i < num_robots; ++i) {
//         float x = config_.window_width * (i + 1) / (num_robots + 1);
//         float y = start_y + depths[i] * level_height;
//         bool is_leader = graph.isLeaderRobot(i);
//         positions.emplace_back(i, x, y, is_leader);
//     }
    
//     return positions;
// }

// // 渲染辅助方法
// void ControlGraphVisualizer::renderRobot(const RobotPosition& pos) {
//     // 使用控制台输出模拟渲染
//     std::cout << "渲染机器人 " << pos.robot_id 
//               << " 位置: (" << pos.x << ", " << pos.y << ")"
//               << (pos.is_leader ? " [领航]" : "") << std::endl;
// }

// void ControlGraphVisualizer::renderEdge(int from_robot, int to_robot, 
//                                        const std::vector<RobotPosition>& positions) {
//     // 使用控制台输出模拟渲染
//     std::cout << "渲染边: " << from_robot << " -> " << to_robot << std::endl;
// }

// void ControlGraphVisualizer::renderInfoPanel(const ControlGraph& graph) {
//     // 显示队形信息
//     std::cout << "=== 队形信息 ===" << std::endl;
//     std::cout << "机器人数量: " << graph.getNumRobots() << std::endl;
//     std::cout << "边数量: " << graph.getEdges().size() << std::endl;
    
//     auto leaders = graph.getLeaderRobots();
//     std::cout << "领航机器人: ";
//     for (int leader : leaders) {
//         std::cout << leader << " ";
//     }
//     std::cout << std::endl;
    
//     std::cout << "队形有效性: " << (graph.isValid() ? "有效" : "无效") << std::endl;
// }

// // FormationEnumerationVisualizer 实现
// FormationEnumerationVisualizer::FormationEnumerationVisualizer(const VisualizationConfig& config)
//     : Visualizer(config), current_formation_index_(0) {}

// bool FormationEnumerationVisualizer::initialize() {
//     std::cout << "FormationEnumerationVisualizer 初始化完成" << std::endl;
//     return true;
// }

// void FormationEnumerationVisualizer::render(const ControlGraph& graph) {
//     std::cout << "=== 队形枚举可视化 ===" << std::endl;
    
//     if (!formations_.empty()) {
//         renderFormationGrid();
//         renderFormationDetails(graph);
//     } else {
//         std::cout << "未设置队形数据" << std::endl;
//     }
// }

// void FormationEnumerationVisualizer::renderTransformation(const TransformationPath& path, float progress) {
//     // 队形枚举可视化器不支持变换动画
//     render(path.target_graph);
// }

// void FormationEnumerationVisualizer::startTransformationAnimation(const TransformationPath& path) {
//     // 队形枚举可视化器不支持变换动画
//     std::cout << "队形枚举可视化器不支持变换动画" << std::endl;
// }

// void FormationEnumerationVisualizer::stopAnimation() {
//     // 无动画可停止
// }

// bool FormationEnumerationVisualizer::isAnimating() const {
//     return false;
// }

// void FormationEnumerationVisualizer::handleEvents() {
//     // 处理队形切换事件
//     // 简化实现
// }

// std::vector<RobotPosition> FormationEnumerationVisualizer::calculateLayout(const ControlGraph& graph) {
//     return circularLayout(graph);
// }

// void FormationEnumerationVisualizer::setFormations(const std::vector<ControlGraph>& formations) {
//     formations_ = formations;
// }

// void FormationEnumerationVisualizer::setCurrentFormationIndex(int index) {
//     if (index >= 0 && index < static_cast<int>(formations_.size())) {
//         current_formation_index_ = index;
//     }
// }

// void FormationEnumerationVisualizer::showFormationComparison(const ControlGraph& graph1, const ControlGraph& graph2) {
//     std::cout << "=== 队形比较 ===" << std::endl;
//     std::cout << "队形1: " << graph1.getNumRobots() << " 个机器人" << std::endl;
//     std::cout << "队形2: " << graph2.getNumRobots() << " 个机器人" << std::endl;
// }

// void FormationEnumerationVisualizer::renderFormationGrid() {
//     std::cout << "显示队形网格 (" << formations_.size() << " 个队形)" << std::endl;
    
//     for (size_t i = 0; i < formations_.size(); ++i) {
//         std::cout << "队形 " << i << ": ";
//         if (i == current_formation_index_) {
//             std::cout << "[当前] ";
//         }
//         std::cout << formations_[i].getNumRobots() << " 个机器人, "
//                   << formations_[i].getEdges().size() << " 条边" << std::endl;
//     }
// }

// void FormationEnumerationVisualizer::renderFormationDetails(const ControlGraph& graph) {
//     std::cout << "当前队形详细信息:" << std::endl;
//     std::cout << "机器人数量: " << graph.getNumRobots() << std::endl;
//     std::cout << "边数量: " << graph.getEdges().size() << std::endl;
//     std::cout << "领航机器人: ";
    
//     auto leaders = graph.getLeaderRobots();
//     for (int leader : leaders) {
//         std::cout << leader << " ";
//     }
//     std::cout << std::endl;
// }

// // TransformationPathVisualizer 实现
// TransformationPathVisualizer::TransformationPathVisualizer(const VisualizationConfig& config)
//     : Visualizer(config) {}

// bool TransformationPathVisualizer::initialize() {
//     std::cout << "TransformationPathVisualizer 初始化完成" << std::endl;
//     return true;
// }

// void TransformationPathVisualizer::render(const ControlGraph& graph) {
//     std::cout << "=== 变换路径可视化 ===" << std::endl;
//     renderPathOverview();
// }

// void TransformationPathVisualizer::renderTransformation(const TransformationPath& path, float progress) {
//     renderStepByStep(path, progress);
// }

// void TransformationPathVisualizer::startTransformationAnimation(const TransformationPath& path) {
//     current_path_ = path;
//     std::cout << "开始变换路径动画" << std::endl;
// }

// void TransformationPathVisualizer::stopAnimation() {
//     std::cout << "停止变换路径动画" << std::endl;
// }

// bool TransformationPathVisualizer::isAnimating() const {
//     return false;
// }

// void TransformationPathVisualizer::handleEvents() {
//     // 处理变换路径相关事件
// }

// std::vector<RobotPosition> TransformationPathVisualizer::calculateLayout(const ControlGraph& graph) {
//     return circularLayout(graph);
// }

// void TransformationPathVisualizer::setTransformationPath(const TransformationPath& path) {
//     current_path_ = path;
// }

// void TransformationPathVisualizer::showCostAnalysis(const TransformationPath& path) {
//     std::cout << "=== 成本分析 ===" << std::endl;
//     std::cout << "总成本: " << path.total_cost << std::endl;
//     std::cout << "操作数量: " << path.operations.size() << std::endl;
    
//     for (size_t i = 0; i < path.operations.size(); ++i) {
//         std::cout << "步骤 " << i << ": " << path.operations[i].description
//                   << " (成本: " << path.operations[i].cost << ")" << std::endl;
//     }
// }

// void TransformationPathVisualizer::showAlternativePaths(const std::vector<TransformationPath>& paths) {
//     std::cout << "=== 替代路径 ===" << std::endl;
//     std::cout << "发现 " << paths.size() << " 条替代路径" << std::endl;
    
//     for (size_t i = 0; i < paths.size(); ++i) {
//         std::cout << "路径 " << i << ": " << paths[i].operations.size() 
//                   << " 步, 总成本: " << paths[i].total_cost << std::endl;
//     }
// }

// void TransformationPathVisualizer::renderPathOverview() {
//     if (current_path_.is_valid) {
//         std::cout << "变换路径概览:" << std::endl;
//         std::cout << "从 " << current_path_.start_graph.getNumRobots() << " 机器人队形" << std::endl;
//         std::cout << "到 " << current_path_.target_graph.getNumRobots() << " 机器人队形" << std::endl;
//         std::cout << "步骤数: " << current_path_.operations.size() << std::endl;
//         std::cout << "总成本: " << current_path_.total_cost << std::endl;
//     } else {
//         std::cout << "无效的变换路径" << std::endl;
//     }
// }

// void TransformationPathVisualizer::renderStepByStep(const TransformationPath& path, float progress) {
//     int current_step = static_cast<int>(progress * path.operations.size());
    
//     std::cout << "=== 逐步变换 ===" << std::endl;
//     std::cout << "当前步骤: " << current_step << "/" << path.operations.size() << std::endl;
    
//     if (current_step < static_cast<int>(path.operations.size())) {
//         std::cout << "当前操作: " << path.operations[current_step].description << std::endl;
//     }
// }

// void TransformationPathVisualizer::renderCostGraph(const TransformationPath& path) {
//     std::cout << "=== 成本图表 ===" << std::endl;
    
//     double cumulative_cost = 0.0;
//     for (size_t i = 0; i < path.operations.size(); ++i) {
//         cumulative_cost += path.operations[i].cost;
//         std::cout << "步骤 " << i << ": 成本 " << path.operations[i].cost
//                   << ", 累计成本 " << cumulative_cost << std::endl;
//     }
// }

// // VisualizerFactory 实现
// std::unique_ptr<Visualizer> VisualizerFactory::createControlGraphVisualizer(const VisualizationConfig& config) {
//     return std::make_unique<ControlGraphVisualizer>(config);
// }

// std::unique_ptr<Visualizer> VisualizerFactory::createFormationEnumerationVisualizer(const VisualizationConfig& config) {
//     return std::make_unique<FormationEnumerationVisualizer>(config);
// }

// std::unique_ptr<Visualizer> VisualizerFactory::createTransformationPathVisualizer(const VisualizationConfig& config) {
//     return std::make_unique<TransformationPathVisualizer>(config);
// }

// } // namespace formation