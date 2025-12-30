// #pragma once

// #include "core/control_graph.h"
// #include "transformation/transformation_algorithm.h"
// #include <vector>
// #include <string>
// #include <memory>
// #include <functional>

// namespace formation {

// // 可视化配置结构体
// struct VisualizationConfig {
//     int window_width = 800;
//     int window_height = 600;
//     std::string window_title = "多机器人队形控制系统";
    
//     // 颜色配置
//     struct Color {
//         float r, g, b, a;
//         Color(float red = 0.0f, float green = 0.0f, float blue = 0.0f, float alpha = 1.0f)
//             : r(red), g(green), b(blue), a(alpha) {}
//     };
    
//     Color background_color = {0.95f, 0.95f, 0.95f, 1.0f}; // 浅灰色背景
//     Color robot_color = {0.2f, 0.6f, 1.0f, 1.0f};        // 蓝色机器人
//     Color leader_color = {1.0f, 0.4f, 0.2f, 1.0f};      // 橙色领航机器人
//     Color edge_color = {0.3f, 0.3f, 0.3f, 1.0f};         // 深灰色边
//     Color highlight_color = {1.0f, 0.8f, 0.0f, 1.0f};    // 黄色高亮
    
//     // 尺寸配置
//     float robot_radius = 20.0f;
//     float leader_radius = 25.0f;
//     float edge_width = 3.0f;
    
//     // 动画配置
//     bool enable_animations = true;
//     float animation_duration = 1.0f; // 秒
    
//     // 布局配置
//     bool auto_layout = true;
//     float layout_spacing = 100.0f;
// };

// // 机器人位置信息
// struct RobotPosition {
//     int robot_id;
//     float x, y;
//     bool is_leader;
    
//     RobotPosition(int id = 0, float pos_x = 0.0f, float pos_y = 0.0f, bool leader = false)
//         : robot_id(id), x(pos_x), y(pos_y), is_leader(leader) {}
// };

// // 可视化事件类型
// enum class VisualizationEvent {
//     ROBOT_CLICKED,      // 机器人被点击
//     EDGE_CLICKED,       // 边被点击
//     FORMATION_CHANGED,  // 队形改变
//     TRANSFORMATION_STARTED, // 变换开始
//     TRANSFORMATION_COMPLETED, // 变换完成
//     ANIMATION_FRAME     // 动画帧更新
// };

// // 可视化事件数据
// struct VisualizationEventData {
//     VisualizationEvent type;
//     int robot_id;
//     int robot_id2; // 用于边事件
//     std::string message;
    
//     VisualizationEventData(VisualizationEvent t, int id1 = -1, int id2 = -1, const std::string& msg = "")
//         : type(t), robot_id(id1), robot_id2(id2), message(msg) {}
// };

// // 可视化器基类
// class Visualizer {
// public:
//     Visualizer(const VisualizationConfig& config = VisualizationConfig());
//     virtual ~Visualizer() = default;
    
//     // 初始化
//     virtual bool initialize() = 0;
    
//     // 渲染
//     virtual void render(const ControlGraph& graph) = 0;
//     virtual void renderTransformation(const TransformationPath& path, float progress) = 0;
    
//     // 动画控制
//     virtual void startTransformationAnimation(const TransformationPath& path) = 0;
//     virtual void stopAnimation() = 0;
//     virtual bool isAnimating() const = 0;
    
//     // 事件处理
//     virtual void handleEvents() = 0;
    
//     // 设置事件回调
//     void setEventCallback(std::function<void(const VisualizationEventData&)> callback);
    
//     // 配置管理
//     VisualizationConfig& getConfig();
//     void setConfig(const VisualizationConfig& config);
    
//     // 布局算法
//     virtual std::vector<RobotPosition> calculateLayout(const ControlGraph& graph) = 0;
    
// protected:
//     VisualizationConfig config_;
//     std::function<void(const VisualizationEventData&)> event_callback_;
    
//     // 触发事件
//     void triggerEvent(const VisualizationEventData& event);
// };

// // 控制图可视化器
// class ControlGraphVisualizer : public Visualizer {
// public:
//     ControlGraphVisualizer(const VisualizationConfig& config = VisualizationConfig());
    
//     bool initialize() override;
//     void render(const ControlGraph& graph) override;
//     void renderTransformation(const TransformationPath& path, float progress) override;
//     void startTransformationAnimation(const TransformationPath& path) override;
//     void stopAnimation() override;
//     bool isAnimating() const override;
//     void handleEvents() override;
//     std::vector<RobotPosition> calculateLayout(const ControlGraph& graph) override;
    
//     // 特定功能
//     void highlightRobot(int robot_id);
//     void highlightEdge(int from_robot, int to_robot);
//     void clearHighlights();
    
//     // 自定义布局
//     void setCustomLayout(const std::vector<RobotPosition>& positions);
    
// private:
//     // 动画状态
//     struct AnimationState {
//         TransformationPath path;
//         float current_progress;
//         bool is_playing;
        
//         AnimationState() : current_progress(0.0f), is_playing(false) {}
//     };
    
//     AnimationState animation_state_;
//     std::vector<RobotPosition> custom_layout_;
    
//     // 渲染辅助方法
//     void renderRobot(const RobotPosition& pos);
//     void renderEdge(int from_robot, int to_robot, const std::vector<RobotPosition>& positions);
//     void renderInfoPanel(const ControlGraph& graph);
    
//     // 布局算法
//     std::vector<RobotPosition> circularLayout(const ControlGraph& graph);
//     std::vector<RobotPosition> forceDirectedLayout(const ControlGraph& graph);
//     std::vector<RobotPosition> hierarchicalLayout(const ControlGraph& graph);
// };

// // 队形枚举可视化器
// class FormationEnumerationVisualizer : public Visualizer {
// public:
//     FormationEnumerationVisualizer(const VisualizationConfig& config = VisualizationConfig());
    
//     bool initialize() override;
//     void render(const ControlGraph& graph) override;
//     void renderTransformation(const TransformationPath& path, float progress) override;
//     void startTransformationAnimation(const TransformationPath& path) override;
//     void stopAnimation() override;
//     bool isAnimating() const override;
//     void handleEvents() override;
//     std::vector<RobotPosition> calculateLayout(const ControlGraph& graph) override;
    
//     // 特定功能
//     void setFormations(const std::vector<ControlGraph>& formations);
//     void setCurrentFormationIndex(int index);
//     void showFormationComparison(const ControlGraph& graph1, const ControlGraph& graph2);
    
// private:
//     std::vector<ControlGraph> formations_;
//     int current_formation_index_;
    
//     void renderFormationGrid();
//     void renderFormationDetails(const ControlGraph& graph);
// };

// // 变换路径可视化器
// class TransformationPathVisualizer : public Visualizer {
// public:
//     TransformationPathVisualizer(const VisualizationConfig& config = VisualizationConfig());
    
//     bool initialize() override;
//     void render(const ControlGraph& graph) override;
//     void renderTransformation(const TransformationPath& path, float progress) override;
//     void startTransformationAnimation(const TransformationPath& path) override;
//     void stopAnimation() override;
//     bool isAnimating() const override;
//     void handleEvents() override;
//     std::vector<RobotPosition> calculateLayout(const ControlGraph& graph) override;
    
//     // 特定功能
//     void setTransformationPath(const TransformationPath& path);
//     void showCostAnalysis(const TransformationPath& path);
//     void showAlternativePaths(const std::vector<TransformationPath>& paths);
    
// private:
//     TransformationPath current_path_;
    
//     void renderPathOverview();
//     void renderStepByStep(const TransformationPath& path, float progress);
//     void renderCostGraph(const TransformationPath& path);
// };

// // 可视化工厂
// class VisualizerFactory {
// public:
//     static std::unique_ptr<Visualizer> createControlGraphVisualizer(
//         const VisualizationConfig& config = VisualizationConfig());
    
//     static std::unique_ptr<Visualizer> createFormationEnumerationVisualizer(
//         const VisualizationConfig& config = VisualizationConfig());
    
//     static std::unique_ptr<Visualizer> createTransformationPathVisualizer(
//         const VisualizationConfig& config = VisualizationConfig());
// };

// } // namespace formation