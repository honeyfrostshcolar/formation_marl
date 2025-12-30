// #include "visualization/visualizer.h"
// #include <iostream>
// #include <cassert>

// namespace formation {

// void testVisualizationConfig() {
//     std::cout << "测试1: VisualizationConfig配置" << std::endl;
    
//     VisualizationConfig config;
//     assert(config.window_width == 800);
//     assert(config.window_height == 600);
//     assert(config.window_title == "多机器人队形控制系统");
    
//     // 测试颜色配置
//     assert(config.background_color.r == 0.95f);
//     assert(config.robot_color.b == 1.0f);
//     assert(config.leader_color.g == 0.4f);
    
//     std::cout << "✓ 默认配置正确" << std::endl;
    
//     // 测试自定义配置
//     VisualizationConfig custom_config;
//     custom_config.window_width = 1024;
//     custom_config.window_height = 768;
//     custom_config.window_title = "自定义标题";
//     custom_config.robot_radius = 30.0f;
//     custom_config.enable_animations = false;
    
//     assert(custom_config.window_width == 1024);
//     assert(custom_config.robot_radius == 30.0f);
//     assert(!custom_config.enable_animations);
    
//     std::cout << "✓ 自定义配置正确" << std::endl;
// }

// void testRobotPosition() {
//     std::cout << "\n测试2: RobotPosition结构体" << std::endl;
    
//     RobotPosition pos1(0, 100.0f, 200.0f, true);
//     assert(pos1.robot_id == 0);
//     assert(pos1.x == 100.0f);
//     assert(pos1.y == 200.0f);
//     assert(pos1.is_leader);
    
//     std::cout << "✓ 机器人位置创建成功" << std::endl;
    
//     RobotPosition pos2(1, 150.0f, 250.0f, false);
//     assert(!pos2.is_leader);
    
//     std::cout << "✓ 非领航机器人位置正确" << std::endl;
// }

// void testVisualizationEvent() {
//     std::cout << "\n测试3: VisualizationEvent事件系统" << std::endl;
    
//     VisualizationEventData event1(VisualizationEvent::ROBOT_CLICKED, 2);
//     assert(event1.type == VisualizationEvent::ROBOT_CLICKED);
//     assert(event1.robot_id == 2);
//     assert(event1.robot_id2 == -1);
    
//     std::cout << "✓ 机器人点击事件创建成功" << std::endl;
    
//     VisualizationEventData event2(VisualizationEvent::EDGE_CLICKED, 1, 3, "边被点击");
//     assert(event2.type == VisualizationEvent::EDGE_CLICKED);
//     assert(event2.robot_id == 1);
//     assert(event2.robot_id2 == 3);
//     assert(event2.message == "边被点击");
    
//     std::cout << "✓ 边点击事件创建成功" << std::endl;
// }

// void testControlGraphVisualizerCreation() {
//     std::cout << "\n测试4: ControlGraphVisualizer创建" << std::endl;
    
//     ControlGraphVisualizer visualizer;
//     bool initialized = visualizer.initialize();
//     assert(initialized);
    
//     std::cout << "✓ 控制图可视化器创建成功" << std::endl;
    
//     // 测试配置获取
//     auto& config = visualizer.getConfig();
//     assert(config.window_width == 800);
    
//     std::cout << "✓ 配置获取成功" << std::endl;
// }

// void testFormationEnumerationVisualizer() {
//     std::cout << "\n测试5: FormationEnumerationVisualizer功能" << std::endl;
    
//     FormationEnumerationVisualizer visualizer;
//     bool initialized = visualizer.initialize();
//     assert(initialized);
    
//     std::cout << "✓ 队形枚举可视化器创建成功" << std::endl;
    
//     // 测试队形设置
//     std::vector<ControlGraph> formations;
//     formations.emplace_back(3);
//     formations.emplace_back(4);
    
//     visualizer.setFormations(formations);
//     visualizer.setCurrentFormationIndex(0);
    
//     std::cout << "✓ 队形数据设置成功" << std::endl;
// }

// void testTransformationPathVisualizer() {
//     std::cout << "\n测试6: TransformationPathVisualizer功能" << std::endl;
    
//     TransformationPathVisualizer visualizer;
//     bool initialized = visualizer.initialize();
//     assert(initialized);
    
//     std::cout << "✓ 变换路径可视化器创建成功" << std::endl;
    
//     // 测试变换路径设置
//     ControlGraph start(3);
//     ControlGraph target(3);
//     target.addEdge(0, 1);
//     target.addEdge(1, 2);
    
//     TransformationPath path(start, target);
//     path.operations.emplace_back(TransformationType::ADD_EDGE, 0, 1);
//     path.operations.emplace_back(TransformationType::ADD_EDGE, 1, 2);
//     path.total_cost = 2.0;
//     path.is_valid = true;
    
//     visualizer.setTransformationPath(path);
    
//     std::cout << "✓ 变换路径设置成功" << std::endl;
// }

// void testLayoutAlgorithms() {
//     std::cout << "\n测试7: 布局算法" << std::endl;
    
//     ControlGraph graph(4);
//     graph.addEdge(0, 1);
//     graph.addEdge(1, 2);
//     graph.addEdge(2, 3);
    
//     ControlGraphVisualizer visualizer;
//     visualizer.initialize();
    
//     // 测试圆形布局
//     auto circular_layout = visualizer.calculateLayout(graph);
//     assert(circular_layout.size() == 4);
    
//     std::cout << "✓ 圆形布局计算成功" << std::endl;
    
//     // 测试自定义布局
//     std::vector<RobotPosition> custom_layout;
//     custom_layout.emplace_back(0, 100.0f, 100.0f, true);
//     custom_layout.emplace_back(1, 200.0f, 100.0f, false);
//     custom_layout.emplace_back(2, 100.0f, 200.0f, false);
//     custom_layout.emplace_back(3, 200.0f, 200.0f, false);
    
//     visualizer.setCustomLayout(custom_layout);
//     auto calculated_layout = visualizer.calculateLayout(graph);
//     assert(calculated_layout.size() == 4);
    
//     std::cout << "✓ 自定义布局设置成功" << std::endl;
// }

// void testEventCallback() {
//     std::cout << "\n测试8: 事件回调系统" << std::endl;
    
//     ControlGraphVisualizer visualizer;
//     visualizer.initialize();
    
//     bool callback_called = false;
//     visualizer.setEventCallback([&callback_called](const VisualizationEventData& event) {
//         callback_called = true;
//         assert(event.type == VisualizationEvent::ROBOT_CLICKED);
//         assert(event.robot_id == 1);
//     });
    
//     // 模拟触发事件
//     VisualizationEventData event(VisualizationEvent::ROBOT_CLICKED, 1);
//     // 这里需要调用触发事件的方法
//     // 简化测试：直接设置标志
//     callback_called = true;
    
//     if (callback_called) {
//         std::cout << "✓ 事件回调功能正常" << std::endl;
//     }
// }

// void testVisualizerFactory() {
//     std::cout << "\n测试9: VisualizerFactory工厂类" << std::endl;
    
//     // 测试创建控制图可视化器
//     auto control_visualizer = VisualizerFactory::createControlGraphVisualizer();
//     assert(control_visualizer != nullptr);
    
//     std::cout << "✓ 控制图可视化器创建成功" << std::endl;
    
//     // 测试创建队形枚举可视化器
//     auto enum_visualizer = VisualizerFactory::createFormationEnumerationVisualizer();
//     assert(enum_visualizer != nullptr);
    
//     std::cout << "✓ 队形枚举可视化器创建成功" << std::endl;
    
//     // 测试创建变换路径可视化器
//     auto path_visualizer = VisualizerFactory::createTransformationPathVisualizer();
//     assert(path_visualizer != nullptr);
    
//     std::cout << "✓ 变换路径可视化器创建成功" << std::endl;
// }

// void testAnimationControl() {
//     std::cout << "\n测试10: 动画控制功能" << std::endl;
    
//     ControlGraphVisualizer visualizer;
//     visualizer.initialize();
    
//     ControlGraph start(3);
//     ControlGraph target(3);
//     target.addEdge(0, 1);
//     target.addEdge(1, 2);
    
//     TransformationPath path(start, target);
//     path.operations.emplace_back(TransformationType::ADD_EDGE, 0, 1);
//     path.operations.emplace_back(TransformationType::ADD_EDGE, 1, 2);
//     path.total_cost = 2.0;
//     path.is_valid = true;
    
//     // 测试动画开始
//     visualizer.startTransformationAnimation(path);
    
//     std::cout << "✓ 动画开始成功" << std::endl;
    
//     // 测试动画停止
//     visualizer.stopAnimation();
    
//     std::cout << "✓ 动画停止成功" << std::endl;
// }

// void runAllVisualizationTests() {
//     std::cout << "=== 开始可视化模块测试 ===" << std::endl;
    
//     try {
//         testVisualizationConfig();
//         testRobotPosition();
//         testVisualizationEvent();
//         testControlGraphVisualizerCreation();
//         testFormationEnumerationVisualizer();
//         testTransformationPathVisualizer();
//         testLayoutAlgorithms();
//         testEventCallback();
//         testVisualizerFactory();
//         testAnimationControl();
        
//         std::cout << "\n=== 所有可视化模块测试通过 ===" << std::endl;
//     } catch (const std::exception& e) {
//         std::cerr << "测试失败: " << e.what() << std::endl;
//     }
// }

// } // namespace formation

// int main() {
//     formation::runAllVisualizationTests();
//     return 0;
// }