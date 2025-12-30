// #include "core/control_graph.h"
// #include "enumeration/formation_enumeration.h"
// #include "transformation/transformation_algorithm.h"
// #include "visualization/visualizer.h"
// #include <iostream>
// #include <vector>
// #include <memory>
// #include <chrono> // 添加：修复使用 std::chrono 但未包含头的问题

// namespace formation {

// class FormationControlSystem {
// private:
//     std::unique_ptr<ControlGraph> current_graph_;
//     std::unique_ptr<FormationEnumerator> enumerator_;
//     std::unique_ptr<TransformationAlgorithm> transformer_;
//     std::unique_ptr<Visualizer> visualizer_;
    
//     int robot_count_;
//     bool is_initialized_;

// public:
//     FormationControlSystem() : robot_count_(0), is_initialized_(false) {}
    
//     bool initialize(int robot_count) {
//         if (robot_count < 2) {
//             std::cerr << "错误: 机器人数量必须至少为2" << std::endl;
//             return false;
//         }
        
//         robot_count_ = robot_count;
        
//         try {
//             // 创建控制图
//             current_graph_ = std::make_unique<ControlGraph>(robot_count_);
            
//             // 创建队形枚举器
//             enumerator_ = std::make_unique<FormationEnumerator>(robot_count_);
            
//             // 创建变换算法
//             transformer_ = std::make_unique<TransformationAlgorithm>();
            
//             // 创建可视化器
//             visualizer_ = VisualizerFactory::createControlGraphVisualizer();
//             if (!visualizer_->initialize()) {
//                 std::cerr << "错误: 可视化器初始化失败" << std::endl;
//                 return false;
//             }
            
//             // 设置事件回调
//             visualizer_->setEventCallback([this](const VisualizationEventData& event) {
//                 handleVisualizationEvent(event);
//             });
            
//             is_initialized_ = true;
//             std::cout << "系统初始化成功，机器人数量: " << robot_count_ << std::endl;
//             return true;
            
//         } catch (const std::exception& e) {
//             std::cerr << "初始化错误: " << e.what() << std::endl;
//             return false;
//         }
//     }
    
//     void enumerateFormations() {
//         if (!is_initialized_) {
//             std::cerr << "错误: 系统未初始化" << std::endl;
//             return;
//         }
        
//         std::cout << "开始枚举所有可能的队形..." << std::endl;
        
//         auto start_time = std::chrono::high_resolution_clock::now();
        
//         // 枚举所有队形
//         auto all_formations = enumerator_->enumerateAllFormations();
        
//         auto end_time = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
//         std::cout << "枚举完成! 共找到 " << all_formations.size() << " 个队形" << std::endl;
//         std::cout << "耗时: " << duration.count() << " 毫秒" << std::endl;
        
//         // 显示统计信息
//         auto stats = enumerator_->getPerformanceData();
//         std::cout << "同构类数量: " << stats.isomorphism_classes << std::endl;
//         std::cout << "有效队形数量: " << stats.valid_formations << std::endl;
        
//         // 切换到队形枚举可视化器
//         auto enum_visualizer = VisualizerFactory::createFormationEnumerationVisualizer();
//         if (enum_visualizer->initialize()) {
//             enum_visualizer->setFormations(all_formations);
//             enum_visualizer->setCurrentFormationIndex(0);
            
//             // 设置事件回调
//             enum_visualizer->setEventCallback([this](const VisualizationEventData& event) {
//                 handleFormationSelection(event);
//             });
            
//             visualizer_ = std::move(enum_visualizer);
//         }
//     }
    
//     void findTransformationPath(const ControlGraph& target) {
//         if (!is_initialized_) {
//             std::cerr << "错误: 系统未初始化" << std::endl;
//             return;
//         }
        
//         std::cout << "寻找从当前队形到目标队形的变换路径..." << std::endl;
        
//         auto start_time = std::chrono::high_resolution_clock::now();
        
//         // 查找变换路径
//         auto path = transformer_->findTransformationPath(*current_graph_, target);
        
//         auto end_time = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
//         if (path.is_valid) {
//             std::cout << "找到有效变换路径!" << std::endl;
//             std::cout << "变换操作数量: " << path.operations.size() << std::endl;
//             std::cout << "总成本: " << path.total_cost << std::endl;
//             std::cout << "耗时: " << duration.count() << " 毫秒" << std::endl;
            
//             // 切换到变换路径可视化器
//             auto path_visualizer = VisualizerFactory::createTransformationPathVisualizer();
//             if (path_visualizer->initialize()) {
//                 path_visualizer->setTransformationPath(path);
//                 path_visualizer->setCurrentGraph(*current_graph_);
//                 path_visualizer->setTargetGraph(target);
                
//                 // 设置事件回调
//                 path_visualizer->setEventCallback([this](const VisualizationEventData& event) {
//                     handleTransformationEvent(event);
//                 });
                
//                 visualizer_ = std::move(path_visualizer);
//             }
//         } else {
//             std::cout << "未找到有效变换路径" << std::endl;
//         }
//     }
    
//     void runInteractiveMode() {
//         if (!is_initialized_) {
//             std::cerr << "错误: 系统未初始化" << std::endl;
//             return;
//         }
        
//         std::cout << "进入交互模式..." << std::endl;
//         std::cout << "输入命令: 'enum' - 枚举队形, 'transform' - 变换队形, 'exit' - 退出" << std::endl;
        
//         std::string command;
//         while (true) {
//             std::cout << ">> ";
//             std::cin >> command;
            
//             if (command == "exit") {
//                 break;
//             } else if (command == "enum") {
//                 enumerateFormations();
//             } else if (command == "transform") {
//                 // 这里可以添加目标队形选择逻辑
//                 std::cout << "请输入目标队形ID: ";
//                 int target_id;
//                 std::cin >> target_id;
                
//                 // 创建简单的目标队形
//                 ControlGraph target(robot_count_);
//                 if (target_id == 1) {
//                     // 链式队形
//                     for (int i = 0; i < robot_count_ - 1; ++i) {
//                         target.addEdge(i, i + 1);
//                     }
//                 } else if (target_id == 2) {
//                     // 星形队形
//                     for (int i = 1; i < robot_count_; ++i) {
//                         target.addEdge(0, i);
//                     }
//                 } else {
//                     std::cout << "无效的目标队形ID" << std::endl;
//                     continue;
//                 }
                
//                 findTransformationPath(target);
//             } else {
//                 std::cout << "未知命令" << std::endl;
//             }
//         }
//     }
    
//     void visualizeCurrentState() {
//         if (!is_initialized_) {
//             std::cerr << "错误: 系统未初始化" << std::endl;
//             return;
//         }
        
//         std::cout << "开始可视化当前状态..." << std::endl;
        
//         // 渲染当前状态
//         visualizer_->render(*current_graph_);
//     }

// private:
//     void handleVisualizationEvent(const VisualizationEventData& event) {
//         switch (event.type) {
//             case VisualizationEvent::ROBOT_CLICKED:
//                 std::cout << "机器人 " << event.robot_id << " 被点击" << std::endl;
//                 break;
//             case VisualizationEvent::EDGE_CLICKED:
//                 std::cout << "边 (" << event.robot_id << " -> " << event.robot_id2 << ") 被点击" << std::endl;
//                 break;
//             case VisualizationEvent::INFO_PANEL_CLICKED:
//                 std::cout << "信息面板被点击: " << event.message << std::endl;
//                 break;
//             default:
//                 break;
//         }
//     }
    
//     void handleFormationSelection(const VisualizationEventData& event) {
//         if (event.type == VisualizationEvent::FORMATION_SELECTED) {
//             std::cout << "队形 " << event.robot_id << " 被选中" << std::endl;
            
//             // 这里可以添加队形选择后的处理逻辑
//             // 例如：显示队形详情，准备变换等
//         }
//     }
    
//     void handleTransformationEvent(const VisualizationEventData& event) {
//         if (event.type == VisualizationEvent::ANIMATION_STARTED) {
//             std::cout << "变换动画开始" << std::endl;
//         } else if (event.type == VisualizationEvent::ANIMATION_COMPLETED) {
//             std::cout << "变换动画完成" << std::endl;
//         }
//     }
// };

// void printUsage() {
//     std::cout << "多机器人队形控制系统" << std::endl;
//     std::cout << "用法: formation_control [机器人数量]" << std::endl;
//     std::cout << "示例: formation_control 4" << std::endl;
//     std::cout << std::endl;
//     std::cout << "功能:" << std::endl;
//     std::cout << "- 枚举所有可能的队形配置" << std::endl;
//     std::cout << "- 查找队形之间的变换路径" << std::endl;
//     std::cout << "- 可视化队形和变换过程" << std::endl;
// }

// } // namespace formation

// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         formation::printUsage();
//         return 1;
//     }
    
//     try {
//         int robot_count = std::stoi(argv[1]);
        
//         formation::FormationControlSystem system;
//         if (!system.initialize(robot_count)) {
//             return 1;
//         }
        
//         // 运行交互模式
//         system.runInteractiveMode();
        
//         std::cout << "程序正常结束" << std::endl;
//         return 0;
        
//     } catch (const std::exception& e) {
//         std::cerr << "程序错误: " << e.what() << std::endl;
//         return 1;
//     }
// }