#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "LidarSimulator.h"
#include "FeatureExtractor.h"
#include "FormationEvaluator.h"
#include "formation_evaluation/formation_evaluator.h"

namespace py = pybind11;

// 封装C++类供Python调用
PYBIND11_MODULE(formation_core, m) {
    m.doc() = "Formation learning core module";
    
    // 绑定基本数据结构
    py::class_<Point2D>(m, "Point2D") //绑定Point2D结构体
        .def(py::init<double, double>(), py::arg("x")=0.0, py::arg("y")=0.0)
        .def_readwrite("x", &Point2D::x)
        .def_readwrite("y", &Point2D::y);
    
    py::class_<RobotState>(m, "RobotState") //绑定RobotState结构体
        .def(py::init<double, double, double>())
        .def_readwrite("position", &RobotState::position)
        .def_readwrite("orientation", &RobotState::orientation);

    py::class_<Environment>()(m, "Environment") //绑定Environment结构体
        .def(py::init<const std::vector<std::vector<int>>&, const RobotState&, double, double>(), 
             py::arg("grid_map"), py::arg("leader_state"), py::arg("safety_threshold"), py::arg("max_comm_distance"))
        .def_readwrite("grid_map", &Environment::gridMap)
        .def_readwrite("leader_state", &Environment::leaderState)
        .def_readwrite("safety_threshold", &Environment::safetyThreshold)
        .def_readwrite("max_comm_distance", &Environment::maxCommDistance)
        .def_readwrite("lidar_scan", &Environment::lidarScan)
        .def_readwrite("corridor_width", &Environment::corridorWidth);

    py::class_<FormationConfig>(m, "FormationConfig") //绑定FormationConfig结构体
        .def(py::init<>())
        .def_readwrite("control_graph", &FormationConfig::controlGraph)
        .def_readwrite("positions", &FormationConfig::positions); 

    py::class_<ControlGraph>(m, "ControlGraph") //绑定ControlGraph类
        .def(py::init<double, double, double>())
        .def_readwrite("position", &RobotState::position)
        .def_readwrite("orientation", &RobotState::orientation);
    
    // 绑定激光雷达仿真器
    py::class_<LidarSimulator>(m, "LidarSimulator") //绑定LidarSimulator类
        .def(py::init<>())
        .def("set_grid_map", &LidarSimulator::setGridMap)
        .def("simulate_scan", &LidarSimulator::simulateScan);
    
    // 绑定特征提取器
    py::class_<FeatureExtractor>(m, "FeatureExtractor")
        .def(py::init<int>())
        .def("extract_features", &FeatureExtractor::extractFeatures);
    
    // 绑定评估函数
    py::class_<FormationEvaluator>(m, "FormationEvaluator")
        .def(py::init<double, double, double>(), 
             py::arg("safety_threshold")=0.4, py::arg("max_comm_distance")=0.3, py::arg("corridor_width")=0.3)
        .def("evaluate_formation", &FormationEvaluator::evaluateFormation);
    
    // 绑定环境生成器
    py::class_<EnvironmentGenerator>(m, "EnvironmentGenerator")
        .def(py::init<>())
        .def("create_corridor", &EnvironmentGenerator::createCorridor)
        .def("create_open_space", &EnvironmentGenerator::createOpenSpace)
        .def("create_complex_obstacles", &EnvironmentGenerator::createComplexObstacles);

    // 编队控制图枚举
    py::class_<FormationEnumerator>(m, "FormationEnumerator") //绑定FormationEnumerator类
        .def(py::init<int>())
        .def("get_all_formations", &FormationEnumerator::enumerateAllFormations);
        .def("get_gragh_nums", &FormationEnumerator::getControlGraghNums);

}