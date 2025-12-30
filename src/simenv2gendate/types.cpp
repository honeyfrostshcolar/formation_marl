#include "simenv2gendate/types.h"
#include <cmath>
#include <sstream>

// Point2D 方法实现
double Point2D::distanceTo(const Point2D& other) const {
    return std::sqrt(std::pow(x - other.x, 2) + std::pow(y - other.y, 2));
}

// RobotState 构造函数
RobotState::RobotState(double x, double y, double theta) 
    : position(x, y), orientation(theta) {}

// LidarData 构造函数
LidarData::LidarData(int beams, double max_r) 
    : num_beams(beams), max_range(max_r) {
    ranges.resize(num_beams, max_range);
    angles.resize(num_beams);
    for (int i = 0; i < num_beams; ++i) {
        angles[i] = 2 * M_PI * i / num_beams;
    }
}

// FormationConfig 构造函数
FormationConfig::FormationConfig() : graph_index(-1) {
    controlGraph.setZero();
}

// EnvironmentFeatures 方法实现
std::vector<double> EnvironmentFeatures::toVector() const {
    std::vector<double> features;
    features.push_back(corridor_width);
    features.push_back(front_clearance);
    features.push_back(left_clearance);
    features.push_back(right_clearance);
    features.push_back(obstacle_density);
    features.insert(features.end(), sector_min_dists.begin(), sector_min_dists.end());
    features.insert(features.end(), sector_avg_dists.begin(), sector_avg_dists.end());
    return features;
}

// TrainingSample 方法实现
std::string TrainingSample::serialize() const {
    std::ostringstream oss;
    oss << environment_type << "," 
        << leader_pose.position.x << ","
        << leader_pose.position.y << ","
        << leader_pose.orientation << ","
        << has_expert_label << ","
        << expert_confidence << ","
        << expert_formation.ctrlnums << ",";
        
    std::ostringstream matrix_oss;
    for (int row = 0; row < expert_formation.controlGraph.rows(); ++row) {
        for (int col = 0; col < expert_formation.controlGraph.cols(); ++col) {
            matrix_oss << expert_formation.controlGraph(row, col);
            if (col != expert_formation.controlGraph.cols() - 1) {
                matrix_oss << "|"; // 列间用竖线分隔
            }
        }
        if (row != expert_formation.controlGraph.rows() - 1) {
            matrix_oss << "；"; // 行间用分号分隔
        }
    }
    oss << matrix_oss.str() << ",";

    std::ostringstream positions_oss;
    for (int i = 0; i < expert_formation.positions.size(); ++i) {
        positions_oss << expert_formation.positions[i].x << "|" << expert_formation.positions[i].y;
        if (i != expert_formation.positions.size() - 1) {
            positions_oss << "；"; // 位置间用分号分隔
        }
    }
    oss << positions_oss.str();

    return oss.str();
}