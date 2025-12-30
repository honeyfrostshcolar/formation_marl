#include "simenv2gendate/LidarSimulator.h"
#include <cmath>
#include <algorithm>

LidarSimulator::LidarSimulator() : map_width_(0), map_height_(0) {}

void LidarSimulator::setGridMap(const std::vector<std::vector<int>>& grid_map, double resolution) {
    grid_map_ = grid_map;
    resolution_ = resolution;
    map_height_ = grid_map.size();
    if (map_height_ > 0) {
        map_width_ = grid_map[0].size();
    }
}

double LidarSimulator::rayCast(double start_x, double start_y, double angle, double max_range) {
    const double step_size = 0.1;
    double current_distance = 0;
    
    while (current_distance < max_range) {
        double check_x = start_x + current_distance * std::cos(angle);
        double check_y = start_y + current_distance * std::sin(angle);
        
        int grid_x = static_cast<int>(std::round(check_x / resolution_));
        int grid_y = static_cast<int>(std::round(-check_y / resolution_));
        
        // 检查边界
        if (grid_x < 0 || grid_x >= map_width_ || grid_y < 0 || grid_y >= map_height_) {
            return max_range;
        }
        
        // 检查障碍物
        if (grid_map_[grid_y][grid_x] == 1) {
            return current_distance;
        }
        
        current_distance += step_size;
    }
    
    return max_range;
}

LidarData LidarSimulator::simulateScan(double robot_x, double robot_y, double robot_theta, 
                                      int num_beams, double max_range) {
    

    LidarData lidar_data(num_beams, max_range);
    
    for (int i = 0; i < num_beams; ++i) {
        double angle = robot_theta + lidar_data.angles[i]; //以机器人当前角度为0，计算每个光束的角度
        lidar_data.ranges[i] = rayCast(robot_x, robot_y, angle, max_range);
    }
    
    return lidar_data;
}