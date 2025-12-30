#ifndef LIDAR_SIMULATOR_H
#define LIDAR_SIMULATOR_H

#include "types.h"
#include <vector>

class LidarSimulator {
private:
    std::vector<std::vector<int>> grid_map_;
    int map_width_, map_height_;
    double resolution_;
    
public:
    LidarSimulator();
    
    void setGridMap(const std::vector<std::vector<int>>& grid_map, double resolution);
    
    // 射线投射算法
    double rayCast(double start_x, double start_y, double angle, double max_range);
    
    // 模拟激光雷达扫描
    LidarData simulateScan(double robot_x, double robot_y, double robot_theta, 
                          int num_beams = 360, double max_range = 10.0);
};

#endif