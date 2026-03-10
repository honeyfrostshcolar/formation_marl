#include "simenv2gendate/FeatureExtractor.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

FeatureExtractor::FeatureExtractor(int sectors) : num_sectors_(sectors) {}

EnvironmentFeatures FeatureExtractor::extractFeatures(const LidarData& lidar) {
    EnvironmentFeatures features;
    int num_beams = lidar.ranges.size();
    int sector_size = num_beams / num_sectors_; // 每个扇区的扫描点数量
    
    //0号门是从车头顺时针开始数
    features.sector_min_dists.resize(num_sectors_);
    features.sector_avg_dists.resize(num_sectors_);
    
    // 计算各扇形区域的特征
    for (int i = 0; i < num_sectors_; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        double sum_dist = 0.0;
        int count = 0;
        
        // 这里要注意的是，扫描点的索引必须要和储存时候的索引对应上
        int start_idx = i * sector_size; // 当前扇区的第一个扫描点
        int end_idx = std::min((i + 1) * sector_size, num_beams); // 当前扇区的最后一个扫描点
        
        for (int j = start_idx; j < end_idx; ++j) {
            if (lidar.ranges[j] < lidar.max_range) {
                min_dist = std::min(min_dist, lidar.ranges[j]);
                sum_dist += lidar.ranges[j];
                count++;
            }
        }
        
        features.sector_min_dists[i] = (min_dist < lidar.max_range) ? min_dist : lidar.max_range; // 当前扇区的最小距离
        features.sector_avg_dists[i] = (count > 0) ? (sum_dist / count) : lidar.max_range; // 当前扇区的平均距离
        // if(i==7) std::cout << "7号扇区的平均距离：" << features.sector_avg_dists[i] << std::endl;
    }
    
    // 提取关键特征
    // 假设0号扇形是正前方，2号是左侧，6号是右侧
    features.front_clearance = std::min(features.sector_min_dists[0], features.sector_min_dists[7]); // 前方最小距离
    features.left_clearance = std::min(features.sector_min_dists[5], features.sector_min_dists[6]); // 左侧最小距离
    features.right_clearance = std::min(features.sector_min_dists[1], features.sector_min_dists[2]); // 右侧最小距离
    features.corridor_width = features.sector_min_dists[1] + features.sector_min_dists[6]; // 通道宽度
    
    // 计算障碍物密度（只检测的是30%范围内的障碍物，可根据训练情况调整）
    int close_count = 0;
    for (double range : lidar.ranges) {
        if (range < lidar.max_range * 0.3) {  // 30%范围内有障碍物
            close_count++;
        }
    }
    features.obstacle_density = static_cast<double>(close_count) / num_beams;
    
    return features;
}