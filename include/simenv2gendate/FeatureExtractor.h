#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "types.h"

class FeatureExtractor {
private:
    int num_sectors_; // 扇区数量
    
public:
    FeatureExtractor(int sectors = 8); // 默认8个扇区
    
    EnvironmentFeatures extractFeatures(const LidarData& lidar);
};

#endif