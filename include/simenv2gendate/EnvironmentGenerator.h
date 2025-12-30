#ifndef ENVIRONMENT_GENERATOR_H
#define ENVIRONMENT_GENERATOR_H

#include <vector>
#include <random>

class EnvironmentGenerator {
private:
    std::mt19937 rng_; //对象rng_是一个随机数引擎，可用来产生高质量的随机整数
    
public:
    EnvironmentGenerator();
    
    // 创建走廊环境
    std::vector<std::vector<int>> createCorridor(int width, int height);
    
    // 创建开阔环境
    std::vector<std::vector<int>> createOpenSpace(int width, int height);
    
    // 创建复杂障碍环境
    std::vector<std::vector<int>> createComplexObstacles(int width, int height);
    
    // 创建随机环境
    std::vector<std::vector<int>> createRandomEnvironment(int width, int height);
};

#endif