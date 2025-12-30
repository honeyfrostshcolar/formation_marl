#include "simenv2gendate/EnvironmentGenerator.h"
#include <random>
#include <algorithm>

EnvironmentGenerator::EnvironmentGenerator() {
    std::random_device rd;
    rng_.seed(rd());
}

std::vector<std::vector<int>> EnvironmentGenerator::createCorridor(int width, int height) {
    std::vector<std::vector<int>> grid(height, std::vector<int>(width, 0));
    
    // 创建两侧墙壁
    int wall_width = 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (x < wall_width || x >= width - wall_width) {
                grid[y][x] = 1;  // 墙壁
            }
        }
    }
    
    return grid;
}

std::vector<std::vector<int>> EnvironmentGenerator::createOpenSpace(int width, int height) {
    std::vector<std::vector<int>> grid(height, std::vector<int>(width, 0));
    
    // 只在边缘放置障碍物
    int border = 3;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (y < border || y >= height - border || x < border || x >= width - border) {
                grid[y][x] = 1;
            }
        }
    }
    
    return grid;
}

std::vector<std::vector<int>> EnvironmentGenerator::createComplexObstacles(int width, int height) {
    std::vector<std::vector<int>> grid(height, std::vector<int>(width, 0));
    
    std::uniform_int_distribution<int> dist_x(0, width - 1);
    std::uniform_int_distribution<int> dist_y(0, height - 1);
    std::uniform_int_distribution<int> dist_size(1, 5);
    
    // 随机添加一些障碍物
    int num_obstacles = 10;
    for (int i = 0; i < num_obstacles; ++i) {
        int center_x = dist_x(rng_);
        int center_y = dist_y(rng_);
        int size = dist_size(rng_);
        
        for (int dy = -size; dy <= size; ++dy) {
            for (int dx = -size; dx <= size; ++dx) {
                int x = center_x + dx;
                int y = center_y + dy;
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    if (dx * dx + dy * dy <= size * size) {
                        grid[y][x] = 1;
                    }
                }
            }
        }
    }
    
    return grid;
}

std::vector<std::vector<int>> EnvironmentGenerator::createRandomEnvironment(int width, int height) {
    std::uniform_int_distribution<int> dist_type(0, 2);
    int type = dist_type(rng_);
    
    switch (type) {
        case 0: return createCorridor(width, height);
        case 1: return createOpenSpace(width, height);
        case 2: return createComplexObstacles(width, height);
        default: return createOpenSpace(width, height);
    }
}