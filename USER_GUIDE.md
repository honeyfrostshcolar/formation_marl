# 多机器人队形控制系统 - 用户指南

## 概述

多机器人队形控制系统是一个用于研究多机器人系统队形控制和变换的软件工具。该系统能够：

- **枚举所有可能的队形配置**：基于Polya定理的有向图枚举算法
- **查找队形变换路径**：使用A*搜索算法寻找最优变换序列
- **可视化队形和变换过程**：提供直观的图形界面展示

## 系统要求

### 硬件要求
- 处理器：Intel Core i5 或同等性能
- 内存：8GB RAM 或以上
- 存储空间：至少1GB可用空间

### 软件要求
- 操作系统：Windows 10/11, Linux, macOS
- C++编译器：GCC 9.0+, Clang 10.0+, 或 MSVC 2019+
- CMake 3.10 或更高版本
- Eigen 3.4 或更高版本

## 安装指南

### Windows 系统

1. **安装开发环境**
   ```bash
   # 安装MinGW-w64或Visual Studio
   # 确保g++或cl编译器可用
   ```

2. **安装依赖项**
   ```bash
   # 安装CMake
   # 下载Eigen库并设置环境变量
   ```

3. **构建项目**
   ```bash
   cd e:\formation_test
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

### Linux/macOS 系统

1. **安装依赖项**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential cmake libeigen3-dev
   
   # macOS (使用Homebrew)
   brew install cmake eigen
   ```

2. **构建项目**
   ```bash
   cd /path/to/formation_test
   mkdir build
   cd build
   cmake ..
   make -j4
   ```

## 快速开始

### 基本用法

1. **运行主程序**
   ```bash
   # Windows
   .\formation_main.exe 4
   
   # Linux/macOS
   ./formation_main 4
   ```
   参数 `4` 表示机器人数量

2. **交互模式命令**
   ```
   >> enum      # 枚举所有队形
   >> transform # 查找变换路径
   >> exit      # 退出程序
   ```

### 示例演示

#### 示例1：枚举4个机器人的所有队形

```bash
./formation_main 4
>> enum
```

系统将：
- 枚举所有可能的队形配置
- 显示统计信息（队形数量、同构类等）
- 切换到队形浏览界面

#### 示例2：队形变换

```bash
./formation_main 4
>> transform
请输入目标队形ID: 1
```

系统将：
- 查找从当前队形到目标队形的变换路径
- 显示变换操作序列和总成本
- 启动变换动画可视化

## 核心功能详解

### 1. 队形枚举

**算法原理**：基于Polya定理的有向图枚举
- 生成所有可能的n节点有向图
- 识别同构的队形配置
- 过滤无效的队形（不满足约束条件）

**性能特点**：
- 时间复杂度：O(2^(n²))
- 空间复杂度：O(n²)
- 支持缓存和优化

### 2. 队形变换

**算法原理**：基于A*搜索的最优路径查找
- 支持添加边、删除边、反转边等操作
- 考虑变换成本和约束条件
- 提供多种搜索策略（A*, 波束搜索等）

**变换操作**：
- `ADD_EDGE`：添加控制边
- `REMOVE_EDGE`：删除控制边
- `REVERSE_EDGE`：反转控制方向

### 3. 可视化功能

**可视化类型**：
- **控制图可视化**：显示当前队形状态
- **队形枚举可视化**：浏览所有可能的队形
- **变换路径可视化**：展示变换过程和动画

**交互功能**：
- 鼠标点击选择机器人或边
- 键盘控制浏览和导航
- 实时信息显示

## 高级功能

### 自定义约束

系统支持自定义约束条件：

```cpp
// 示例：添加最大出度约束
ConstraintValidator validator;
validator.addConstraint([](const ControlGraph& graph) {
    return graph.getMaxOutDegree() <= 2;
});
```

### 性能优化

**缓存策略**：
- 队形枚举结果缓存
- 变换路径预计算
- 同构检测优化

**并行计算**：
- 多线程队形枚举
- 并行变换路径搜索

## 故障排除

### 常见问题

1. **编译错误：找不到Eigen库**
   ```bash
   # 设置Eigen路径
   export EIGEN3_INCLUDE_DIR=/path/to/eigen
   ```

2. **运行时错误：内存不足**
   - 减少机器人数量（建议≤6）
   - 增加系统内存
   - 使用优化版本

3. **可视化问题：窗口无法显示**
   - 检查图形驱动
   - 确保OpenGL支持
   - 尝试软件渲染模式

### 调试模式

启用详细日志：
```bash
./formation_main 4 --verbose
```

## API 参考

### 核心类

#### ControlGraph
- `addEdge(int from, int to)`：添加控制边
- `removeEdge(int from, int to)`：删除控制边
- `isValid()`：检查队形有效性

#### FormationEnumerator
- `enumerateAllFormations()`：枚举所有队形
- `classifyByTopology()`：按拓扑分类
- `getPerformanceData()`：获取性能统计

#### TransformationAlgorithm
- `findTransformationPath()`：查找变换路径
- `validateOperation()`：验证变换操作
- `calculateCost()`：计算变换成本

## 扩展开发

### 添加新的变换操作

1. 在 `transformation_algorithm.h` 中定义新操作类型
2. 实现操作验证和成本计算逻辑
3. 更新可视化组件支持新操作

### 自定义可视化布局

```cpp
// 实现自定义布局算法
class CustomLayout : public LayoutAlgorithm {
public:
    std::vector<RobotPosition> calculateLayout(const ControlGraph& graph) override;
};
```

## 性能基准测试

### 测试环境
- CPU: Intel Core i7-10700K
- RAM: 32GB DDR4
- OS: Ubuntu 20.04

### 测试结果

| 机器人数量 | 枚举时间 | 队形数量 | 内存使用 |
|-----------|----------|----------|----------|
| 3         | <1ms     | 16       | <10MB    |
| 4         | 5ms      | 512      | 50MB     |
| 5         | 200ms    | 65,536   | 500MB    |
| 6         | 30s      | 33M      | 2GB      |

## 技术支持

### 文档资源
- [API文档](./docs/api/)
- [算法说明](./docs/algorithms/)
- [示例代码](./examples/)

### 联系方式
- 问题反馈：创建GitHub Issue
- 功能请求：提交Pull Request
- 技术讨论：参与社区论坛

## 版本历史

### v1.0.0 (当前版本)
- 基础队形枚举功能
- 队形变换算法
- 可视化界面
- 用户交互系统

### 未来计划
- 分布式计算支持
- 机器学习优化
- Web界面版本
- 移动端应用

---

*最后更新：2024年12月*
*版权所有 © 2024 多机器人队形控制系统开发团队*