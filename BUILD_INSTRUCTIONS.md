# 构建说明

## 环境要求

### 必需依赖
- **C++编译器**: 
  - GCC/G++ 7.0+ 或 
  - Microsoft Visual Studio 2019+ 或
  - Clang 6.0+
- **CMake 3.10+**
- **Eigen 3.3+** (线性代数库)

### 可选依赖
- **简单图形库** (如SFML或OpenGL基础实现) - 用于可视化
- **控制台图形输出** - 基础显示
- **简单测试框架** (可选) - 用于单元测试

## 安装指南

### Windows系统
1. **安装Visual Studio 2019或更高版本**
   - 下载并安装Visual Studio Community (免费)
   - 确保选择"使用C++的桌面开发"工作负载

2. **安装CMake**
   - 从 https://cmake.org/download/ 下载CMake
   - 安装时选择"Add CMake to the system PATH"

3. **安装Eigen库**
   ```cmd
   # 方法1: 使用vcpkg (推荐)
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   .\vcpkg install eigen3
   
   # 方法2: 手动下载
   # 从 https://eigen.tuxfamily.org/ 下载Eigen
   # 解压到项目目录或系统包含路径
   ```

### Linux系统
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake libeigen3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake eigen3-devel
```

### macOS系统
```bash
# 使用Homebrew
brew install cmake eigen
```

## 构建步骤

### 使用CMake构建
```bash
# 创建构建目录
mkdir build
cd build

# 配置项目
cmake ..

# 编译
cmake --build .

# 运行主程序
./bin/formation_control

# 运行测试
./bin/test_formation_control
```

### 手动编译（备用方案）
```bash
# 编译主程序
g++ -std=c++17 -I./include -I/path/to/eigen src/*.cpp src/core/*.cpp -o formation_control

# 编译测试程序
g++ -std=c++17 -I./include -I/path/to/eigen test/*.cpp src/core/*.cpp -o test_formation_control
```

## 项目结构验证

在构建之前，请确保项目结构如下：
```
formation_test/
├── src/
│   ├── core/
│   │   ├── control_graph.cpp
│   │   └── constraints.cpp
│   └── main.cpp
├── include/
│   └── core/
│       ├── control_graph.h
│       └── constraints.h
├── test/
│   └── test_main.cpp
├── CMakeLists.txt
└── README.md
```

## 故障排除

### 常见问题

1. **CMake找不到Eigen**
   ```bash
   # 指定Eigen路径
   cmake -DEigen3_DIR=/path/to/eigen ..
   ```

2. **编译错误：找不到头文件**
   - 确保Eigen库已正确安装
   - 检查包含路径设置

3. **链接错误**
   - 确保所有源文件都已添加到CMakeLists.txt
   - 检查编译器标准设置（C++17）

### 验证安装

运行以下命令验证环境：
```bash
# 检查编译器
g++ --version

# 检查CMake
cmake --version

# 检查Eigen（如果手动安装）
ls /usr/include/eigen3/  # Linux
ls /usr/local/include/eigen3/  # macOS
```

## 下一步

完成环境配置后，可以开始项目的第二阶段开发：队形枚举算法实现。