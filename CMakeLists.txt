cmake_minimum_required(VERSION 3.10)
project(FormationControl VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# 设置编译选项
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -O2 -Wno-unused-parameter -Wno-sign-compare)
endif()

# 查找依赖库
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pybind11 REQUIRED)

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# ========== 核心库目标 ==========
# 只收集核心库的源文件，不包含 main.cpp
file(GLOB CORE_SRC_FILES 
    ${CMAKE_SOURCE_DIR}/src/core/*.cpp
    ${CMAKE_SOURCE_DIR}/src/enumeration/*.cpp
    ${CMAKE_SOURCE_DIR}/src/transformation/*.cpp
    ${CMAKE_SOURCE_DIR}/src/visualization/*.cpp
    ${CMAKE_SOURCE_DIR}/src/formationevaluation/*.cpp
    ${CMAKE_SOURCE_DIR}/src/simenv2gendate/*.cpp

)

# 创建静态库
add_library(formation_core_cpp STATIC ${CORE_SRC_FILES})
target_include_directories(formation_core_cpp PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(formation_core_cpp PUBLIC Eigen3::Eigen)

# ========== Python绑定模块 ==========
file(GLOB BIND_SRC_FILES ${CMAKE_SOURCE_DIR}/src/formationlearning/pybind11_interface.cpp)
pybind11_add_module(formation_core_py ${BIND_SRC_FILES})
target_link_libraries(formation_core_py PRIVATE formation_core_cpp Eigen3::Eigen)
set_target_properties(formation_core_py PROPERTIES OUTPUT_NAME "formation_core")
set_target_properties(formation_core_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/script
)


# ========== 主程序 ==========
add_executable(formation_main ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries(formation_main formation_core_cpp)
set_target_properties(formation_main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(simenv2gen_main ${CMAKE_SOURCE_DIR}/src/simenv2gendate/main.cpp)
target_link_libraries(simenv2gen_main formation_core_cpp)
set_target_properties(simenv2gen_main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ========== 测试配置 ==========
# option(BUILD_TESTS "Build test executables" ON)

# if(BUILD_TESTS)
#     # 查找 Google Test
#     find_package(GTest REQUIRED)
    
#     enable_testing()

#     # 手动指定测试文件，避免使用 GLOB
#     set(TEST_SOURCES
#         ${CMAKE_SOURCE_DIR}/test/test_main.cpp
#         ${CMAKE_SOURCE_DIR}/test/test_enumeration.cpp
#         ${CMAKE_SOURCE_DIR}/test/test_transformation.cpp
        
#     )

#     foreach(test_src ${TEST_SOURCES})
#         get_filename_component(test_name ${test_src} NAME_WE)

#         # 只链接核心库，不包含主程序源文件
#         add_executable(${test_name} ${test_src})
#         target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
#         target_link_libraries(${test_name} 
#             PRIVATE 
#             formation_core_cpp
#             GTest::gtest 
#             GTest::gtest_main
#         )

#         set_target_properties(${test_name} PROPERTIES 
#             RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
#         )

#         # 注册为 CTest 测试
#         add_test(NAME ${test_name} COMMAND ${test_name})
        
#         # 添加测试发现（Google Test 专用）
#         gtest_discover_tests(${test_name}
#             WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
#         )
#     endforeach()
# endif()
