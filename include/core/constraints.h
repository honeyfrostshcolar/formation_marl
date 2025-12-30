#ifndef FORMATION_CONTROL_CONSTRAINTS_H
#define FORMATION_CONTROL_CONSTRAINTS_H

#include "core/control_graph.h"
#include <vector>
#include <string>

namespace formation {

/**
 * @brief 约束验证结果结构体（提前定义，供 ConstraintValidator 返回使用）
 */
struct ValidationResult {
    bool is_valid;                          // 是否有效
    bool leader_constraint_violated;       // 领航机器人约束违反
    bool label_order_constraint_violated;  // 标签顺序约束违反
    bool control_dimension_constraint_violated; // 控制维度约束违反
    std::vector<std::string> details;      // 详细错误信息
    
    ValidationResult() 
        : is_valid(true)
        , leader_constraint_violated(false)
        , label_order_constraint_violated(false)
        , control_dimension_constraint_violated(false) {}
};

/**
 * @brief 约束验证器类，专门用于验证控制图的各种约束
 */
class ConstraintValidator {
public:
    /**
     * @brief 验证控制图是否满足所有约束
     * @param graph 控制图
     * @return 验证结果
     */
    static ValidationResult validate(const ControlGraph& graph);
    
    /**
     * @brief 验证单个变换步骤是否合法
     * @param from 原始控制图
     * @param to 目标控制图
     * @return 是否合法
     */
    static bool validateTransformationStep(const ControlGraph& from, const ControlGraph& to);
    
    /**
     * @brief 获取详细的约束违反信息
     * @param graph 控制图
     * @return 违反的约束信息列表
     */
    static std::vector<std::string> getViolationDetails(const ControlGraph& graph);
    
    /**
     * @brief 检查控制图是否连通
     * @param graph 控制图
     * @return 是否连通
     */
    static bool isConnected(const ControlGraph& graph);
    
    /**
     * @brief 检查控制图是否是无环图
     * @param graph 控制图
     * @return 是否无环
     */
    static bool isAcyclic(const ControlGraph& graph);
    
    /**
     * @brief 计算控制图的深度（最长路径长度）
     * @param graph 控制图
     * @return 图的深度
     */
    static int calculateDepth(const ControlGraph& graph);
    
    /**
     * @brief 检查两个控制图是否可以通过单步变换相互转换
     * @param graph1 第一个控制图
     * @param graph2 第二个控制图
     * @return 是否可以通过单步变换转换
     */
    static bool canTransformInOneStep(const ControlGraph& graph1, const ControlGraph& graph2);
};

} // namespace formation

#endif // FORMATION_CONTROL_CONSTRAINTS_H