#!/usr/bin/env python3
import formation_core

# 测试枚举器
enumerator = formation_core.FormationEnumerator(3)
formations = enumerator.get_all_formations()
print(f"成功加载C++模块，生成{len(formations)}个控制图")

# 测试评估器
evaluator = formation_core.FormationEvaluator(0.4, 0.3, 0.3)
print("评估器初始化成功")
