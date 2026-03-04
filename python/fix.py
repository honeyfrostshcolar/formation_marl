#!/usr/bin/env python3
"""
修复pkg_resources._vendor导入问题
"""

import sys
import os

# 添加打印信息以调试
print("=" * 60)
print("开始修复导入问题")
print("=" * 60)

# 1. 检查pkg_resources模块
import pkg_resources
print(f"pkg_resources位置: {pkg_resources.__file__}")
print(f"pkg_resources版本: {pkg_resources.__version__}")

# 2. 尝试手动修复 _vendor 模块
try:
    # 尝试直接导入 _vendor
    from pkg_resources import _vendor
    print("✓ 成功导入 pkg_resources._vendor")
except ImportError as e:
    print(f"✗ 无法导入 _vendor: {e}")
    
    # 尝试其他路径
    try:
        # 有些版本中_vendor在pkg_resources.extern中
        import pkg_resources.extern
        pkg_resources._vendor = pkg_resources.extern
        print("✓ 通过extern修复_vendor")
    except Exception:
        print("✗ 无法通过extern修复")
        
        # 尝试直接设置一个虚拟模块
        class DummyVendor:
            pass
        
        pkg_resources._vendor = DummyVendor()
        
        # 创建packaging子模块
        pkg_resources._vendor.packaging = sys.modules.get('packaging')
        if pkg_resources._vendor.packaging is None:
            try:
                import packaging
                pkg_resources._vendor.packaging = packaging
                print("✓ 手动添加packaging模块")
            except ImportError:
                print("✗ 无法导入packaging，尝试安装")
                # 这里可以提示用户安装packaging
                pass

# 3. 测试修复是否成功
try:
    from pkg_resources._vendor.packaging.version import parse as parse_version
    print("✓ 成功导入 parse_version")
    # 保存到sys.modules以便其他模块使用
    sys.modules['pkg_resources._vendor.packaging.version'] = __import__('packaging.version')
except Exception as e:
    print(f"✗ 仍然无法导入: {e}")
    print("尝试直接安装packaging...")
    
    # 如果packaging不存在，需要安装
    import subprocess
    import importlib
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging==24.0"])
        import packaging.version
        pkg_resources._vendor.packaging = packaging
        print("✓ 已安装并导入packaging")
    except Exception as install_error:
        print(f"✗ 安装失败: {install_error}")

print("=" * 60)
print("修复完成")
print("=" * 60)

# 4. 验证修复
try:
    # 现在尝试导入ray
    import ray
    print(f"✓ 成功导入ray, 版本: {ray.__version__}")
    
    # 尝试导入tune
    from ray import tune
    print("✓ 成功导入ray.tune")
    
    # 尝试导入rllib相关模块
    from ray.rllib.algorithms.ppo import PPOConfig
    print("✓ 成功导入PPOConfig")
    
    from ray.tune.registry import register_env
    print("✓ 成功导入register_env")
    
except Exception as e:
    print(f"✗ 导入ray失败: {e}")
    print("可能需要重新安装ray或使用不同版本")
    print("建议: pip install 'ray[rllib]==2.5.1'")