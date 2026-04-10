"""测试振幅比值计算函数"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.amplitude_ratio import compute_amplitude_ratio

def test_amplitude_ratio():
    """测试计算振幅比值"""
    # 测试参数
    a = 0.3
    omega = 0.2
    l = 2
    m = 2

    print(f"测试参数: a={a}, ω={omega}, l={l}, m={m}")

    try:
        result = compute_amplitude_ratio(a, omega, l, m)
        print(f"✓ 成功计算振幅比值")
        print(f"  λ = {result['lambda']}")
        print(f"  B_inc/B_ref = {result['ratio']}")
        print(f"  |B_inc/B_ref| = {result['ratio_abs']:.6e}")
        return True
    except Exception as e:
        print(f"✗ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_amplitude_ratio()
    sys.exit(0 if success else 1)
