"""测试 compute_lambda 函数"""
import sys
import os
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.compute_lambda import compute_lambda

def test_compute_lambda():
    """测试计算角向分离常数"""
    # 测试参数
    a = 0.1
    omega = 0.1
    l = 2
    m = 2

    print(f"测试参数: a={a}, ω={omega}, l={l}, m={m}")

    try:
        lambda_val = compute_lambda(a, omega, l, m)
        print(f"✓ 成功计算 λ = {lambda_val}")
        return True
    except Exception as e:
        print(f"✗ 计算失败: {e}")
        return False

if __name__ == "__main__":
    success = test_compute_lambda()
    sys.exit(0 if success else 1)
