import math
from gradient_descent import gradient_descent

def test_cubic_min_1():
    """
    Test case 1: Minimize f(x) = x^3 - 3x^2 + 2x + 1
    """
    result = gradient_descent(a=1, b=-3, c=2, d=1, initial_x=1, learning_rate=0.01, num_iters=10000)
    assert math.isclose(result, 1.577, abs_tol=1e-3)

def test_cubic_min_2():
    """
    Test case 2: Minimize f(x) = 16x^3 + 2x^2 - x + 1
    """
    result = gradient_descent(a=16, b=2, c=-1, d=1, initial_x=0, learning_rate=0.01, num_iters=1000)
    assert math.isclose(result, 0.1085, abs_tol=1e-3)

def test_cubic_min_3():
    """
    Test case 3: Minimize f(x) = x^3 + 2x^2
    """
    result = gradient_descent(a=1, b=2, c=0, d=0, initial_x=0.5, learning_rate=0.01, num_iters=1000)
    assert math.isclose(result, 0, abs_tol=1e-3)
