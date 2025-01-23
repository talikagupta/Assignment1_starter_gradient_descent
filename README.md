# Homework 1: Gradient-Descent-Based Solver

## Objective
Write a Python program to implement a gradient descent solver that minimizes a cubic function of the form:

    f(x) = ax^3 + bx^2 + cx + d

## Instructions
1. Complete the `gradient_descent` function in `gradient_descent.py`.
2. The function should:
   - Initialize `x` with `initial_x`.
   - Iteratively update `x` using the gradient descent update rule for `num_iters` iterations.
   - Return the value of `x` after `num_iters` updates.

## Gradient Descent update rule
The update rule for gradient descent is:

    x_new = x - alpha * f'(x)

Where:
- `f'(x)` is the derivative of `f(x)`.
- `alpha` is the learning rate.

## Function Signature
```python
def gradient_descent(
    a: float, b: float, c: float, d: float, initial_x: float, learning_rate: float, num_iters: int
) -> float:
    """
    Perform gradient descent to minimize the function f(x) = ax^3 + bx^2 + cx + d.

    Parameters:
        a (float): Coefficient of x^3 in the cubic function.
        b (float): Coefficient of x^2 in the cubic function.
        c (float): Coefficient of x in the cubic function.
        d (float): Constant term in the cubic function.
        initial_x (float): Initial value for x.
        learning_rate (float): Learning rate for gradient descent.
        num_iters (int): Number of update steps.

    Returns:
        float: The value of x after num_iters iterations that minimizes f(x).
    """
