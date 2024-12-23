import math
from math import comb

def fn(x, n):
    """
    Calculates the polynomial fn(x) based on the description.
    
    fn(x) = sum from i=0 to n of (1/4^i) * binomial(2i, i) * x * (1 - x^2)^i
    """
    result = 0
    for i in range(n + 1):
        term = (1 / (4 ** i)) * comb(2 * i, i) * (x * (1 - x**2) ** i)
        result += term
    return result

def newcomp(a, b, n, d):
    """
    Implements the NewComp algorithm to approximate whether a > b, a < b, or a = b.
    
    Parameters:
    - a: First input value in [0, 1]
    - b: Second input value in [0, 1]
    - n: Degree of fn(x)
    - d: Number of iterations
    
    Returns:
    - A value close to 1 if a > b, 0 if a < b, and 1/2 if a ≈ b.
    """
    # Step 1: Compute the difference x = a - b
    x = a - b
    
    # Step 2: Iterate d times to refine x using fn(x)
    for _ in range(d):
        x = fn(x, n)  # Compute fn(x)
    
    # Step 5: Return the final result as (x + 1) / 2
    return (x + 1) / 2

# Example usage
a = 1
b = 2
n = 1  # Degree of fn(x)
d = 3  # Number of iterations

result = newcomp(a, b, n, d)
print("NewComp result:", result)

import numpy as np

# Define the function for polynomial approximation
def polynomial_approximation(x, y, k):
    # Calculate the absolute difference
    diff = abs(x - y)
    # Compute the polynomial approximation
    return 1 - diff**k

# Test the function with some examples
x = 3
y = 3.5
k_values = [1, 2, 3]

# Demonstrate for different x, y, and k
for k in k_values:
    print(f"Polynomial Approximation with k={k}:")
    result = polynomial_approximation(x, y, k)
    print(f"f({x}, {y}) = {result:.4f}")
    print("\n")

