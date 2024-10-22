import random
import numpy as np
import pandas as pd

def gradient(A:np.array, b:np.array, x:np.array):
    # return 2 * np.dot(A.T, (np.dot(A, x) - b))

    return 2 * np.dot((np.dot(A, x) - b).T, A)

def learning_rate(A:np.array):
    # 0.012774369968774082
    return abs(0.5 / np.linalg.norm(A))

def get_A(file='HW2_ellipse.csv'):
    df = pd.read_csv(file)
    df.columns = ['x', 'y']
    x = df['x']
    y = df['y']
    x_squared = x ** 2
    y_squared = y ** 2
    # Create a matrix with x_squared as column one and y_squared as column two
    A = np.column_stack((x_squared, y_squared))
    return A

def print_result(A:np.array, b:np.array, x:np.array, dim:int=999):
    print("\nX värde : ", x, "\n")
    print("Function-value : ", function(A, b, x), "\n")
    
def function(A:np.array, b:np.array, x:np.array, dim:int=999):
    sum = 0
    for i in range(dim):
        sum += (A[i][0] * x[0] + A[i][1] * x[1] - 1) ** 2
    return sum


dim = 999
x = np.array([random.uniform(0, 1) for _ in range(2)])
A = get_A()
b = np.array([1 for _ in range(dim)])

# while np.linalg.norm(gradient(A, b, x)) > 1e-2:
for i in range(10):
    x -= learning_rate(A) * gradient(A, b, x)

print_result(A, b, x)