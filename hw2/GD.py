import random
import numpy as np
import numpy.linalg as linalg
import pandas as pd

def gradient(A:np.array, b:np.array, x:np.array):
    return 2 * np.dot((np.dot(A, x) - b).T, A)

def learning_rate(A:np.array):
    return np.dot(A.T, A) * 0.5

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
    print("X värde : ")
    print(x, "\n\n")
    print("Function-value : ", function(A, b, x))
    

def get_b(dim=999):
    return np.array([1 for _ in range(dim)])

def function(A:np.array, b:np.array, x:np.array, dim:int=999):
    sum = 0
    for i in range(dim):
        sum += (A[i][0] * x[0] + A[i][1] * x[1] - 1) ** 2
    return sum

def found_min(x:np.array, stop_criteria=1e-10):
    for value in x:
        if value > stop_criteria:
            return False
    return True


dim = 999
x = np.array([random.uniform(0, 20) for _ in range(2)])
stop = True
A = np.array([[random.uniform(0, 20) for _ in range(2)] for _ in range(dim)])
b = np.array([1 for _ in range(dim)])


while found_min(gradient(A, b, x)):
    x -= learning_rate(A)

print_result(A, b, x)